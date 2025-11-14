#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import time
from contextlib import nullcontext
from pathlib import Path
from pprint import pformat
from typing import Any

import torch
import yaml
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

from lerobot.common.datasets.factory import make_dataset, resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.transforms import ImageTransforms
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy


def load_splits_yaml(raw_dataset_dir: Path) -> dict | None:
    """Load splits.yaml file and return split information.
    
    Expected format:
        train:
          - 0
          - 1
          - 2
        eval:
          - 10
          - 11
    """
    splits_path = raw_dataset_dir / "splits.yaml"
    logging.info(f"Loading splits.yaml from {splits_path}")
    if not splits_path.exists():
        logging.info(f"No splits.yaml found at {splits_path}, using full dataset for training")
        return None
    
    with open(splits_path, 'r') as f:
        splits = yaml.safe_load(f)
    
    return splits


def make_lerobot_dataset_with_episodes(cfg: TrainPipelineConfig, episodes: list[int]) -> LeRobotDataset:
    """Create a LeRobotDataset with specific episodes."""
    image_transforms = (
        ImageTransforms(cfg.dataset.image_transforms) if cfg.dataset.image_transforms.enable else None
    )

    ds_meta = LeRobotDatasetMetadata(
        cfg.dataset.repo_id, root=cfg.dataset.root, revision=cfg.dataset.revision
    )
    delta_timestamps = resolve_delta_timestamps(cfg.policy, ds_meta)
    return LeRobotDataset(
        cfg.dataset.repo_id,
        root=cfg.dataset.root,
        episodes=episodes,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        revision=cfg.dataset.revision,
        video_backend=cfg.dataset.video_backend,
    )


def eval_policy_on_dataset(
    policy: PreTrainedPolicy,
    eval_dataset: LeRobotDataset,
    device: torch.device,
    train_batch_size: int,
    use_amp: bool = False,
) -> dict:
    """Evaluate policy on an eval dataset by computing loss on eval data."""
    policy.eval()
    # Create dataloader for evaluation
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        # Can increase as long as it doesn't overfill GPU memory.
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=0,  # Use 0 workers to avoid issues with multiprocessing
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    
    total_loss = 0.0
    num_batches = 0
    total_loss_dict = {}
    
    with torch.no_grad():
        for batch in eval_dataloader:
            # Move batch to device
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            # Compute loss
            with torch.autocast(device_type=device.type) if use_amp else nullcontext():
                loss, loss_dict = policy.forward(batch)
            
            # The loss is already averaged within a batch.
            total_loss += loss.item()
            for k, v in loss_dict.items():
                if k not in total_loss_dict:
                    total_loss_dict[k] = 0.0
                total_loss_dict[k] += v
            num_batches += 1

    # Average across batches.
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    for k in total_loss_dict.keys():
        v = total_loss_dict[k]
        total_loss_dict[k] = v / num_batches if num_batches > 0 else float('inf')
    
    return {
        "eval_loss": avg_loss,
        **total_loss_dict,
    }


def save_checkpoint_manifest(
    output_dir: Path,
    step: int,
    train_loss: float,
    eval_loss: float | None,
    checkpoint_dir: str,
    checkpoint_reason: str,
) -> None:
    """Save checkpoint information to manifest.jsonl file."""
    manifest_path = output_dir / "manifest.jsonl"
    
    manifest_entry = {
        "step": step,
        "train_loss": train_loss,
        "eval_loss": eval_loss,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_reason": checkpoint_reason,
    }
    
    # Append to manifest file
    with open(manifest_path, "a") as f:
        f.write(json.dumps(manifest_entry) + "\n")


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    device = get_device_from_parameters(policy)
    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)
    grad_scaler.scale(loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    # Optimizer's gradients are already unscaled, so scaler.step does not unscale them,
    # although it still skips optimizer.step() if the gradients contain infs or NaNs.
    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    # Updates the scale for next iteration.
    grad_scaler.update()

    optimizer.zero_grad()

    # Step through pytorch scheduler at every batch instead of epoch
    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        # To possibly update an internal buffer (for instance an Exponential Moving Average like in TDMPC).
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict


@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    # Try to load splits.yaml to split dataset into train/eval
    dataset_root = Path(cfg.dataset.root) if cfg.dataset.root else None
    splits = None
    splits_file_path = None
    if dataset_root and dataset_root.exists():
        splits = load_splits_yaml(dataset_root)
        if splits is not None:
            splits_file_path = dataset_root / "splits.yaml"
    
    train_dataset = None
    eval_dataset = None
    
    if splits and "train" in splits and "eval" in splits:
        # Create separate train and eval datasets
        # Episode indices should already be integers in the splits.yaml
        train_episodes = splits["train"]
        eval_episodes = splits["eval"]
        
        logging.info(f"Found splits.yaml with {len(train_episodes)} train and {len(eval_episodes)} eval episodes")
        
        train_dataset = make_lerobot_dataset_with_episodes(cfg, train_episodes)
        logging.info(f"Train set (num_episodes={train_dataset.num_episodes}, num_frames={train_dataset.num_frames}): {train_episodes}")
        
        if cfg.eval_freq > 0:
            eval_dataset = make_lerobot_dataset_with_episodes(cfg, eval_episodes)
            logging.info(f"Eval set (num_episodes={eval_dataset.num_episodes}, num_frames={eval_dataset.num_frames}): {eval_episodes}")
        
        # Save a copy of splits.yaml to the output directory for reproducibility
        if splits_file_path and splits_file_path.exists():
            output_splits_path = cfg.output_dir / "splits.yaml"
            output_splits_path.write_text(splits_file_path.read_text())
            logging.info(f"Saved splits.yaml to {output_splits_path}")
        
        # Use train_dataset as the main dataset for compatibility
        dataset = train_dataset
    else:
        # No splits found, use full dataset for training
        dataset = make_dataset(cfg)
        train_dataset = dataset

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy")
    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=dataset.meta,
    )

    logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # Track best eval loss for saving best checkpoints
    best_eval_loss = float('inf')

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = (cfg.eval_freq > 0 and step % cfg.eval_freq == 0) or is_saving_step

        current_train_loss = train_tracker.loss.val

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # Track eval loss for checkpoint manifest
        current_eval_loss = None
        
        if is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            
            # Environment-based evaluation (if env is available)
            if cfg.env:
                with (
                    torch.no_grad(),
                    torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
                ):
                    eval_info = eval_policy(
                        eval_env,
                        policy,
                        cfg.eval.n_episodes,
                        videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                        max_episodes_rendered=4,
                        start_seed=cfg.seed,
                    )

                eval_metrics = {
                    "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                    "pc_success": AverageMeter("success", ":.1f"),
                    "eval_s": AverageMeter("eval_s", ":.3f"),
                }
                eval_tracker = MetricsTracker(
                    cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
                )
                eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
                eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
                eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
                logging.info(eval_tracker)
                if wandb_logger:
                    wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                    wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")
            
            # Dataset-based evaluation (if eval_dataset is available)
            elif eval_dataset is not None:
                logging.info(f"Running evaluation on hold-out dataset with {eval_dataset.num_episodes} episodes ({eval_dataset.num_frames} frames)")
                start_time = time.perf_counter()
                eval_results = eval_policy_on_dataset(
                    policy,
                    eval_dataset,
                    device,
                    train_batch_size=cfg.batch_size,
                    use_amp=cfg.policy.use_amp,
                )
                eval_time = time.perf_counter() - start_time
                current_eval_loss = eval_results["eval_loss"]
                
                logging.info(f"Hold-out dataset eval loss: {current_eval_loss:.4f} (time: {eval_time:.2f}s)")
                
                if wandb_logger:
                    dataset_eval_dict = {
                        "dataset_eval_time": eval_time,
                        **eval_results
                    }
                    wandb_logger.log_dict(dataset_eval_dict, step, mode="eval")

        # Checkpointing logic - save checkpoint if scheduled OR if eval loss improved
        should_save_checkpoint = False
        checkpoint_reason = ""
        
        if cfg.save_checkpoint:
            if is_saving_step:
                should_save_checkpoint = True
                checkpoint_reason = "scheduled"
            elif current_eval_loss is not None and current_eval_loss < best_eval_loss:
                should_save_checkpoint = True
                checkpoint_reason = "best_eval_loss"
                best_eval_loss = current_eval_loss
                
        if should_save_checkpoint:
            logging.info(f"Checkpoint policy after step {step} ({checkpoint_reason})")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            save_checkpoint(checkpoint_dir, step, cfg, policy, optimizer, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)
            
            # Save checkpoint manifest
            save_checkpoint_manifest(
                cfg.output_dir,
                step,
                current_train_loss,
                current_eval_loss,
                checkpoint_dir.name,
                checkpoint_reason,
            )

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
