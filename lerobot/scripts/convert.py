# import cv2
# from pathlib import Path
# from lerobot.common.datasets.video_utils import decode_video_frames
import json

import numpy as np
import torch

from lerobot.common.constants import HF_LEROBOT_HOME

# from lerobot.common.datasets.utils import load_info
# from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# # ---- CONFIGURATION ----
# REPO_ID = "TrossenRoboticsCommunity/trossen_ai_stationary_peg_insertion"           # Replace with your actual repo_id
# EPISODE_INDEX = 0                          # Choose the episode to play
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# === Step 1: Load dataset ===
repo_id = "EVAL_TX_13/eval_act_trossen_ai_stationary_test"
local_root = HF_LEROBOT_HOME / "EVAL_TX_03/eval_act_trossen_ai_stationary_test"  # Can be None to use cache
load_from_hub = False  # True to load from hub

# Load total_episodes from info.json
meta_folder = local_root / "meta"
info_file = meta_folder / "info.json"
total_episodes = 0


if info_file.exists():
    with open(info_file) as f:
        info_data = json.load(f)
        total_episodes = info_data.get("total_episodes", 0)
else:
    raise FileNotFoundError(f"info.json not found in {meta_folder}")


# Path to the episodes_stats.jsonl file
episodes_stats_path = meta_folder / "episodes_stats.jsonl"

# Load the existing stats
if not episodes_stats_path.exists():
    raise FileNotFoundError(f"{episodes_stats_path} not found.")

with open(episodes_stats_path) as f:
    episodes_stats = [json.loads(line) for line in f]


def modify_action_or_state(array):
    """Common transformation logic for both action and observation.state."""
    half = len(array) // 2
    first_half = np.copy(array[:half])
    second_half = np.copy(array[half:])

    # Convert to radians except the last element of each half
    first_half[:-1] = np.deg2rad(first_half[:-1])
    second_half[:-1] = np.deg2rad(second_half[:-1])

    # Scale the last element
    first_half[-1] /= 10000
    second_half[-1] /= 10000

    return np.concatenate([first_half, second_half])


def compute_stats(data_list):
    data = np.array(data_list)
    return {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
        "count": [len(data)],
    }


# Start dataset edit loop
for episode_index in range(total_episodes):
    dataset = (
        LeRobotDataset(repo_id=repo_id, root=local_root, episodes=[episode_index])
        if not load_from_hub
        else LeRobotDataset(repo_id=repo_id, root=None, force_cache_sync=False)
    )

    modified_actions = []
    modified_states = []

    def modify_entry(entry, actions_list, states_list):
        modified_action = modify_action_or_state(entry["action"])
        modified_state = modify_action_or_state(entry["observation.state"])

        entry["action"] = torch.tensor(modified_action, dtype=torch.float32)
        entry["observation.state"] = torch.tensor(modified_state, dtype=torch.float32)

        actions_list.append(modified_action)
        states_list.append(modified_state)

        return entry

    def map_fn(entry, actions_list=modified_actions, states_list=modified_states):
        return modify_entry(entry, actions_list, states_list)

    dataset.hf_dataset = dataset.hf_dataset.map(map_fn)
    # Update parquet file
    output_path = dataset.root / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
    dataset.hf_dataset.to_parquet(str(output_path))
    print(f"Saved modified dataset to: {output_path}")

    # Update stats
    new_action_stats = compute_stats(modified_actions)
    new_state_stats = compute_stats(modified_states)

    for ep_stat in episodes_stats:
        if ep_stat["episode_index"] == episode_index:
            ep_stat["stats"]["action"] = new_action_stats
            ep_stat["stats"]["observation.state"] = new_state_stats
            break

# Save updated stats
with open(episodes_stats_path, "w") as f:
    for ep_stat in episodes_stats:
        f.write(json.dumps(ep_stat) + "\n")

print(f"Updated episodes_stats saved to: {episodes_stats_path}")
