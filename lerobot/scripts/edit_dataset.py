#!/usr/bin/env python
"""
Edit LeRobot datasets: delete episodes, merge datasets, and validate integrity.

Dataset paths use the format ${HF_USER}/<dataset_name> and are resolved relative
to --root (default: ~/.cache/huggingface/lerobot). For merge, each input dataset
can have its own root via --root-a / --root-b.

Setup:
    HF_USER=$(huggingface-cli whoami | head -n 1)

Usage:
    # Validate a dataset
    python lerobot/scripts/edit_dataset.py validate \
        --dataset-dir ${HF_USER}/trossen_ai_solo_dataset

    # Delete episodes 2, 5, and 7
    python lerobot/scripts/edit_dataset.py delete \
        --dataset-dir ${HF_USER}/trossen_ai_solo_dataset \
        --episodes 2 5 7 \
        --output-dir ${HF_USER}/trossen_ai_solo_dataset_trimmed

    # Merge two datasets
    python lerobot/scripts/edit_dataset.py merge \
        --dataset-a ${HF_USER}/trossen_ai_solo_dataset_a \
        --dataset-b ${HF_USER}/trossen_ai_solo_dataset_b \
        --output-dir ${HF_USER}/trossen_ai_solo_merged \
        --task "pick and place" \
        --robot-type trossen_solo_ai

    # Merge datasets from different locations
    python lerobot/scripts/edit_dataset.py merge \
        --root-a /data/lab1 \
        --root-b /data/lab2 \
        --dataset-a ${HF_USER}/trossen_ai_solo_dataset \
        --dataset-b ${HF_USER}/trossen_ai_solo_dataset \
        --output-dir ${HF_USER}/trossen_ai_solo_merged

    # Override the default root for all paths
    python lerobot/scripts/edit_dataset.py validate \
        --root /data/my_datasets \
        --dataset-dir ${HF_USER}/trossen_ai_solo_dataset

    # Run any command from a JSON config file:
    python lerobot/scripts/edit_dataset.py --config config.json

    # Example merge config (merge_config.json):
    {
        "command": "merge",
        "dataset_a": "${HF_USER}/trossen_ai_solo_dataset_a",
        "dataset_b": "${HF_USER}/trossen_ai_solo_dataset_b",
        "output_dir": "${HF_USER}/trossen_ai_solo_merged",
        "task": "pick and place",
        "robot_type": "trossen_solo_ai",
        "fps": 30
    }

All episode indices, global frame indices, chunk directories, parquet files,
video files, image directories, and metadata (info.json, episodes.jsonl,
tasks.jsonl, episodes_stats.jsonl) are renumbered and rewritten consistently.
"""

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import (
    aggregate_stats,
    compute_episode_stats,
)
from lerobot.common.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    DEFAULT_PARQUET_PATH,
    DEFAULT_VIDEO_PATH,
    EPISODES_PATH,
    EPISODES_STATS_PATH,
    INFO_PATH,
    STATS_PATH,
    TASKS_PATH,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    serialize_dict,
    write_info,
    write_json,
    write_jsonlines,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_path(path: Path, root: Path | None = None) -> Path:
    """Resolve a dataset path. If the path is not absolute, treat it as
    relative to *root* (which defaults to ``HF_LEROBOT_HOME``,
    i.e. ``~/.cache/huggingface/lerobot``)."""
    path = Path(path)
    if path.is_absolute():
        return path
    root = Path(root) if root is not None else HF_LEROBOT_HOME
    return root / path


def _episode_chunk(episode_index: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> int:
    return episode_index // chunks_size


def _parquet_path(root: Path, ep_idx: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> Path:
    return root / DEFAULT_PARQUET_PATH.format(
        episode_chunk=_episode_chunk(ep_idx, chunks_size),
        episode_index=ep_idx,
    )


def _video_path(root: Path, video_key: str, ep_idx: int, chunks_size: int = DEFAULT_CHUNK_SIZE) -> Path:
    return root / DEFAULT_VIDEO_PATH.format(
        episode_chunk=_episode_chunk(ep_idx, chunks_size),
        video_key=video_key,
        episode_index=ep_idx,
    )


def _image_dir(root: Path, image_key: str, ep_idx: int) -> Path:
    """Return the directory holding frames for one episode of an image key."""
    return root / f"images/{image_key}/episode_{ep_idx:06d}"


def _get_video_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v["dtype"] == "video"]


def _get_image_keys(features: dict) -> list[str]:
    return [k for k, v in features.items() if v["dtype"] == "image"]


def _reindex_parquet(src_path: Path, dst_path: Path, new_ep_idx: int,
                     global_index_start: int, video_keys: list[str],
                     image_keys: list[str], chunks_size: int):
    """Read a parquet file, rewrite index columns & media paths, save to dst."""
    table = pq.read_table(src_path)
    num_rows = table.num_rows

    # Build replacement columns
    replacements = {
        "episode_index": pa.array([new_ep_idx] * num_rows, type=pa.int64()),
        "index": pa.array(list(range(global_index_start, global_index_start + num_rows)), type=pa.int64()),
    }

    # If frame_index doesn't start at 0 we fix it (shouldn't happen normally)
    frame_indices = table.column("frame_index").to_pylist()
    if frame_indices and frame_indices[0] != 0:
        replacements["frame_index"] = pa.array(list(range(num_rows)), type=pa.int64())

    # Update video path columns
    new_chunk = _episode_chunk(new_ep_idx, chunks_size)
    for vk in video_keys:
        if vk in table.column_names:
            new_vpath = DEFAULT_VIDEO_PATH.format(
                episode_chunk=new_chunk, video_key=vk, episode_index=new_ep_idx
            )
            replacements[vk] = pa.array([new_vpath] * num_rows, type=pa.string())

    # Update image path columns
    for ik in image_keys:
        if ik in table.column_names:
            new_paths = [
                DEFAULT_IMAGE_PATH.format(
                    image_key=ik, episode_index=new_ep_idx, frame_index=fi
                )
                for fi in range(num_rows)
            ]
            replacements[ik] = pa.array(new_paths, type=pa.string())

    # Rebuild the table preserving column order
    columns = []
    for name in table.column_names:
        if name in replacements:
            columns.append(replacements[name])
        else:
            columns.append(table.column(name))

    new_table = pa.table({name: col for name, col in zip(table.column_names, columns)},
                         schema=table.schema)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, dst_path)
    return num_rows


def _update_task_index_in_parquet(parquet_path: Path, ep_info: dict, task_to_new_index: dict):
    """Rewrite task_index column in a parquet file based on new task mapping."""
    table = pq.read_table(parquet_path)
    if "task_index" not in table.column_names:
        return

    ep_tasks = ep_info.get("tasks", [])
    if ep_tasks and ep_tasks[0] in task_to_new_index:
        new_task_idx = task_to_new_index[ep_tasks[0]]
    else:
        new_task_idx = 0
        if ep_tasks:
            logger.warning(
                f"Task '{ep_tasks[0]}' not found in task mapping; defaulting to task_index=0"
            )

    num_rows = table.num_rows
    new_col = pa.array([new_task_idx] * num_rows, type=pa.int64())

    col_idx = table.column_names.index("task_index")
    table = table.set_column(col_idx, "task_index", new_col)
    pq.write_table(table, parquet_path)


def _write_all_meta(output_dir: Path, info: dict, episodes: list[dict],
                    tasks: dict, episodes_stats: dict | None, features: dict):
    """Write all metadata files to output_dir/meta/."""
    write_info(info, output_dir)

    # tasks.jsonl
    tasks_list = [{"task_index": idx, "task": task} for idx, task in sorted(tasks.items())]
    write_jsonlines(tasks_list, output_dir / TASKS_PATH)

    # episodes.jsonl
    write_jsonlines(episodes, output_dir / EPISODES_PATH)

    # episodes_stats.jsonl
    if episodes_stats:
        stats_list = []
        for ep_idx, stats in sorted(episodes_stats.items()):
            stats_list.append({"episode_index": ep_idx, "stats": serialize_dict(stats)})
        write_jsonlines(stats_list, output_dir / EPISODES_STATS_PATH)

    # Aggregate stats -> stats.json
    if episodes_stats:
        all_ep_stats = list(episodes_stats.values())
        if all_ep_stats:
            agg = aggregate_stats(all_ep_stats)
            serialized = serialize_dict(agg)
            write_json(serialized, output_dir / STATS_PATH)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def _format_stat_change(label: str, old_val, new_val) -> str:
    """Format a single stat comparison line, e.g. 'min: [0.1, 0.2] -> [0.1, 0.3]'."""
    def _fmt(v):
        if isinstance(v, np.ndarray):
            if v.size <= 8:
                return np.array2string(v, precision=4, separator=", ", suppress_small=True)
            return f"array(shape={v.shape})"
        return str(v)
    return f"      {label}: {_fmt(old_val)} -> {_fmt(new_val)}"


def _print_delete_summary(
    dataset_dir: Path,
    output_dir: Path,
    in_place: bool,
    deleted_eps: list[int],
    old_info: dict,
    new_info: dict,
    old_episodes: dict,
    new_episodes: list[dict],
    old_tasks: dict,
    new_tasks: dict,
    episode_remap: dict[int, int],
    old_stats: dict | None,
    new_stats: dict | None,
):
    deleted_frames = sum(old_episodes[e]["length"] for e in deleted_eps)
    deleted_tasks = set(old_tasks.values()) - set(new_tasks.values())

    lines = [
        "",
        "=" * 64,
        " DELETE SUMMARY",
        "=" * 64,
        "",
        "  Source      : " + str(dataset_dir),
    ]
    if in_place:
        lines.append("  Output      : (in-place)")
        lines.append("")
        lines.append("  !! WARNING: The original dataset was overwritten.")
        lines.append("     This change CANNOT be recovered.")
    else:
        lines.append("  Output      : " + str(output_dir))
        lines.append("")
        lines.append("  The original dataset is unchanged at the source path.")

    lines += [
        "",
        "  EPISODES DELETED:",
    ]
    for ep_idx in sorted(deleted_eps):
        ep = old_episodes[ep_idx]
        lines.append(
            f"    Episode {ep_idx}  ({ep['length']} frames, "
            f"tasks={ep.get('tasks', [])})"
        )

    lines += [
        "",
        "  EPISODE RENUMBERING:",
    ]
    for old_idx, new_idx in sorted(episode_remap.items()):
        length = old_episodes[old_idx]["length"]
        if old_idx == new_idx:
            lines.append(f"    {old_idx} -> {new_idx}  ({length} frames, unchanged)")
        else:
            lines.append(f"    {old_idx} -> {new_idx}  ({length} frames)")

    lines += [
        "",
        f"  {'':20s} {'BEFORE':>12s}  {'AFTER':>12s}  {'CHANGE':>12s}",
        f"  {'Episodes':20s} {old_info['total_episodes']:>12d}  {new_info['total_episodes']:>12d}  {new_info['total_episodes'] - old_info['total_episodes']:>+12d}",
        f"  {'Frames':20s} {old_info['total_frames']:>12d}  {new_info['total_frames']:>12d}  {new_info['total_frames'] - old_info['total_frames']:>+12d}",
        f"  {'Deleted frames':20s} {'':>12s}  {'':>12s}  {-deleted_frames:>+12d}",
        f"  {'Videos':20s} {old_info.get('total_videos', 0):>12d}  {new_info.get('total_videos', 0):>12d}  {new_info.get('total_videos', 0) - old_info.get('total_videos', 0):>+12d}",
        f"  {'Tasks':20s} {old_info['total_tasks']:>12d}  {new_info['total_tasks']:>12d}  {new_info['total_tasks'] - old_info['total_tasks']:>+12d}",
        f"  {'Chunks':20s} {old_info.get('total_chunks', 0):>12d}  {new_info.get('total_chunks', 0):>12d}  {new_info.get('total_chunks', 0) - old_info.get('total_chunks', 0):>+12d}",
    ]

    if deleted_tasks:
        lines += ["", "  TASKS REMOVED (no longer referenced by any episode):"]
        for t in sorted(deleted_tasks):
            lines.append(f"    - '{t}'")

    # Stats changes
    if old_stats and new_stats:
        lines += ["", "  STATS CHANGES (aggregated):"]
        for key in sorted(set(old_stats.keys()) | set(new_stats.keys())):
            if key not in old_stats:
                lines.append(f"    {key}: (new)")
                continue
            if key not in new_stats:
                lines.append(f"    {key}: (removed)")
                continue
            changes = []
            for stat_name in ["min", "max", "mean", "std", "count"]:
                old_v = old_stats[key].get(stat_name)
                new_v = new_stats[key].get(stat_name)
                if old_v is not None and new_v is not None:
                    if not np.allclose(old_v, new_v, rtol=1e-4, atol=1e-6, equal_nan=True):
                        changes.append(_format_stat_change(stat_name, old_v, new_v))
            if changes:
                lines.append(f"    {key}:")
                lines.extend(changes)

    lines += ["", "=" * 64]
    logger.info("\n".join(lines))


def _print_merge_summary(
    dataset_a_dir: Path,
    dataset_b_dir: Path,
    output_dir: Path,
    info_a: dict,
    info_b: dict,
    new_info: dict,
    episodes_a: dict,
    episodes_b: dict,
    new_episodes: list[dict],
    old_tasks_a: dict,
    old_tasks_b: dict,
    new_tasks: dict,
    task_override: str | None,
    robot_type_override: str | None,
    fps_override: int | None,
    old_stats_a: dict | None,
    old_stats_b: dict | None,
    new_stats: dict | None,
):
    num_a = len(episodes_a)
    num_b = len(episodes_b)
    frames_a = sum(ep["length"] for ep in episodes_a.values())
    frames_b = sum(ep["length"] for ep in episodes_b.values())
    total_eps = num_a + num_b
    total_frames = frames_a + frames_b

    lines = [
        "",
        "=" * 64,
        " MERGE SUMMARY",
        "=" * 64,
        "",
        "  Dataset A   : " + str(dataset_a_dir),
        "  Dataset B   : " + str(dataset_b_dir),
        "  Output      : " + str(output_dir),
        "",
        "  Source datasets are unchanged. Only the output is new.",
        "",
        "  EPISODE MAPPING:",
        f"    Dataset A episodes 0..{num_a - 1}  ->  output 0..{num_a - 1}  ({frames_a} frames)",
        f"    Dataset B episodes 0..{num_b - 1}  ->  output {num_a}..{total_eps - 1}  ({frames_b} frames)",
        "",
        f"  {'':20s} {'DATASET A':>12s}  {'DATASET B':>12s}  {'MERGED':>12s}",
        f"  {'Episodes':20s} {info_a['total_episodes']:>12d}  {info_b['total_episodes']:>12d}  {new_info['total_episodes']:>12d}",
        f"  {'Frames':20s} {info_a['total_frames']:>12d}  {info_b['total_frames']:>12d}  {new_info['total_frames']:>12d}",
        f"  {'Videos':20s} {info_a.get('total_videos', 0):>12d}  {info_b.get('total_videos', 0):>12d}  {new_info.get('total_videos', 0):>12d}",
        f"  {'Tasks':20s} {info_a['total_tasks']:>12d}  {info_b['total_tasks']:>12d}  {new_info['total_tasks']:>12d}",
        f"  {'Chunks':20s} {info_a.get('total_chunks', 0):>12d}  {info_b.get('total_chunks', 0):>12d}  {new_info.get('total_chunks', 0):>12d}",
    ]

    # Overrides applied
    overrides = []
    overrides.append(f"    dataset_name : '{new_info.get('repo_id', '?')}'  (from output dir)")
    if task_override:
        overrides.append(f"    task         : '{task_override}'  (all episodes)")
    if robot_type_override:
        old_rt = info_a.get("robot_type", "?")
        overrides.append(f"    robot_type   : '{old_rt}' -> '{robot_type_override}'")
    if fps_override:
        old_fps = info_a.get("fps", "?")
        overrides.append(f"    fps          : {old_fps} -> {fps_override}")

    if overrides:
        lines += ["", "  OVERRIDES APPLIED:"]
        lines.extend(overrides)

    # Task changes
    tasks_a_set = set(old_tasks_a.values())
    tasks_b_set = set(old_tasks_b.values())
    tasks_new_set = set(new_tasks.values())
    if task_override:
        lines += [
            "", "  TASK OVERRIDE:",
            f"    Original A tasks: {sorted(tasks_a_set)}",
            f"    Original B tasks: {sorted(tasks_b_set)}",
            f"    Merged task:      ['{task_override}']",
            "",
            "    !! WARNING: Original per-episode task labels have been discarded.",
            "       This cannot be recovered from the merged output.",
        ]
    else:
        new_tasks_from_b = tasks_b_set - tasks_a_set
        if new_tasks_from_b:
            lines += ["", f"  NEW TASKS FROM B: {sorted(new_tasks_from_b)}"]

    # Stats comparison
    if new_stats:
        lines += ["", "  AGGREGATED STATS (merged):"]
        for key in sorted(new_stats.keys()):
            st = new_stats[key]
            count = st.get("count")
            count_str = f", count={int(count[0])}" if count is not None else ""
            mean = st.get("mean")
            mean_str = ""
            if mean is not None:
                if mean.size <= 8:
                    mean_str = f", mean={np.array2string(mean, precision=4, separator=', ', suppress_small=True)}"
                else:
                    mean_str = f", mean=array(shape={mean.shape})"
            lines.append(f"    {key}{count_str}{mean_str}")

    lines += ["", "=" * 64]
    logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Stats recomputation
# ---------------------------------------------------------------------------

def _extract_video_frame_paths(dataset_dir: Path, video_key: str, ep_idx: int,
                               num_frames: int, fps: int,
                               chunks_size: int) -> list[str]:
    """Extract frames from a video file to a temp dir and return their paths.

    Falls back to returning an empty list if ffmpeg is not available or the
    video file doesn't exist.
    """
    import subprocess
    import tempfile

    vid = _video_path(dataset_dir, video_key, ep_idx, chunks_size)
    if not vid.exists():
        logger.warning(f"Video not found for stats: {vid}")
        return []

    tmp = Path(tempfile.mkdtemp(prefix=f"lerobot_stats_ep{ep_idx:06d}_{video_key}_"))
    pattern = str(tmp / "frame_%06d.png")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(vid), "-q:v", "2", pattern],
            capture_output=True, check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.warning(f"ffmpeg frame extraction failed for {vid}: {e}")
        return []

    paths = sorted(tmp.glob("frame_*.png"))
    return [str(p) for p in paths]


def _build_episode_data_from_parquet(dataset_dir: Path, ep_idx: int, features: dict,
                                     chunks_size: int, fps: int) -> dict:
    """Read a parquet file and reconstruct the episode_data dict that
    ``compute_episode_stats`` expects.

    For numeric features   -> np.ndarray
    For image features     -> list of image file paths (absolute)
    For video features     -> list of image file paths (extracted from mp4)
    """
    pq_path = _parquet_path(dataset_dir, ep_idx, chunks_size)
    if not pq_path.exists():
        raise FileNotFoundError(f"Parquet not found: {pq_path}")

    table = pq.read_table(pq_path)
    num_frames = table.num_rows
    episode_data: dict = {}
    _tmp_dirs: list[str] = []  # track for cleanup later

    for key, ft in features.items():
        if key in DEFAULT_FEATURES:
            continue
        dtype = ft["dtype"]

        if dtype == "video":
            paths = _extract_video_frame_paths(
                dataset_dir, key, ep_idx, num_frames, fps, chunks_size
            )
            if paths:
                # Remember temp dir for cleanup
                _tmp_dirs.append(str(Path(paths[0]).parent))
            episode_data[key] = paths

        elif dtype == "image":
            if key in table.column_names:
                # Column stores relative paths; make absolute
                rel_paths = table.column(key).to_pylist()
                episode_data[key] = [str(dataset_dir / p) for p in rel_paths]
            else:
                img_dir = _image_dir(dataset_dir, key, ep_idx)
                if img_dir.exists():
                    episode_data[key] = sorted(str(p) for p in img_dir.glob("frame_*.png"))
                else:
                    episode_data[key] = []

        elif dtype == "string":
            continue

        else:
            if key in table.column_names:
                col = table.column(key)
                episode_data[key] = np.stack(col.to_pylist()).astype(dtype)
            else:
                logger.warning(f"Feature '{key}' not found in parquet columns")

    # Attach tmp_dirs list so caller can clean up
    episode_data["_tmp_dirs"] = _tmp_dirs
    return episode_data


def recompute_stats(dataset_dir: Path):
    """Recompute episodes_stats.jsonl and stats.json from the actual data on disk."""
    dataset_dir = Path(dataset_dir)
    info = load_info(dataset_dir)
    features = info["features"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    fps = info.get("fps", 30)
    episodes_dict = load_episodes(dataset_dir)

    logger.info(f"Recomputing stats for {len(episodes_dict)} episodes ...")

    new_ep_stats = {}
    for ep_idx in sorted(episodes_dict.keys()):
        logger.info(f"  Computing stats for episode {ep_idx} ...")
        try:
            ep_data = _build_episode_data_from_parquet(
                dataset_dir, ep_idx, features, chunks_size, fps
            )
            tmp_dirs = ep_data.pop("_tmp_dirs", [])
            ep_stats = compute_episode_stats(ep_data, features)
            new_ep_stats[ep_idx] = ep_stats

            # Clean up temp video frames
            for td in tmp_dirs:
                shutil.rmtree(td, ignore_errors=True)
        except Exception as e:
            logger.error(f"  Failed to compute stats for episode {ep_idx}: {e}")

    # Write episodes_stats.jsonl
    stats_list = []
    for ep_idx, stats in sorted(new_ep_stats.items()):
        stats_list.append({"episode_index": ep_idx, "stats": serialize_dict(stats)})
    write_jsonlines(stats_list, dataset_dir / EPISODES_STATS_PATH)

    # Write aggregated stats.json
    if new_ep_stats:
        agg = aggregate_stats(list(new_ep_stats.values()))
        write_json(serialize_dict(agg), dataset_dir / STATS_PATH)

    logger.info(f"Stats recomputed and written for {len(new_ep_stats)} episodes")
    return new_ep_stats


# ---------------------------------------------------------------------------
# Validate dataset
# ---------------------------------------------------------------------------

def validate_dataset(dataset_dir: Path, root: Path | None = None) -> bool:
    """Run comprehensive validation checks on a dataset.

    Returns True if all checks pass, False otherwise.
    Logs every issue found so the user gets a full report.
    Relative paths are resolved against *root* (default: ``~/.cache/huggingface/lerobot``).
    """
    dataset_dir = _resolve_path(dataset_dir, root)
    errors: list[str] = []
    warnings: list[str] = []

    def err(msg: str):
        errors.append(msg)
        logger.error(f"  FAIL: {msg}")

    def warn(msg: str):
        warnings.append(msg)
        logger.warning(f"  WARN: {msg}")

    logger.info(f"Validating dataset at {dataset_dir} ...")

    # ------------------------------------------------------------------
    # 1. Meta files exist
    # ------------------------------------------------------------------
    logger.info("[1/8] Checking meta files exist ...")
    for name, rel in [("info.json", INFO_PATH), ("episodes.jsonl", EPISODES_PATH),
                      ("tasks.jsonl", TASKS_PATH)]:
        if not (dataset_dir / rel).exists():
            err(f"Missing required meta file: {rel}")

    has_ep_stats = (dataset_dir / EPISODES_STATS_PATH).exists()
    has_stats = (dataset_dir / STATS_PATH).exists()
    if not has_ep_stats and not has_stats:
        warn("Neither episodes_stats.jsonl nor stats.json found — stats are missing")

    if errors:
        # Can't continue without core meta files
        logger.error(f"Validation aborted: {len(errors)} critical errors")
        return False

    # ------------------------------------------------------------------
    # 2. Load metadata
    # ------------------------------------------------------------------
    info = load_info(dataset_dir)
    features = info["features"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    episodes_dict = load_episodes(dataset_dir)
    tasks, _ = load_tasks(dataset_dir)
    video_keys = _get_video_keys(features)
    image_keys = _get_image_keys(features)

    # ------------------------------------------------------------------
    # 3. info.json totals vs actual counts
    # ------------------------------------------------------------------
    logger.info("[2/8] Checking info.json totals ...")

    actual_total_eps = len(episodes_dict)
    if info["total_episodes"] != actual_total_eps:
        err(f"info.json total_episodes={info['total_episodes']} but episodes.jsonl has {actual_total_eps} entries")

    actual_total_tasks = len(tasks)
    if info["total_tasks"] != actual_total_tasks:
        err(f"info.json total_tasks={info['total_tasks']} but tasks.jsonl has {actual_total_tasks} entries")

    actual_total_frames = sum(ep["length"] for ep in episodes_dict.values())
    if info["total_frames"] != actual_total_frames:
        err(f"info.json total_frames={info['total_frames']} but sum of episode lengths={actual_total_frames}")

    expected_chunks = (math.ceil(actual_total_eps / chunks_size)) if actual_total_eps > 0 else 0
    if info.get("total_chunks", expected_chunks) != expected_chunks:
        warn(f"info.json total_chunks={info.get('total_chunks')} but expected {expected_chunks}")

    expected_total_videos = actual_total_eps * len(video_keys)
    if info.get("total_videos", expected_total_videos) != expected_total_videos:
        err(f"info.json total_videos={info.get('total_videos')} but expected {expected_total_videos}")

    # ------------------------------------------------------------------
    # 4. Episode index contiguity
    # ------------------------------------------------------------------
    logger.info("[3/8] Checking episode index contiguity ...")
    ep_indices = sorted(episodes_dict.keys())
    expected_indices = list(range(len(ep_indices)))
    if ep_indices != expected_indices:
        err(f"Episode indices are not contiguous 0..{len(ep_indices)-1}. "
            f"Found: {ep_indices[:10]}{'...' if len(ep_indices) > 10 else ''}")

    # ------------------------------------------------------------------
    # 5. Parquet files exist and have correct schema
    # ------------------------------------------------------------------
    logger.info("[4/8] Checking parquet files ...")
    global_idx = 0
    for ep_idx in ep_indices:
        ep_info = episodes_dict[ep_idx]
        pq_path = _parquet_path(dataset_dir, ep_idx, chunks_size)
        if not pq_path.exists():
            err(f"Missing parquet: {pq_path.relative_to(dataset_dir)}")
            global_idx += ep_info["length"]
            continue

        table = pq.read_table(pq_path)
        num_rows = table.num_rows

        # Length matches episodes.jsonl
        if num_rows != ep_info["length"]:
            err(f"Episode {ep_idx}: parquet has {num_rows} rows but episodes.jsonl says length={ep_info['length']}")

        # Required columns present
        for col in ["timestamp", "frame_index", "episode_index", "index", "task_index"]:
            if col not in table.column_names:
                err(f"Episode {ep_idx}: missing required column '{col}' in parquet")

        # episode_index column is consistent
        if "episode_index" in table.column_names:
            ep_col = table.column("episode_index").to_pylist()
            if any(v != ep_idx for v in ep_col):
                err(f"Episode {ep_idx}: episode_index column contains values other than {ep_idx}")

        # frame_index is 0..N-1
        if "frame_index" in table.column_names:
            fi_col = table.column("frame_index").to_pylist()
            if fi_col != list(range(num_rows)):
                err(f"Episode {ep_idx}: frame_index not contiguous 0..{num_rows-1}")

        # global index continuity
        if "index" in table.column_names:
            idx_col = table.column("index").to_pylist()
            expected_range = list(range(global_idx, global_idx + num_rows))
            if idx_col != expected_range:
                err(f"Episode {ep_idx}: global index mismatch — expected {global_idx}..{global_idx+num_rows-1}, "
                    f"got {idx_col[0]}..{idx_col[-1]}")

        # task_index values are valid
        if "task_index" in table.column_names:
            ti_col = set(table.column("task_index").to_pylist())
            invalid_ti = ti_col - set(tasks.keys())
            if invalid_ti:
                err(f"Episode {ep_idx}: task_index values {invalid_ti} not in tasks.jsonl")

        # Check feature columns exist (non-video non-default)
        for key, ft in features.items():
            if key in DEFAULT_FEATURES or ft["dtype"] == "video":
                continue
            if key not in table.column_names:
                warn(f"Episode {ep_idx}: feature '{key}' not found in parquet columns")

        global_idx += num_rows

    # ------------------------------------------------------------------
    # 6. Video files exist
    # ------------------------------------------------------------------
    logger.info("[5/8] Checking video files ...")
    for ep_idx in ep_indices:
        for vk in video_keys:
            vid = _video_path(dataset_dir, vk, ep_idx, chunks_size)
            if not vid.exists():
                err(f"Missing video: {vid.relative_to(dataset_dir)}")

    # ------------------------------------------------------------------
    # 7. Image directories (if image features exist)
    # ------------------------------------------------------------------
    logger.info("[6/8] Checking image files ...")
    for ep_idx in ep_indices:
        ep_info = episodes_dict[ep_idx]
        for ik in image_keys:
            img_d = _image_dir(dataset_dir, ik, ep_idx)
            if not img_d.exists():
                err(f"Missing image dir: {img_d.relative_to(dataset_dir)}")
            else:
                n_imgs = len(list(img_d.glob("frame_*.png")))
                if n_imgs != ep_info["length"]:
                    err(f"Episode {ep_idx}, image key '{ik}': expected {ep_info['length']} frames, found {n_imgs}")

    # ------------------------------------------------------------------
    # 8. Tasks referenced by episodes exist in tasks.jsonl
    # ------------------------------------------------------------------
    logger.info("[7/8] Checking task references ...")
    all_task_strings = set(tasks.values())
    for ep_idx, ep_info in episodes_dict.items():
        for t in ep_info.get("tasks", []):
            if t not in all_task_strings:
                err(f"Episode {ep_idx}: task '{t}' not found in tasks.jsonl")

    # ------------------------------------------------------------------
    # 9. Episodes stats consistency
    # ------------------------------------------------------------------
    logger.info("[8/8] Checking episodes_stats consistency ...")
    if has_ep_stats:
        ep_stats = load_episodes_stats(dataset_dir)
        stats_ep_indices = set(ep_stats.keys())
        expected_ep_indices = set(ep_indices)
        missing_stats = expected_ep_indices - stats_ep_indices
        extra_stats = stats_ep_indices - expected_ep_indices
        if missing_stats:
            err(f"episodes_stats.jsonl missing entries for episodes: {sorted(missing_stats)}")
        if extra_stats:
            warn(f"episodes_stats.jsonl has extra entries for episodes: {sorted(extra_stats)}")

        # Check stat shapes match features
        for ep_idx in sorted(stats_ep_indices & expected_ep_indices)[:5]:  # spot-check first 5
            for key, ft in features.items():
                if key in DEFAULT_FEATURES or ft["dtype"] == "string":
                    continue
                if key not in ep_stats[ep_idx]:
                    warn(f"Episode {ep_idx} stats: missing feature '{key}'")
                    continue
                st = ep_stats[ep_idx][key]
                for stat_name in ["min", "max", "mean", "std", "count"]:
                    if stat_name not in st:
                        err(f"Episode {ep_idx} stats['{key}']: missing '{stat_name}'")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    if errors:
        logger.error(f"\nValidation FAILED: {len(errors)} error(s), {len(warnings)} warning(s)")
        return False
    elif warnings:
        logger.warning(f"\nValidation PASSED with {len(warnings)} warning(s)")
        return True
    else:
        logger.info("\nValidation PASSED — dataset is consistent")
        return True


# ---------------------------------------------------------------------------
# Delete episodes
# ---------------------------------------------------------------------------

def delete_episodes(dataset_dir: Path, episodes_to_delete: list[int],
                    output_dir: Path | None = None, root: Path | None = None):
    """Delete specified episodes, renumber the rest, and recompute stats.

    If output_dir is None the dataset is modified in-place (via a temp dir swap).
    Relative paths are resolved against *root* (default: ``~/.cache/huggingface/lerobot``).
    """
    dataset_dir = _resolve_path(dataset_dir, root)
    if output_dir is not None:
        output_dir = _resolve_path(output_dir, root)
    in_place = output_dir is None
    if not in_place:
        output_dir = Path(output_dir)
        if output_dir.resolve() == dataset_dir.resolve():
            raise ValueError(
                "Output directory must not be the same as the source dataset directory. "
                "Omit --output-dir for in-place editing."
            )
    if in_place:
        output_dir = dataset_dir.parent / (dataset_dir.name + "_edit_tmp")
    output_dir = Path(output_dir)

    # Validate source before editing
    logger.info("=== Validating source dataset before delete ===")
    src_valid = validate_dataset(dataset_dir)
    if not src_valid:
        logger.warning("Source dataset validation reported issues — continuing anyway")

    old_info = load_info(dataset_dir)
    info = dict(old_info)
    features = info["features"]
    chunks_size = info.get("chunks_size", DEFAULT_CHUNK_SIZE)
    fps = info.get("fps", 30)
    episodes_dict = load_episodes(dataset_dir)
    video_keys = _get_video_keys(features)
    image_keys = _get_image_keys(features)

    tasks, _ = load_tasks(dataset_dir)

    # Load old aggregated stats for the summary comparison
    old_stats = None
    if (dataset_dir / STATS_PATH).exists():
        old_stats = load_stats(dataset_dir)

    # Validate episodes to delete
    delete_set = set(episodes_to_delete)
    existing = set(episodes_dict.keys())
    invalid = delete_set - existing
    if invalid:
        raise ValueError(f"Episodes {sorted(invalid)} do not exist (available: {sorted(existing)})")

    keep_eps = sorted(existing - delete_set)

    if not keep_eps:
        raise ValueError(
            "Cannot delete all episodes — this would produce an empty dataset. "
            "Delete the dataset directory instead."
        )
    logger.info(f"Deleting {len(delete_set)} episodes, keeping {len(keep_eps)} episodes")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Rebuild task mapping from remaining episodes
    task_to_new_index: dict[str, int] = {}
    new_tasks: dict[int, str] = {}
    for old_ep_idx in keep_eps:
        for task_str in episodes_dict[old_ep_idx].get("tasks", []):
            if task_str not in task_to_new_index:
                new_idx = len(task_to_new_index)
                task_to_new_index[task_str] = new_idx
                new_tasks[new_idx] = task_str

    # Copy and reindex kept episodes
    new_episodes = []
    global_frame_idx = 0

    for new_ep_idx, old_ep_idx in enumerate(keep_eps):
        ep_info = episodes_dict[old_ep_idx]

        src_pq = _parquet_path(dataset_dir, old_ep_idx, chunks_size)
        dst_pq = _parquet_path(output_dir, new_ep_idx, chunks_size)
        if src_pq.exists():
            num_frames = _reindex_parquet(
                src_pq, dst_pq, new_ep_idx, global_frame_idx,
                video_keys, image_keys, chunks_size
            )
            _update_task_index_in_parquet(dst_pq, ep_info, task_to_new_index)
        else:
            num_frames = ep_info.get("length", 0)
            logger.warning(f"Parquet not found for episode {old_ep_idx}: {src_pq}")

        for vk in video_keys:
            src_vid = _video_path(dataset_dir, vk, old_ep_idx, chunks_size)
            dst_vid = _video_path(output_dir, vk, new_ep_idx, chunks_size)
            if src_vid.exists():
                dst_vid.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_vid, dst_vid)

        for ik in image_keys:
            src_img = _image_dir(dataset_dir, ik, old_ep_idx)
            dst_img = _image_dir(output_dir, ik, new_ep_idx)
            if src_img.exists():
                shutil.copytree(src_img, dst_img, dirs_exist_ok=True)

        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": ep_info.get("tasks", []),
            "length": num_frames,
        })
        global_frame_idx += num_frames

    # Build info
    total_videos = len(keep_eps) * len(video_keys) if video_keys else 0
    new_info = dict(info)
    new_info["total_episodes"] = len(keep_eps)
    new_info["total_frames"] = global_frame_idx
    new_info["total_tasks"] = len(new_tasks)
    new_info["total_videos"] = total_videos
    new_info["total_chunks"] = (_episode_chunk(len(keep_eps) - 1, chunks_size) + 1) if keep_eps else 0
    new_info["splits"] = {"train": f"0:{len(keep_eps)}"}

    # Write meta without stats first (recompute will fill them in)
    _write_all_meta(output_dir, new_info, new_episodes, new_tasks, {}, features)

    # Recompute stats from scratch
    logger.info("=== Recomputing stats after delete ===")
    recompute_stats(output_dir)

    # Load new aggregated stats for summary comparison
    new_stats = None
    if (output_dir / STATS_PATH).exists():
        new_stats = load_stats(output_dir)

    if in_place:
        backup = dataset_dir.parent / (dataset_dir.name + "_backup")
        dataset_dir.rename(backup)
        output_dir.rename(dataset_dir)
        final_dir = dataset_dir
    else:
        final_dir = output_dir

    # Validate output
    logger.info("=== Validating dataset after delete ===")
    out_valid = validate_dataset(final_dir)
    if not out_valid:
        if in_place:
            logger.error(
                "Output validation FAILED. The backup is preserved at: %s",
                backup,
            )
        else:
            logger.error("Output validation FAILED for: %s", final_dir)
    elif in_place:
        # Only remove the backup after validation succeeds
        shutil.rmtree(backup)
        logger.info("Backup removed after successful validation")

    # Print summary
    episode_remap = {old_idx: new_idx for new_idx, old_idx in enumerate(keep_eps)}
    _print_delete_summary(
        dataset_dir=dataset_dir,
        output_dir=final_dir,
        in_place=in_place,
        deleted_eps=sorted(delete_set),
        old_info=old_info,
        new_info=new_info,
        old_episodes=episodes_dict,
        new_episodes=new_episodes,
        old_tasks=tasks,
        new_tasks=new_tasks,
        episode_remap=episode_remap,
        old_stats=old_stats,
        new_stats=new_stats,
    )


# ---------------------------------------------------------------------------
# Merge datasets
# ---------------------------------------------------------------------------

def merge_datasets(
    dataset_a_dir: Path,
    dataset_b_dir: Path,
    output_dir: Path,
    task: str | None = None,
    robot_type: str | None = None,
    fps: int | None = None,
    root: Path | None = None,
    root_a: Path | None = None,
    root_b: Path | None = None,
):
    """Merge dataset B into dataset A, recompute stats, and validate.

    The dataset name (repo_id) in info.json is derived automatically from the
    output directory path (last two components, e.g. ``user/dataset_name``).

    Args:
        dataset_a_dir: Path to the first (base) dataset.
        dataset_b_dir: Path to the second dataset to append.
        output_dir: Where the merged dataset is written.
        task: Override all episodes with this single task string.
        robot_type: Override robot_type in info.json.
        fps: Override fps in info.json.
        root: Default root directory for resolving relative paths
            (default: ``~/.cache/huggingface/lerobot``).
        root_a: Root directory for dataset A (overrides *root* for dataset A only).
        root_b: Root directory for dataset B (overrides *root* for dataset B only).
    """
    dataset_a_dir = _resolve_path(dataset_a_dir, root_a or root)
    dataset_b_dir = _resolve_path(dataset_b_dir, root_b or root)
    output_dir = _resolve_path(output_dir, root)

    if output_dir.resolve() == dataset_a_dir.resolve() or output_dir.resolve() == dataset_b_dir.resolve():
        raise ValueError("Output directory must not be the same as either input dataset directory")

    # Validate both sources
    logger.info("=== Validating source dataset A ===")
    src_a_valid = validate_dataset(dataset_a_dir)
    if not src_a_valid:
        logger.warning("Source dataset A validation reported issues — continuing anyway")
    logger.info("=== Validating source dataset B ===")
    src_b_valid = validate_dataset(dataset_b_dir)
    if not src_b_valid:
        logger.warning("Source dataset B validation reported issues — continuing anyway")

    info_a = load_info(dataset_a_dir)
    info_b = load_info(dataset_b_dir)
    features_a = info_a["features"]
    chunks_size_a = info_a.get("chunks_size", DEFAULT_CHUNK_SIZE)
    chunks_size_b = info_b.get("chunks_size", DEFAULT_CHUNK_SIZE)
    chunks_size = chunks_size_a

    video_keys = _get_video_keys(features_a)
    image_keys = _get_image_keys(features_a)

    episodes_a = load_episodes(dataset_a_dir)
    episodes_b = load_episodes(dataset_b_dir)

    _validate_merge_compatibility(info_a, info_b, episodes_a, episodes_b)

    tasks_a, _ = load_tasks(dataset_a_dir)
    tasks_b, _ = load_tasks(dataset_b_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build task mapping
    if task is not None:
        unified_tasks = {0: task}
        task_to_new_index = {task: 0}
        logger.info(f"Overriding all episode tasks with: '{task}'")
    else:
        unified_tasks = dict(tasks_a)
        task_to_new_index = {t: idx for idx, t in tasks_a.items()}
        for _, task_str in sorted(tasks_b.items()):
            if task_str not in task_to_new_index:
                new_idx = len(unified_tasks)
                unified_tasks[new_idx] = task_str
                task_to_new_index[task_str] = new_idx

    new_episodes = []
    global_frame_idx = 0
    new_ep_idx = 0

    # Process dataset A
    for old_ep_idx in sorted(episodes_a.keys()):
        ep_info = episodes_a[old_ep_idx]
        num_frames = _copy_episode(
            dataset_a_dir, output_dir, old_ep_idx, new_ep_idx,
            global_frame_idx, video_keys, image_keys, chunks_size_a, chunks_size,
            ep_info, task_to_new_index, task_override=task,
        )
        ep_tasks = [task] if task else ep_info.get("tasks", [])
        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": ep_tasks,
            "length": num_frames,
        })
        global_frame_idx += num_frames
        new_ep_idx += 1

    # Process dataset B
    for old_ep_idx in sorted(episodes_b.keys()):
        ep_info = episodes_b[old_ep_idx]
        num_frames = _copy_episode(
            dataset_b_dir, output_dir, old_ep_idx, new_ep_idx,
            global_frame_idx, video_keys, image_keys, chunks_size_b, chunks_size,
            ep_info, task_to_new_index, task_override=task,
        )
        ep_tasks = [task] if task else ep_info.get("tasks", [])
        new_episodes.append({
            "episode_index": new_ep_idx,
            "tasks": ep_tasks,
            "length": num_frames,
        })
        global_frame_idx += num_frames
        new_ep_idx += 1

    # Build info
    total_episodes = new_ep_idx
    total_videos = total_episodes * len(video_keys) if video_keys else 0
    new_info = dict(info_a)
    new_info["total_episodes"] = total_episodes
    new_info["total_frames"] = global_frame_idx
    new_info["total_tasks"] = len(unified_tasks)
    new_info["total_videos"] = total_videos
    new_info["total_chunks"] = (_episode_chunk(total_episodes - 1, chunks_size) + 1) if total_episodes else 0
    new_info["splits"] = {"train": f"0:{total_episodes}"}

    # Derive repo_id from output directory (last 2 components, e.g. "user/dataset_name")
    # Derive repo_id from output path — expects "user/dataset_name" structure
    parts = output_dir.resolve().parts
    if len(parts) >= 3:  # ('/', ..., 'user', 'dataset_name')
        new_info["repo_id"] = "/".join(parts[-2:])
    else:
        new_info["repo_id"] = parts[-1] if parts else "unknown"
        logger.warning(
            f"Output path has fewer than 2 components; repo_id set to '{new_info['repo_id']}'. "
            "Consider using a path like '${HF_USER}/dataset_name'."
        )
    if robot_type is not None:
        new_info["robot_type"] = robot_type
    if fps is not None:
        new_info["fps"] = fps

    # Write meta without stats first (recompute will fill them in)
    _write_all_meta(output_dir, new_info, new_episodes, unified_tasks, {}, features_a)

    # Recompute stats from scratch on the merged data
    logger.info("=== Recomputing stats after merge ===")
    recompute_stats(output_dir)

    # Load new aggregated stats for summary
    new_stats = None
    if (output_dir / STATS_PATH).exists():
        new_stats = load_stats(output_dir)

    # Load old stats for reference in summary
    old_stats_a = None
    if (dataset_a_dir / STATS_PATH).exists():
        old_stats_a = load_stats(dataset_a_dir)
    old_stats_b = None
    if (dataset_b_dir / STATS_PATH).exists():
        old_stats_b = load_stats(dataset_b_dir)

    # Validate the merged output
    logger.info("=== Validating merged dataset ===")
    out_valid = validate_dataset(output_dir)
    if not out_valid:
        logger.error("Output validation FAILED for merged dataset: %s", output_dir)

    # Print summary
    _print_merge_summary(
        dataset_a_dir=dataset_a_dir,
        dataset_b_dir=dataset_b_dir,
        output_dir=output_dir,
        info_a=info_a,
        info_b=info_b,
        new_info=new_info,
        episodes_a=episodes_a,
        episodes_b=episodes_b,
        new_episodes=new_episodes,
        old_tasks_a=tasks_a,
        old_tasks_b=tasks_b,
        new_tasks=unified_tasks,
        task_override=task,
        robot_type_override=robot_type,
        fps_override=fps,
        old_stats_a=old_stats_a,
        old_stats_b=old_stats_b,
        new_stats=new_stats,
    )


def _copy_episode(src_dir: Path, dst_dir: Path, old_ep_idx: int, new_ep_idx: int,
                  global_frame_idx: int, video_keys: list[str], image_keys: list[str],
                  src_chunks_size: int, dst_chunks_size: int,
                  ep_info: dict, task_to_new_index: dict,
                  task_override: str | None = None) -> int:
    """Copy and reindex a single episode from src to dst. Returns frame count."""
    src_pq = _parquet_path(src_dir, old_ep_idx, src_chunks_size)
    dst_pq = _parquet_path(dst_dir, new_ep_idx, dst_chunks_size)

    if src_pq.exists():
        num_frames = _reindex_parquet(
            src_pq, dst_pq, new_ep_idx, global_frame_idx,
            video_keys, image_keys, dst_chunks_size
        )
        if task_override is not None:
            _update_task_index_in_parquet(
                dst_pq, {"tasks": [task_override]}, task_to_new_index,
            )
        else:
            _update_task_index_in_parquet(dst_pq, ep_info, task_to_new_index)
    else:
        num_frames = ep_info.get("length", 0)
        logger.warning(f"Parquet not found: {src_pq}")

    for vk in video_keys:
        src_vid = _video_path(src_dir, vk, old_ep_idx, src_chunks_size)
        dst_vid = _video_path(dst_dir, vk, new_ep_idx, dst_chunks_size)
        if src_vid.exists():
            dst_vid.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_vid, dst_vid)

    for ik in image_keys:
        src_img = _image_dir(src_dir, ik, old_ep_idx)
        dst_img = _image_dir(dst_dir, ik, new_ep_idx)
        if src_img.exists():
            shutil.copytree(src_img, dst_img, dirs_exist_ok=True)

    return num_frames


def _validate_merge_compatibility(info_a: dict, info_b: dict,
                                  episodes_a: dict | None = None,
                                  episodes_b: dict | None = None):
    """Print a compatibility report comparing both datasets, then raise on
    hard errors (feature schema mismatch).  Everything else is a warning.
    """
    fa = info_a["features"]
    fb = info_b["features"]

    warnings_list: list[str] = []
    errors_list: list[str] = []

    def _warn(msg: str):
        warnings_list.append(msg)

    def _error(msg: str):
        errors_list.append(msg)

    # ------------------------------------------------------------------
    # Helpers to categorise features
    # ------------------------------------------------------------------
    skip = set(DEFAULT_FEATURES.keys())

    def _camera_keys(feats):
        return sorted(k for k, v in feats.items() if v["dtype"] in ("video", "image"))

    def _state_keys(feats):
        return sorted(k for k, v in feats.items()
                      if k.startswith("observation") and v["dtype"] not in ("video", "image", "string")
                      and k not in skip)

    def _action_keys(feats):
        return sorted(k for k, v in feats.items()
                      if k.startswith("action") and k not in skip)

    # ------------------------------------------------------------------
    # 1. Robot type
    # ------------------------------------------------------------------
    robot_a = info_a.get("robot_type", "unknown")
    robot_b = info_b.get("robot_type", "unknown")
    if robot_a != robot_b:
        _warn(f"Robot type: A='{robot_a}' vs B='{robot_b}'  (will use A's)")

    # ------------------------------------------------------------------
    # 2. FPS
    # ------------------------------------------------------------------
    fps_a = info_a.get("fps")
    fps_b = info_b.get("fps")
    if fps_a != fps_b:
        _warn(f"FPS: A={fps_a} vs B={fps_b}  (will use A's)")

    # ------------------------------------------------------------------
    # 3. Codebase / subversion
    # ------------------------------------------------------------------
    for field in ("codebase_version", "trossen_subversion"):
        va = info_a.get(field, "n/a")
        vb = info_b.get(field, "n/a")
        if va != vb:
            _warn(f"{field}: A='{va}' vs B='{vb}'")

    # ------------------------------------------------------------------
    # 4. Cameras: count, names, resolution, codec
    # ------------------------------------------------------------------
    cam_a = _camera_keys(fa)
    cam_b = _camera_keys(fb)
    if len(cam_a) != len(cam_b):
        _warn(f"Camera count: A has {len(cam_a)} ({cam_a}) vs B has {len(cam_b)} ({cam_b})")
    elif cam_a != cam_b:
        _warn(f"Camera names: A={cam_a} vs B={cam_b}")

    # Per-camera resolution & codec check (for cameras that exist in both)
    for cam in sorted(set(cam_a) & set(cam_b)):
        shape_a = tuple(fa[cam]["shape"])
        shape_b = tuple(fb[cam]["shape"])
        if shape_a != shape_b:
            _warn(f"Camera '{cam}' resolution: A={shape_a} vs B={shape_b}")
        info_ca = fa[cam].get("info", {})
        info_cb = fb[cam].get("info", {})
        codec_a = info_ca.get("video.codec") or info_ca.get("codec_name", "?")
        codec_b = info_cb.get("video.codec") or info_cb.get("codec_name", "?")
        if codec_a != codec_b:
            _warn(f"Camera '{cam}' codec: A='{codec_a}' vs B='{codec_b}'")

    # ------------------------------------------------------------------
    # 5. Observation state: keys, shapes, joint names
    # ------------------------------------------------------------------
    state_a = _state_keys(fa)
    state_b = _state_keys(fb)
    if state_a != state_b:
        _warn(f"Observation state keys: A={state_a} vs B={state_b}")
    for key in sorted(set(state_a) & set(state_b)):
        sa, sb = tuple(fa[key]["shape"]), tuple(fb[key]["shape"])
        if sa != sb:
            _warn(f"'{key}' shape (arm DOF): A={sa} vs B={sb}")
        names_a = fa[key].get("names") or []
        names_b = fb[key].get("names") or []
        if names_a != names_b:
            _warn(f"'{key}' joint names: A={names_a} vs B={names_b}")

    # ------------------------------------------------------------------
    # 6. Action: keys, shapes, joint names
    # ------------------------------------------------------------------
    act_a = _action_keys(fa)
    act_b = _action_keys(fb)
    if act_a != act_b:
        _warn(f"Action keys: A={act_a} vs B={act_b}")
    for key in sorted(set(act_a) & set(act_b)):
        sa, sb = tuple(fa[key]["shape"]), tuple(fb[key]["shape"])
        if sa != sb:
            _warn(f"'{key}' shape (action DOF): A={sa} vs B={sb}")
        names_a = fa[key].get("names") or []
        names_b = fb[key].get("names") or []
        if names_a != names_b:
            _warn(f"'{key}' action names: A={names_a} vs B={names_b}")

    # ------------------------------------------------------------------
    # 7. Episode count & lengths
    # ------------------------------------------------------------------
    eps_a_count = info_a.get("total_episodes", "?")
    eps_b_count = info_b.get("total_episodes", "?")
    frames_a = info_a.get("total_frames", "?")
    frames_b = info_b.get("total_frames", "?")

    if episodes_a is not None and episodes_b is not None:
        lengths_a = [ep["length"] for ep in episodes_a.values()]
        lengths_b = [ep["length"] for ep in episodes_b.values()]
        avg_a = sum(lengths_a) / len(lengths_a) if lengths_a else 0
        avg_b = sum(lengths_b) / len(lengths_b) if lengths_b else 0
        min_a, max_a = (min(lengths_a), max(lengths_a)) if lengths_a else (0, 0)
        min_b, max_b = (min(lengths_b), max(lengths_b)) if lengths_b else (0, 0)

        if avg_a > 0 and avg_b > 0 and (avg_a / avg_b > 2 or avg_b / avg_a > 2):
            _warn(
                f"Episode length: "
                f"A avg={avg_a:.0f} (range {min_a}-{max_a}) vs "
                f"B avg={avg_b:.0f} (range {min_b}-{max_b})  — >2x difference"
            )

    # ------------------------------------------------------------------
    # 8. Hard errors: every feature must exist in both with matching dtype/shape
    # ------------------------------------------------------------------
    for key in set(fa.keys()) | set(fb.keys()):
        if key not in fa:
            _error(f"Feature '{key}' exists in dataset B but not A")
        elif key not in fb:
            _error(f"Feature '{key}' exists in dataset A but not B")
        else:
            if fa[key]["dtype"] != fb[key]["dtype"]:
                _error(f"Feature '{key}' dtype mismatch: A='{fa[key]['dtype']}' vs B='{fb[key]['dtype']}'")
            if tuple(fa[key]["shape"]) != tuple(fb[key]["shape"]):
                _error(f"Feature '{key}' shape mismatch: A={fa[key]['shape']} vs B={fb[key]['shape']}")

    # ------------------------------------------------------------------
    # Print the report
    # ------------------------------------------------------------------
    report_lines = [
        "",
        "=" * 64,
        " MERGE COMPATIBILITY REPORT",
        "=" * 64,
        "",
        f"  Dataset A : {info_a.get('robot_type', '?')}  |  {eps_a_count} episodes  |  {frames_a} frames  |  {fps_a} fps",
        f"  Dataset B : {info_b.get('robot_type', '?')}  |  {eps_b_count} episodes  |  {frames_b} frames  |  {fps_b} fps",
        "",
        f"  Cameras   : A={cam_a}",
        f"              B={cam_b}",
        f"  State keys: A={state_a}",
        f"              B={state_b}",
        f"  Action keys: A={act_a}",
        f"               B={act_b}",
        "",
    ]

    if episodes_a is not None and episodes_b is not None:
        report_lines += [
            f"  Episodes  : A  {len(lengths_a)} eps, avg {avg_a:.0f} frames (range {min_a}-{max_a})",
            f"              B  {len(lengths_b)} eps, avg {avg_b:.0f} frames (range {min_b}-{max_b})",
            "",
        ]

    if not warnings_list and not errors_list:
        report_lines += ["  Result    : ALL COMPATIBLE", ""]
    else:
        if warnings_list:
            report_lines += ["  WARNINGS:"]
            for w in warnings_list:
                report_lines.append(f"    - {w}")
            report_lines.append("")
        if errors_list:
            report_lines += ["  ERRORS (merge cannot proceed):"]
            for e in errors_list:
                report_lines.append(f"    - {e}")
            report_lines.append("")

    report_lines.append("=" * 64)
    report = "\n".join(report_lines)
    logger.info(report)

    # Raise on hard errors
    if errors_list:
        raise ValueError(
            f"Merge blocked by {len(errors_list)} incompatibility error(s):\n"
            + "\n".join(f"  - {e}" for e in errors_list)
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _run_from_config(config_path: str):
    """Load a JSON config file and dispatch the appropriate command."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg = json.load(f)

    command = cfg.pop("command", None)
    if command is None:
        raise ValueError("Config JSON must contain a 'command' key ('delete', 'merge', or 'validate').")

    root = Path(cfg["root"]) if cfg.get("root") else None

    if command == "delete":
        required = {"dataset_dir", "episodes"}
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"Delete config missing required keys: {missing}")
        delete_episodes(
            dataset_dir=Path(cfg["dataset_dir"]),
            episodes_to_delete=cfg["episodes"],
            output_dir=Path(cfg["output_dir"]) if cfg.get("output_dir") else None,
            root=root,
        )

    elif command == "merge":
        required = {"dataset_a", "dataset_b", "output_dir"}
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"Merge config missing required keys: {missing}")
        root_a = Path(cfg["root_a"]) if cfg.get("root_a") else None
        root_b = Path(cfg["root_b"]) if cfg.get("root_b") else None
        merge_datasets(
            dataset_a_dir=Path(cfg["dataset_a"]),
            dataset_b_dir=Path(cfg["dataset_b"]),
            output_dir=Path(cfg["output_dir"]),
            task=cfg.get("task"),
            robot_type=cfg.get("robot_type"),
            fps=cfg.get("fps"),
            root=root,
            root_a=root_a,
            root_b=root_b,
        )

    elif command == "validate":
        required = {"dataset_dir"}
        missing = required - set(cfg.keys())
        if missing:
            raise ValueError(f"Validate config missing required keys: {missing}")
        ok = validate_dataset(Path(cfg["dataset_dir"]), root=root)
        if not ok:
            raise SystemExit(1)

    else:
        raise ValueError(f"Unknown command '{command}'. Must be 'delete', 'merge', or 'validate'.")


def main():
    parser = argparse.ArgumentParser(
        description="Edit LeRobot datasets: delete episodes, merge datasets, or validate integrity.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument("--config", type=str, default=None,
                        help="Path to a JSON config file containing all arguments. "
                             "When provided, all other CLI flags and the subcommand are ignored.")
    parser.add_argument("--root", type=Path, default=None,
                        help="Root directory for resolving relative dataset paths. "
                             "Defaults to ~/.cache/huggingface/lerobot (HF_LEROBOT_HOME).")

    subparsers = parser.add_subparsers(dest="command")

    # --- delete ---
    del_parser = subparsers.add_parser("delete", help="Delete episodes from a dataset")
    del_parser.add_argument("--dataset-dir", type=Path, required=True,
                            help="Path to the dataset (absolute, or relative to --root)")
    del_parser.add_argument("--episodes", type=int, nargs="+", required=True,
                            help="Episode indices to delete")
    del_parser.add_argument("--output-dir", type=Path, default=None,
                            help="Output directory (absolute, or relative to --root). "
                                 "Default: edit in-place.")

    # --- merge ---
    merge_parser = subparsers.add_parser("merge", help="Merge two datasets")
    merge_parser.add_argument("--dataset-a", type=Path, required=True,
                              help="Path to dataset A (absolute, or relative to --root / --root-a)")
    merge_parser.add_argument("--dataset-b", type=Path, required=True,
                              help="Path to dataset B (absolute, or relative to --root / --root-b)")
    merge_parser.add_argument("--output-dir", type=Path, required=True,
                              help="Output directory (absolute, or relative to --root)")
    merge_parser.add_argument("--root-a", type=Path, default=None,
                              help="Root directory for dataset A (overrides --root for A only)")
    merge_parser.add_argument("--root-b", type=Path, default=None,
                              help="Root directory for dataset B (overrides --root for B only)")
    merge_parser.add_argument("--task", type=str, default=None,
                              help="Override ALL episodes with this single task string. "
                                   "If omitted, per-episode tasks from both inputs are preserved.")
    merge_parser.add_argument("--robot-type", type=str, default=None,
                              help="Override robot_type in info.json (default: from dataset A)")
    merge_parser.add_argument("--fps", type=int, default=None,
                              help="Override fps in info.json (default: from dataset A)")

    # --- validate ---
    val_parser = subparsers.add_parser("validate", help="Validate dataset integrity")
    val_parser.add_argument("--dataset-dir", type=Path, required=True,
                            help="Path to the dataset (absolute, or relative to --root)")

    args = parser.parse_args()

    # JSON config mode
    if args.config:
        _run_from_config(args.config)
        return

    if args.command is None:
        parser.print_help()
        parser.error("Either --config or a subcommand (delete/merge/validate) is required.")

    root = args.root

    if args.command == "delete":
        delete_episodes(args.dataset_dir, args.episodes, args.output_dir, root=root)
    elif args.command == "merge":
        merge_datasets(
            args.dataset_a,
            args.dataset_b,
            args.output_dir,
            task=args.task,
            robot_type=args.robot_type,
            fps=args.fps,
            root=root,
            root_a=args.root_a,
            root_b=args.root_b,
        )
    elif args.command == "validate":
        ok = validate_dataset(args.dataset_dir, root=root)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
