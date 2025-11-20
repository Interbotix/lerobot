#!/usr/bin/env python3
"""
Split a LeRobot dataset into skill-based episodes.

OVERVIEW
========
Given a LeRobot dataset where episodes have been labeled with skills and time intervals,
this script splits each episode into separate episodes based on skill time intervals.

Each skill interval becomes a new episode with:
- Properly time-sliced parquet data and videos
- Reset frame indices and timestamps (starting from 0)
- Appropriate task indices based on skill type
- Preserved metadata with source episode tracking

INPUT DATASET STRUCTURE
=======================
The source dataset must have:

1. meta/episodes_metadata.jsonl with "skills" field:
   {
     "episode_index": 26,
     "skills": {
       "pick_up_bread": [[0.0, 11.4], [24.5, 35.8]],
       "put_bread_on_table": [[11.4, 15.4]]
     },
     "other_metadata": "value"
   }

2. skill_to_prompt.yaml mapping skills to prompts:
   pick_up_bread: "Pick up one slice of bread."
   put_bread_on_table: "Put one slice of bread at the blue cross mark on the table."

3. Standard LeRobot structure (data/, videos/, meta/)

USAGE
=====
Basic usage:
    python split_dataset_by_skills.py \
      --source-root /path/to/source/dataset \
      --output-root /path/to/output/dataset \
      --repo-id "chef_robotics/lettuce-sandwich"

Test mode (process only first N episodes with skills):
    python split_dataset_by_skills.py \
      --source-root /path/to/source \
      --output-root /path/to/output-test \
      --repo-id "org/dataset" \
      --max-episodes 2

Advanced options:
    --skip-validation     Skip dataset validation (use with caution)
    --video-keys          Specify cameras to process (auto-detected by default)
    --max-episodes N      Process only first N episodes (for testing)

OUTPUT DATASET STRUCTURE
========================
The output dataset will have:

1. meta/tasks.jsonl - Task definitions from skills:
   {"task_index": 0, "task": "Pick up one slice of bread."}
   {"task_index": 1, "task": "Put one slice of bread at the blue cross mark on the table."}

2. meta/episodes_metadata.jsonl - New episode metadata:
   {
     "episode_index": 0,
     "skills": {"pick_up_bread": [[0.0, 11.4]]},
     "source_episode": "chef_robotics/lettuce-sandwich/26:[0.0, 11.4]",
     "other_metadata": "preserved from source"
   }

3. data/chunk-XXX/episode_XXXXXX.parquet - Time-sliced parquet data
4. videos/chunk-XXX/{video_key}/episode_XXXXXX.mp4 - Time-sliced videos
5. meta/episodes.jsonl, meta/episodes_stats.jsonl, meta/info.json - Updated metadata


EXAMPLE
=======
Given episode 26 with:
    "skills": {
      "pick_up_bread": [[0.0, 11.4], [24.5, 35.8]],
      "put_bread_on_table": [[11.4, 15.4]]
    }

Creates 3 new episodes:
- Episode 0: frames [0.0, 11.4], task_index=0 (pick_up_bread)
- Episode 1: frames [24.5, 35.8], task_index=0 (pick_up_bread)
- Episode 2: frames [11.4, 15.4], task_index=1 (put_bread_on_table)

IMPORTANT NOTES
===============
- Episodes without "skills" field are skipped
- Frame indices and timestamps are reset to start from 0 in each new episode
- Videos are split using ffmpeg with libsvtav1 codec
- Task indices are assigned based on skill order in skill_to_prompt.yaml
- All metadata fields (except skills and episode_index) are preserved
- Episode statistics are recomputed for each split episode
- FPS is automatically read from source dataset's meta/info.json

VALIDATION
==========
The script automatically validates before processing:
- Required files exist
- At least one episode has skills
- Skills structure is valid (proper time intervals)
- All skills have prompts defined
- Sample data/video files exist

If validation fails, the script exits with detailed error messages.

TROUBLESHOOTING
===============
- FFmpeg not found: Install ffmpeg (brew install ffmpeg or apt-get install ffmpeg)
- Validation errors: Check error messages for missing files or invalid skill intervals
- Out of memory: Try --max-episodes to process in smaller batches
- Video codec errors: Source videos must use libsvtav1 codec
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml


def load_episodes_metadata(metadata_path: Path) -> List[Dict[str, Any]]:
    """Load episodes metadata from JSONL file."""
    episodes = []
    with open(metadata_path, "r") as f:
        for line in f:
            episodes.append(json.loads(line.strip()))
    return episodes


def load_skill_to_prompt(yaml_path: Path) -> Dict[str, str]:
    """Load skill to prompt mapping from YAML file."""
    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)
    # Remove comment-only keys and extract actual skill mappings
    return {k: v for k, v in data.items() if isinstance(v, str)}


def get_unique_skills_from_episodes(episodes: List[Dict[str, Any]]) -> List[str]:
    """Extract unique skills from all episodes, maintaining order."""
    skills_set = []
    seen = set()
    for episode in episodes:
        if "skills" in episode:
            for skill in episode["skills"].keys():
                if skill not in seen:
                    skills_set.append(skill)
                    seen.add(skill)
    return skills_set


def create_tasks_jsonl(output_path: Path, skill_to_task_index: Dict[str, int], skill_to_prompt: Dict[str, str]) -> None:
    """Create tasks.jsonl with task_index and prompts for each skill."""
    with open(output_path, "w") as f:
        for skill, task_index in skill_to_task_index.items():
            task_data = {
                "task_index": task_index,
                "task": skill_to_prompt[skill]
            }
            f.write(json.dumps(task_data) + "\n")


def split_parquet_by_time(
    parquet_path: Path,
    start_time: float,
    end_time: float,
    fps: float
) -> pd.DataFrame:
    """
    Extract a time slice from a parquet file.
    
    Args:
        parquet_path: Path to the source parquet file
        start_time: Start time in seconds
        end_time: End time in seconds
        fps: Frames per second
        
    Returns:
        DataFrame with the extracted frames
    """
    df = pd.read_parquet(parquet_path)
    
    # Calculate frame indices
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Extract the slice
    sliced_df = df.iloc[start_frame:end_frame].copy()
    
    # Reset frame_index to start from 0
    if "frame_index" in sliced_df.columns:
        sliced_df["frame_index"] = range(len(sliced_df))
    
    # Reset index in the dataframe
    if "index" in sliced_df.columns:
        sliced_df["index"] = range(len(sliced_df))
    
    # Reset timestamp to start from 0.0 to match the split videos
    # Videos split with ffmpeg start at timestamp 0, so parquet data must match
    if "timestamp" in sliced_df.columns:
        # Preserve the original time intervals
        original_timestamps = sliced_df["timestamp"].values
        # Shift so first timestamp is 0.0
        sliced_df["timestamp"] = original_timestamps - original_timestamps[0]
    
    return sliced_df


def compute_episode_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute statistics for an episode from its dataframe.
    
    Args:
        df: DataFrame containing episode data
        
    Returns:
        Dictionary with statistics for each feature
    """
    stats = {}
    
    for col in df.columns:
        col_data = df[col]
        
        # Handle different data types
        if col_data.dtype == object:
            # For array columns (like action, observation.state, images)
            try:
                # Try to stack arrays
                if isinstance(col_data.iloc[0], (list, np.ndarray)):
                    stacked = np.stack(col_data.values)
                    
                    # Check if it's image data (3D or 4D)
                    if len(stacked.shape) >= 3:
                        # Image data - compute stats across batch dimension only
                        stats[col] = {
                            "min": np.min(stacked, axis=0).tolist(),
                            "max": np.max(stacked, axis=0).tolist(),
                            "mean": np.mean(stacked, axis=0).tolist(),
                            "std": np.std(stacked, axis=0).tolist(),
                            "count": [len(stacked)]
                        }
                    else:
                        # 1D or 2D array data (action, state, etc.)
                        stats[col] = {
                            "min": np.min(stacked, axis=0).tolist(),
                            "max": np.max(stacked, axis=0).tolist(),
                            "mean": np.mean(stacked, axis=0).tolist(),
                            "std": np.std(stacked, axis=0).tolist(),
                            "count": [len(stacked)]
                        }
                else:
                    # Scalar object type, treat as numeric
                    col_array = col_data.astype(float).values
                    stats[col] = {
                        "min": [float(np.min(col_array))],
                        "max": [float(np.max(col_array))],
                        "mean": [float(np.mean(col_array))],
                        "std": [float(np.std(col_array))],
                        "count": [len(col_array)]
                    }
            except (ValueError, TypeError):
                # Skip columns that can't be converted
                continue
        else:
            # Numeric columns
            col_array = col_data.values
            stats[col] = {
                "min": [float(np.min(col_array))],
                "max": [float(np.max(col_array))],
                "mean": [float(np.mean(col_array))],
                "std": [float(np.std(col_array))],
                "count": [len(col_array)]
            }
    
    return stats


def split_video_by_time(
    video_path: Path,
    output_path: Path,
    start_time: float,
    end_time: float
) -> None:
    """
    Extract a time slice from a video file using ffmpeg.
    
    Args:
        video_path: Path to the source video
        output_path: Path to save the output video
        start_time: Start time in seconds
        end_time: End time in seconds
    """
    import subprocess
    
    duration = end_time - start_time
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use ffmpeg to extract video segment
    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-ss", str(start_time),
        "-t", str(duration),
        "-c:v", "libsvtav1",  # Use same codec as original
        "-crf", "30",
        "-preset", "8",
        "-c:a", "copy",
        "-y",  # Overwrite output file
        str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"Error splitting video {video_path}: {e.stderr}")
        raise


def validate_dataset(dataset_root: Path) -> None:
    """
    Validate that a dataset is ready for skill-based splitting.
    
    Args:
        dataset_root: Root directory of the dataset to validate
        
    Raises:
        ValueError: If validation fails
        FileNotFoundError: If required files are missing
    """
    print(f"Validating dataset at: {dataset_root}")
    print("=" * 80)
    
    # Check 1: Required files exist
    print("\n[1/5] Checking required files...")
    required_files = [
        "meta/episodes_metadata.jsonl",
        "meta/info.json",
        "skill_to_prompt.yaml"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = dataset_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path} exists")
        else:
            print(f"  ✗ {file_path} NOT FOUND")
            missing_files.append(file_path)
    
    if missing_files:
        raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
    
    # Check 2: Load and parse episodes_metadata.jsonl
    print("\n[2/5] Checking episodes_metadata.jsonl...")
    episodes = []
    episodes_with_skills = []
    
    with open(dataset_root / "meta/episodes_metadata.jsonl", "r") as f:
        for line_num, line in enumerate(f, 1):
            try:
                episode = json.loads(line.strip())
                episodes.append(episode)
                
                if "skills" in episode:
                    episodes_with_skills.append(episode)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num} in episodes_metadata.jsonl: Invalid JSON - {e}")
    
    print(f"  ✓ Total episodes: {len(episodes)}")
    print(f"  ✓ Episodes with skills: {len(episodes_with_skills)}")
    
    if len(episodes_with_skills) == 0:
        raise ValueError("No episodes have a 'skills' field! Cannot split dataset.")
    
    # Check 3: Extract unique skills and validate structure
    print("\n[3/5] Validating skills structure...")
    all_skills: Set[str] = set()
    skill_intervals_count = 0
    validation_errors = []
    
    for episode in episodes_with_skills:
        episode_idx = episode.get("episode_index", "unknown")
        skills = episode["skills"]
        
        if not isinstance(skills, dict):
            validation_errors.append(f"Episode {episode_idx}: 'skills' is not a dict")
            continue
        
        for skill, intervals in skills.items():
            all_skills.add(skill)
            
            if not isinstance(intervals, list):
                validation_errors.append(f"Episode {episode_idx}, skill '{skill}': intervals is not a list")
                continue
            
            for interval in intervals:
                if not isinstance(interval, list) or len(interval) != 2:
                    validation_errors.append(f"Episode {episode_idx}, skill '{skill}': invalid interval {interval}")
                    continue
                
                start, end = interval
                if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                    validation_errors.append(f"Episode {episode_idx}, skill '{skill}': interval times must be numbers")
                    continue
                
                if start >= end:
                    validation_errors.append(f"Episode {episode_idx}, skill '{skill}': start time >= end time: [{start}, {end}]")
                
                skill_intervals_count += 1
    
    if validation_errors:
        error_msg = "\n  ✗ ".join(["Skills validation errors:"] + validation_errors)
        raise ValueError(error_msg)
    
    print(f"  ✓ Unique skills found: {len(all_skills)}")
    print(f"  ✓ Total skill intervals: {skill_intervals_count}")
    print(f"  → Will create {skill_intervals_count} new episodes")
    
    # Check 4: Validate skill_to_prompt.yaml
    print("\n[4/5] Validating skill_to_prompt.yaml...")
    with open(dataset_root / "skill_to_prompt.yaml", "r") as f:
        skill_to_prompt = yaml.safe_load(f)
    
    # Filter out non-string values (comments)
    skill_to_prompt = {k: v for k, v in skill_to_prompt.items() if isinstance(v, str)}
    
    missing_skills = all_skills - set(skill_to_prompt.keys())
    
    if missing_skills:
        print(f"  ⚠ Skills in episodes but not in skill_to_prompt.yaml: {missing_skills}")
        print("    (These will use default prompts)")
    
    matched_skills = all_skills & set(skill_to_prompt.keys())
    print(f"  ✓ Skills with prompts: {len(matched_skills)}/{len(all_skills)}")
    
    # Check 5: Sample data and video files
    print("\n[5/5] Checking data and video files (sampling)...")
    
    with open(dataset_root / "meta/info.json", "r") as f:
        info = json.load(f)
    
    chunks_size = info.get("chunks_size", 1000)
    
    # Check a sample of episodes with skills
    sample_size = min(3, len(episodes_with_skills))
    sample_episodes = episodes_with_skills[:sample_size]
    
    missing_data_files = []
    for episode in sample_episodes:
        episode_idx = episode["episode_index"]
        episode_chunk = episode_idx // chunks_size
        
        # Check data file
        data_path = dataset_root / f"data/chunk-{episode_chunk:03d}/episode_{episode_idx:06d}.parquet"
        if data_path.exists():
            print(f"  ✓ Episode {episode_idx}: data file exists")
        else:
            print(f"  ✗ Episode {episode_idx}: data file NOT FOUND at {data_path}")
            missing_data_files.append(str(data_path))
        
        # Check at least one video file
        video_found = False
        for key, value in info["features"].items():
            if key.startswith("observation.images.") and value.get("dtype") == "video":
                # Use full feature name as that's how directories are named
                video_path = dataset_root / f"videos/chunk-{episode_chunk:03d}/{key}/episode_{episode_idx:06d}.mp4"
                if video_path.exists():
                    video_found = True
                    break
        
        if video_found:
            print(f"  ✓ Episode {episode_idx}: video files exist")
        else:
            print(f"  ⚠ Episode {episode_idx}: no video files found (may be okay if dataset has no videos)")
    
    if missing_data_files:
        raise FileNotFoundError(f"Missing data files: {', '.join(missing_data_files)}")
    
    # Summary
    print("\n" + "=" * 80)
    print("✓ VALIDATION PASSED - Dataset is ready for splitting!")
    print(f"\nSummary:")
    print(f"  - Episodes to process: {len(episodes_with_skills)}")
    print(f"  - New episodes to create: {skill_intervals_count}")
    print(f"  - Unique skills: {len(all_skills)}")
    print(f"  - Skills: {', '.join(sorted(all_skills))}")
    print("=" * 80 + "\n")


def split_dataset(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    video_keys: List[str] = None,
    skip_validation: bool = False,
    max_episodes: int = None
) -> None:
    """
    Split a LeRobot dataset by skills.
    
    Args:
        source_root: Root directory of the source dataset
        output_root: Root directory for the output dataset
        repo_id: Repository ID for the source dataset
        video_keys: List of video keys to process (e.g., ['cam_high', 'cam_low'])
        skip_validation: Skip validation step (default: False)
        max_episodes: Maximum number of episodes with skills to process (default: None = all)
        
    Raises:
        ValueError: If dataset validation fails
        FileNotFoundError: If required files are missing
    """
    # Validate dataset before processing
    if not skip_validation:
        validate_dataset(source_root)
    else:
        print("Skipping validation (--skip-validation flag set)")
        print("=" * 80 + "\n")
    
    if max_episodes is not None:
        print(f"⚠️  TEST MODE: Only processing first {max_episodes} episodes with skills")
        print("=" * 80 + "\n")
    
    # Load metadata
    source_meta = source_root / "meta"
    episodes_metadata = load_episodes_metadata(source_meta / "episodes_metadata.jsonl")
    
    # Load skill to prompt mapping
    skill_to_prompt_path = source_root / "skill_to_prompt.yaml"
    if not skill_to_prompt_path.exists():
        raise FileNotFoundError(f"skill_to_prompt.yaml not found at {skill_to_prompt_path}")
    skill_to_prompt = load_skill_to_prompt(skill_to_prompt_path)
    
    # Get unique skills and create skill to task_index mapping
    unique_skills = get_unique_skills_from_episodes(episodes_metadata)

    # Check that all skills from episodes are a subset of skill_to_prompt
    missing_skills = set(unique_skills) - set(skill_to_prompt.keys())
    if missing_skills:
        raise ValueError(f"Missing skills in skill_to_prompt.yaml: {missing_skills}")

    skill_to_task_index = {skill: idx for idx, skill in enumerate(skill_to_prompt.keys())}
    
    # Create output directories
    output_root.mkdir(parents=True, exist_ok=True)
    output_meta = output_root / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    
    # Copy skill_to_prompt.yaml
    shutil.copy(skill_to_prompt_path, output_root / "skill_to_prompt.yaml")
    
    # Create tasks.jsonl
    create_tasks_jsonl(output_meta / "tasks.jsonl", skill_to_task_index, skill_to_prompt)
    
    # Load source info.json
    with open(source_meta / "info.json", "r") as f:
        source_info = json.load(f)
    
    # Read FPS from source dataset
    fps = source_info.get("fps", 30.0)
    print(f"Using FPS from dataset: {fps}")
    print()
    
    # If video_keys not provided, extract from info.json
    if video_keys is None:
        video_keys = []
        for key, value in source_info["features"].items():
            if key.startswith("observation.images.") and value.get("dtype") == "video":
                # Use full feature name as video_key since that's how directories are named
                video_keys.append(key)
    
    # Process each episode and split by skills
    new_episode_index = 0
    new_episodes_metadata = []
    new_episodes = []
    new_episodes_stats = []
    
    total_frames = 0
    processed_episodes_count = 0
    
    for episode in episodes_metadata:
        source_episode_index = episode["episode_index"]
        
        # Skip episodes without skills
        if "skills" not in episode:
            continue
        
        # Check if we've reached the max episodes limit
        if max_episodes is not None and processed_episodes_count >= max_episodes:
            print(f"\nReached max episodes limit ({max_episodes}). Stopping processing.")
            break
        
        processed_episodes_count += 1
        skills = episode["skills"]
        
        # Get source data path
        episode_chunk = source_episode_index // source_info["chunks_size"]
        source_data_path = source_root / f"data/chunk-{episode_chunk:03d}/episode_{source_episode_index:06d}.parquet"
        
        if not source_data_path.exists():
            print(f"Warning: Data file not found for episode {source_episode_index}: {source_data_path}")
            continue
        
        # Process each skill and its time intervals
        for skill, intervals in skills.items():
            task_index = skill_to_task_index[skill]
            
            for interval in intervals:
                start_time, end_time = interval
                duration = end_time - start_time
                
                # Split parquet data
                try:
                    episode_df = split_parquet_by_time(source_data_path, start_time, end_time, fps)
                except Exception as e:
                    print(f"Error processing episode {source_episode_index}, skill {skill}, interval {interval}: {e}")
                    continue
                
                # Update episode_index and task_index in the dataframe
                if "episode_index" in episode_df.columns:
                    episode_df["episode_index"] = new_episode_index
                if "task_index" in episode_df.columns:
                    episode_df["task_index"] = task_index
                
                # Save new parquet file
                new_episode_chunk = new_episode_index // source_info["chunks_size"]
                output_data_dir = output_root / f"data/chunk-{new_episode_chunk:03d}"
                output_data_dir.mkdir(parents=True, exist_ok=True)
                output_data_path = output_data_dir / f"episode_{new_episode_index:06d}.parquet"
                episode_df.to_parquet(output_data_path, engine="pyarrow", index=False)
                
                # Split videos for each camera
                for video_key in video_keys:
                    # video_key is the full feature name like "observation.images.cam_high"
                    source_video_path = source_root / f"videos/chunk-{episode_chunk:03d}/{video_key}/episode_{source_episode_index:06d}.mp4"
                    
                    if source_video_path.exists():
                        output_video_dir = output_root / f"videos/chunk-{new_episode_chunk:03d}/{video_key}"
                        output_video_path = output_video_dir / f"episode_{new_episode_index:06d}.mp4"
                        
                        try:
                            split_video_by_time(source_video_path, output_video_path, start_time, end_time)
                        except Exception as e:
                            print(f"Error splitting video for episode {source_episode_index}, camera {video_key}: {e}")
                    else:
                        print(f"Warning: Source video not found: {source_video_path}")
                
                # Create new episode metadata
                new_metadata = {k: v for k, v in episode.items() if k != "skills" and k != "episode_index"}
                new_metadata["episode_index"] = new_episode_index
                new_metadata["skills"] = {skill: [[0.0, duration]]}
                new_metadata["source_episode"] = f"{repo_id}/{source_episode_index}:[{start_time}, {end_time}]"
                new_episodes_metadata.append(new_metadata)
                
                # Create episode entry for episodes.jsonl
                # Try to load tasks from source if available
                source_episodes_path = source_meta / "episodes.jsonl"
                task_description = skill_to_prompt.get(skill, f"Execute {skill}")
                new_episode_entry = {
                    "episode_index": new_episode_index,
                    "tasks": [task_description],
                    "length": len(episode_df)
                }
                new_episodes.append(new_episode_entry)
                
                # Compute episode stats from the sliced dataframe
                episode_stats = compute_episode_stats(episode_df)
                new_stats_entry = {
                    "episode_index": new_episode_index,
                    "stats": episode_stats
                }
                new_episodes_stats.append(new_stats_entry)
                
                total_frames += len(episode_df)
                
                print(f"Created episode {new_episode_index} from source episode {source_episode_index}, "
                      f"skill '{skill}', interval [{start_time:.2f}, {end_time:.2f}]")
                
                new_episode_index += 1
    
    # Save new episodes metadata
    with open(output_meta / "episodes_metadata.jsonl", "w") as f:
        for metadata in new_episodes_metadata:
            f.write(json.dumps(metadata) + "\n")
    
    with open(output_meta / "episodes.jsonl", "w") as f:
        for episode_entry in new_episodes:
            f.write(json.dumps(episode_entry) + "\n")
    
    with open(output_meta / "episodes_stats.jsonl", "w") as f:
        for stats_entry in new_episodes_stats:
            f.write(json.dumps(stats_entry) + "\n")
    
    # Create new info.json
    new_info = source_info.copy()
    new_info["total_episodes"] = new_episode_index
    new_info["total_frames"] = total_frames
    new_info["total_tasks"] = len(unique_skills)
    new_info["total_videos"] = new_episode_index * len(video_keys)
    
    # Update splits
    new_info["splits"] = {"train": f"0:{new_episode_index}"}
    
    with open(output_meta / "info.json", "w") as f:
        json.dump(new_info, f, indent=4)
    
    print(f"\nDataset split complete!")
    print(f"Total new episodes: {new_episode_index}")
    print(f"Total frames: {total_frames}")
    print(f"Total tasks: {len(unique_skills)}")


def main():
    parser = argparse.ArgumentParser(
        description="Split a LeRobot dataset by skill time intervals"
    )
    parser.add_argument(
        "--source-root",
        type=str,
        required=True,
        help="Root directory of the source dataset"
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root directory for the output dataset"
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the source dataset (e.g., 'chef_robotics/lettuce-sandwich')"
    )
    parser.add_argument(
        "--video-keys",
        type=str,
        nargs="+",
        default=None,
        help="List of video keys to process (e.g., cam_high cam_low). If not provided, auto-detected from info.json"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip dataset validation step (advanced users only)"
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Maximum number of episodes with skills to process (for testing). Default: process all episodes"
    )
    
    args = parser.parse_args()
    
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    
    if not source_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {source_root}")
    
    split_dataset(
        source_root=source_root,
        output_root=output_root,
        repo_id=args.repo_id,
        video_keys=args.video_keys,
        skip_validation=args.skip_validation,
        max_episodes=args.max_episodes
    )


if __name__ == "__main__":
    main()

