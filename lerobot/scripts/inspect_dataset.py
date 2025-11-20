#!/usr/bin/env python3
"""
Quick script to inspect a LeRobot dataset and show available cameras and basic info.

Usage:
    python inspect_dataset.py --repo-id <repo_id> --root <dataset_root>
"""

import argparse
import sys
from pathlib import Path

# Add the lerobot module to path if needed
sys.path.insert(0, str(Path(__file__).parent / "sandi" / "third_party" / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(repo_id: str, root: str):
    """Load and inspect a dataset."""
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,  # Don't download videos for inspection
    )
    
    print("\n" + "=" * 80)
    print("DATASET INFORMATION")
    print("=" * 80)
    print(f"Repository ID: {dataset.repo_id}")
    print(f"Root: {dataset.root}")
    print(f"Total Episodes: {dataset.meta.total_episodes}")
    print(f"Total Frames: {dataset.meta.total_frames}")
    print(f"FPS: {dataset.fps}")
    print(f"Robot Type: {dataset.meta.robot_type}")
    
    print("\n" + "=" * 80)
    print("CAMERA KEYS (for --camera argument)")
    print("=" * 80)
    if dataset.meta.camera_keys:
        for i, camera in enumerate(dataset.meta.camera_keys, 1):
            print(f"{i}. {camera}")
    else:
        print("No cameras found in dataset")
    
    print("\n" + "=" * 80)
    print("ALL FEATURES")
    print("=" * 80)
    for feature_name, feature_info in dataset.features.items():
        dtype = feature_info.get('dtype', 'unknown')
        shape = feature_info.get('shape', 'unknown')
        print(f"  {feature_name}: {dtype} {shape}")
    
    print("\n" + "=" * 80)
    print("TASKS")
    print("=" * 80)
    for task_idx, task_desc in dataset.meta.tasks.items():
        print(f"  {task_idx}: {task_desc}")
    
    print("\n" + "=" * 80)
    print("SAMPLE EPISODES")
    print("=" * 80)
    # Show first 5 episodes
    for ep_idx in range(min(5, dataset.meta.total_episodes)):
        ep_info = dataset.meta.episodes[ep_idx]
        print(f"Episode {ep_idx}:")
        print(f"  Length: {ep_info['length']} frames")
        print(f"  Tasks: {ep_info['tasks']}")
    
    if dataset.meta.total_episodes > 5:
        print(f"  ... and {dataset.meta.total_episodes - 5} more episodes")
    
    # Check for existing metadata file
    metadata_file = Path(root) / "meta" / "episodes_metadata.jsonl"
    print("\n" + "=" * 80)
    print("CUSTOM METADATA")
    print("=" * 80)
    if metadata_file.exists():
        import jsonlines
        with jsonlines.open(metadata_file) as reader:
            entries = list(reader)
        print(f"✓ Found episodes_metadata.jsonl with {len(entries)} labeled episodes")
        
        # Show unique metadata keys
        all_keys = set()
        for entry in entries:
            all_keys.update(entry.keys())
        all_keys.discard("episode_index")
        if all_keys:
            print(f"  Metadata keys: {', '.join(sorted(all_keys))}")
    else:
        print("✗ No episodes_metadata.jsonl found")
        print(f"  Run label_episodes.py to create it")


def main():
    parser = argparse.ArgumentParser(description="Inspect LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID of the dataset"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory of the dataset"
    )
    
    args = parser.parse_args()
    inspect_dataset(args.repo_id, args.root)


if __name__ == "__main__":
    main()

