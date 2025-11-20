#!/usr/bin/env python3
"""
Script to label episodes in a LeRobot dataset with custom metadata labels (k-v pairs).

Usage:
    python label_episodes.py --repo-id <repo_id> --root <dataset_root> \
        --camera <camera_key> --metadata-keys key1 key2 key3 \
        [--video-backend pyav|torchcodec] [--skip-labeled]

Example:
    python third_party/lerobot/lerobot/scripts/label_episodes.py \
        --repo-id chef_robotcs/lettuce_sandwich \
        --root /Users/sherrychen/Documents/chef_dev/ChefResearch/sandi/datasets/chef_robotics/lettuce-sandwich \
        --camera observation.images.cam_high \
        --metadata-keys right_arm_helping \
        --video-backend pyav \
        --skip-labeled

Notes:
    - On macOS, pyav (default) is typically faster than torchcodec for video decoding.
    - Use --skip-labeled to automatically skip episodes that are already fully labeled (no prompts shown).

"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import jsonlines
import numpy as np
import torch
from tqdm import tqdm

# Add the lerobot module to path if needed
sys.path.insert(0, str(Path(__file__).parent / "sandi" / "third_party" / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class EpisodeLabeler:
    def __init__(
        self,
        dataset: LeRobotDataset,
        camera_key: str,
        metadata_keys: list[str],
        metadata_file: Path,
        start_episode: int = 0,
        skip_labeled: bool = False,
    ):
        self.dataset = dataset
        self.camera_key = camera_key
        self.metadata_keys = metadata_keys
        self.metadata_file = metadata_file
        self.start_episode = start_episode
        self.skip_labeled = skip_labeled
        
        # Load existing metadata if it exists
        self.existing_metadata = self._load_existing_metadata()
        
        # Validate camera key
        if camera_key not in dataset.meta.camera_keys:
            raise ValueError(
                f"Camera '{camera_key}' not found. Available cameras: {dataset.meta.camera_keys}"
            )
    
    def _load_existing_metadata(self) -> dict:
        """Load existing metadata from file if it exists."""
        if not self.metadata_file.exists():
            return {}
        
        metadata = {}
        with jsonlines.open(self.metadata_file, "r") as reader:
            for entry in reader:
                metadata[entry["episode_index"]] = entry
        
        print(f"Loaded {len(metadata)} existing metadata entries from {self.metadata_file}")
        return metadata
    
    def _save_metadata(self, episode_index: int, metadata: dict) -> None:
        """Save or update metadata for an episode."""
        entry = {
            "episode_index": episode_index,
            **metadata
        }
        
        # Update in-memory
        self.existing_metadata[episode_index] = entry
        
        # Write all metadata back to file
        self.metadata_file.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(self.metadata_file, "w") as writer:
            for ep_idx in sorted(self.existing_metadata.keys()):
                writer.write(self.existing_metadata[ep_idx])
    
    def _get_episode_frames(self, episode_index: int) -> np.ndarray:
        """Get all frames from an episode for a specific camera by loading video file directly."""
        # Get the video file path directly
        video_path = Path(self.dataset.root) / self.dataset.meta.get_video_file_path(episode_index, self.camera_key)
        
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Open video file with cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        # Read all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            raise RuntimeError(f"No frames found in video: {video_path}")
        
        return np.array(frames)
    
    def _play_episode(self, episode_index: int, fps: int = 30, loop: bool = True) -> None:
        """Play an episode video in a window."""
        print(f"\nLoading episode {episode_index}...")
        frames = self._get_episode_frames(episode_index)
        
        window_name = f"Episode {episode_index} - Camera: {self.camera_key}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        delay = int(1000 / fps)  # milliseconds per frame
        
        print(f"Playing episode {episode_index} ({len(frames)} frames at {fps} FPS)")
        print("Press 'q' to finish watching, 'r' to replay, SPACE to pause/resume")
        
        paused = False
        frame_idx = 0
        
        while True:
            if not paused:
                # Display current frame with episode info
                frame = frames[frame_idx].copy()
                h, w = frame.shape[:2]
                
                # Add text overlay
                text = f"Episode {episode_index} | Frame {frame_idx + 1}/{len(frames)}"
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2)
                
                cv2.imshow(window_name, frame)
                frame_idx += 1
                
                if frame_idx >= len(frames):
                    if loop:
                        frame_idx = 0
                    else:
                        break
            else:
                # Just wait for key press when paused
                cv2.imshow(window_name, frames[frame_idx - 1])
            
            key = cv2.waitKey(delay if not paused else 100)
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                frame_idx = 0
                paused = False
            elif key == ord(' '):
                paused = not paused
                status = "Paused" if paused else "Playing"
                print(f"{status} at frame {frame_idx}")
        
        cv2.destroyWindow(window_name)
    
    def _get_metadata_input(self) -> dict:
        """Prompt user to input metadata values."""
        metadata = {}
        
        print("\nEnter metadata values:")
        print("  - For boolean: enter 'true', 'false', 't', 'f', 'yes', 'no', '1', '0'")
        print("  - For numbers: enter numeric value")
        print("  - For strings: enter any text")
        print("  - Press Enter to skip a key")
        
        for key in self.metadata_keys:
            while True:
                value = input(f"  {key}: ").strip()
                
                if not value:
                    print(f"    Skipped {key}")
                    break
                
                # Try to parse as boolean
                if value.lower() in ['true', 't', 'yes', '1']:
                    metadata[key] = True
                    break
                elif value.lower() in ['false', 'f', 'no', '0']:
                    metadata[key] = False
                    break
                
                # Try to parse as number
                try:
                    if '.' in value:
                        metadata[key] = float(value)
                    else:
                        metadata[key] = int(value)
                    break
                except ValueError:
                    # Keep as string
                    metadata[key] = value
                    break
        
        return metadata
    
    def label_episode(self, episode_index: int) -> tuple[bool, str]:
        """Label a single episode. Returns (labeled, reason) tuple.
        
        Reasons: 'all_keys_present', 'user_skip', 'incomplete_skip', 'no_metadata', 'labeled'
        """
        print("\n" + "=" * 80)
        print(f"Episode {episode_index} / {self.dataset.num_episodes - 1}")
        
        # Check if episode already has metadata
        if episode_index in self.existing_metadata:
            existing = self.existing_metadata[episode_index]
            
            # Check if all required metadata keys are present and non-empty
            all_keys_present = all(
                key in existing and existing[key] is not None and existing[key] != ""
                for key in self.metadata_keys
            )
            
            if all_keys_present:
                print(f"\n✓ Episode already has all required metadata, skipping:")
                for key in self.metadata_keys:
                    print(f"  {key}: {existing[key]}")
                return (False, "all_keys_present")
            
            # Some keys are missing, show what exists and ask to overwrite
            print(f"\nExisting metadata found (incomplete):")
            for key, value in existing.items():
                if key != "episode_index":
                    print(f"  {key}: {value}")
            
            response = input("\nOverwrite existing metadata? (y/n/skip): ").strip().lower()
            if response == 'skip' or response == 's':
                print("Skipping episode")
                return (False, "user_skip")
            elif response != 'y' and response != 'yes':
                print("Keeping existing metadata, moving to next episode")
                return (False, "incomplete_skip")
        
        # Get episode info
        episode_info = self.dataset.meta.episodes[episode_index]
        print(f"\nEpisode info:")
        print(f"  Tasks: {episode_info['tasks']}")
        print(f"  Length: {episode_info['length']} frames")
        
        # Play the episode
        self._play_episode(episode_index, fps=self.dataset.fps, loop=True)
        
        # Get metadata input
        metadata = self._get_metadata_input()
        
        if not metadata:
            print("No metadata entered, skipping episode")
            return (False, "no_metadata")
        
        # Confirm and save
        print("\nMetadata to save:")
        for key, value in metadata.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")
        
        self._save_metadata(episode_index, metadata)
        print(f"✓ Metadata saved for episode {episode_index}")
        return (True, "labeled")

    
    def run(self) -> None:
        """Run the labeling process for all episodes."""
        print("=" * 80)
        print("Episode Labeling Tool")
        print("=" * 80)
        print(f"Dataset: {self.dataset.repo_id}")
        print(f"Total episodes: {self.dataset.num_episodes}")
        print(f"Camera: {self.camera_key}")
        print(f"Metadata keys: {', '.join(self.metadata_keys)}")
        print(f"Output file: {self.metadata_file}")
        if self.skip_labeled:
            print(f"Mode: Auto-skip already labeled episodes")
        print("=" * 80)
        
        episode_index = self.start_episode
        labeled_count = 0
        
        while episode_index < self.dataset.num_episodes:
            labeled, reason = self.label_episode(episode_index)
            if labeled:
                labeled_count += 1
            
            # If skip_labeled is on and episode was already labeled, skip to next automatically
            if self.skip_labeled and reason == "all_keys_present":
                episode_index += 1
                continue
            
            # Ask what to do next
            if episode_index < self.dataset.num_episodes - 1:
                print("\nOptions:")
                print("  [y/n] Continue to next episode")
                print("  [p]   Go back and relabel previous episode")
                print("  [q]   Quit")
                response = input("Choice: ").strip().lower()
                
                if response == 'q' or response == 'quit':
                    print(f"\nStopping. Labeled {labeled_count} episodes.")
                    break
                elif response == 'p' or response == 'prev' or response == 'previous':
                    if episode_index > 0:
                        episode_index -= 1
                        print(f"\n⮐ Going back to episode {episode_index}")
                        continue
                    else:
                        print("\nAlready at the first episode, cannot go back.")
                        continue
                elif response == 'n' or response == 'no':
                    print(f"\nStopping. Labeled {labeled_count} episodes.")
                    break
                # Default: continue to next (y or Enter)
                episode_index += 1
            else:
                # Last episode
                episode_index += 1
        
        print("\n" + "=" * 80)
        print(f"Labeling complete! Labeled {labeled_count} episodes.")
        print(f"Metadata saved to: {self.metadata_file}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Label episodes with custom metadata")
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
    parser.add_argument(
        "--camera",
        type=str,
        required=True,
        help="Camera key to display (e.g., 'observation.images.top')"
    )
    parser.add_argument(
        "--metadata-keys",
        nargs="+",
        required=True,
        help="List of metadata keys to collect (e.g., 'right_arm_helping' 'cube_color')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for metadata (default: <root>/meta/episodes_metadata.jsonl)"
    )
    parser.add_argument(
        "--start-episode",
        type=int,
        default=0,
        help="Episode index to start from (default: 0)"
    )
    parser.add_argument(
        "--skip-labeled",
        action="store_true",
        help="Automatically skip episodes that already have all required metadata keys (no prompts)"
    )
    parser.add_argument(
        "--episodes",
        nargs="+",
        type=int,
        default=None,
        help="Specific episode indices to label (optional)"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Video backend to use for decoding (default: pyav, which is faster on macOS)"
    )
    
    args = parser.parse_args()
    
    # Load dataset
    print("Loading dataset...")
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
        download_videos=True,
        video_backend=args.video_backend,
    )
    
    # Set output file
    if args.output:
        metadata_file = Path(args.output)
    else:
        metadata_file = Path(args.root) / "meta" / "episodes_metadata.jsonl"
    
    # Create labeler
    labeler = EpisodeLabeler(
        dataset=dataset,
        camera_key=args.camera,
        metadata_keys=args.metadata_keys,
        metadata_file=metadata_file,
        start_episode=args.start_episode,
        skip_labeled=args.skip_labeled,
    )
    
    # Run labeling
    try:
        labeler.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved.")
        sys.exit(0)


if __name__ == "__main__":
    main()

