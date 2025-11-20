#!/usr/bin/env python3
"""
Script to label time intervals in episodes.

Usage:
    python label_episode_intervals.py --repo-id <repo_id> --root <dataset_root> \
        --interval-labels label1 label2 label3 \
        --metadata-key <key_name> \
        [--start-episode 0] [--skip-labeled]

Example:
    python third_party/lerobot/lerobot/scripts/label_episode_intervals.py \
        --repo-id chef_robotics/lettuce_sandwich \
        --root /Users/sherrychen/Documents/chef_dev/ChefResearch/sandi/datasets/chef_robotics/lettuce-sandwich \
        --interval-labels pick_up_bread put_bread_on_table pick_up_lettuce put_lettuce_on_bread put_bread_on_lettuce go_to_home_position \
        --metadata-key skills \
        --start-episode 31 \
        --skip-labeled

Notes:
    - This script allows you to label time periods in videos by marking time points
    - Time points are converted to intervals automatically
    - All camera views are displayed simultaneously
    - Use keyboard controls to navigate and label the video
    - Intervals are saved nested under the specified metadata key
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import jsonlines
import numpy as np
from collections import defaultdict

# Add the lerobot module to path if needed
sys.path.insert(0, str(Path(__file__).parent / "sandi" / "third_party" / "lerobot"))

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class TimeIntervalLabeler:
    """Interactive video player for labeling time intervals with multiple camera views."""
    
    def __init__(
        self, 
        video_paths: List[Path], 
        interval_labels: List[str], 
        fps: float = 30.0,
        title: str = "",
    ):
        self.video_paths = video_paths
        self.caps = {}
        self.total_frames = 0
        self.fps = fps
        self.title = title
        
        # Initialize video captures
        print(f"\nInitializing video captures for {len(video_paths)} videos...")
        for i, video_path in enumerate(video_paths):
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get camera name from the path (e.g., "cam_high" from path)
            # The video path structure is: videos/chunk-000/observation.images.cam_high/episode_000000.mp4
            camera_name = video_path.parent.name.split('.')[-1]  # Extract "cam_high" from "observation.images.cam_high"
            self.caps[camera_name] = cap
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"  Camera {i+1}/{len(video_paths)}: {camera_name} ({frame_count} frames)")
            
            # Use the first video to determine timing
            if i == 0:
                self.total_frames = frame_count
        
        self.duration_s = self.total_frames / fps
        self.current_frame = 0
        self.is_playing = False
        
        # Interval labels
        self.interval_labels = interval_labels
        
        # Store labeled time points: [(time, stage), ...] sorted by time
        self.labeled_points = []
        
        # For point selection and deletion
        self.selected_point_index = None
        
        # Colors for visualization (BGR format for OpenCV)
        self.stage_colors = [
            (255, 0, 0),    # Blue
            (0, 255, 0),    # Green
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
            (128, 0, 128),  # Purple
            (255, 165, 0),  # Orange
            (0, 128, 255),  # Orange-Red
        ]
        self.selected_color = (255, 255, 255)  # White for selected point
        
        print(f"Videos loaded: {[p.name for p in video_paths]}")
        print(f"Duration: {self.duration_s:.2f}s, Frames: {self.total_frames}, FPS: {fps}")
        print("\nInterval Labels:")
        for i, label in enumerate(self.interval_labels):
            print(f"  {i} = {label}")
        print("\nControls:")
        print("  SPACE - Play/Pause")
        print("  LEFT/RIGHT - Seek backward/forward (1 second)")
        print("  UP/DOWN - Seek backward/forward (1 frame)")
        print(f"  0-{len(self.interval_labels) - 1} - Label current time point with corresponding interval label")
        print("  S - Show current labels")
        print("  Q - Quit and save")
        print("  R - Reset all labels")
        print("  TAB - Select next point for deletion")
        print("  X/BACKSPACE - Delete selected point")
    
    def get_current_time(self) -> float:
        """Get current playback time in seconds."""
        return self.current_frame / self.fps
    
    def seek_to_time(self, time_s: float) -> None:
        """Seek to specific time in seconds."""
        frame = int(time_s * self.fps)
        frame = max(0, min(frame, self.total_frames - 1))
        self.current_frame = frame
        
        # Seek all video captures
        for cap in self.caps.values():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    def seek_relative(self, delta_s: float) -> None:
        """Seek relative to current position."""
        new_time = self.get_current_time() + delta_s
        self.seek_to_time(new_time)
    
    def label_current_time(self, stage_idx: int) -> None:
        """Label the current time point with a progress stage."""
        if 0 <= stage_idx < len(self.interval_labels):
            current_time = self.get_current_time()
            stage = self.interval_labels[stage_idx]
            
            self.labeled_points.append((current_time, stage))
            # Keep points sorted by time
            self.labeled_points.sort(key=lambda x: x[0])
            print(f"Labeled time point {current_time:.2f}s as '{stage}'")
    
    def get_intervals_from_points(self) -> Dict[str, List[List[float]]]:
        """Generate intervals from labeled time points.
        
        The interval (start, end) is labeled with the stage that ENDS at 'end' time.
        
        Returns:
            Dict mapping stage names to list of intervals [[start_time, end_time], ...].
        """
        if not self.labeled_points:
            return {}
        
        intervals = defaultdict(list)
        
        # Sort points by time, then by stage order for simultaneous points
        def sort_key(point):
            time, stage = point
            stage_order = self.interval_labels.index(stage) if stage in self.interval_labels else 999
            return (time, stage_order)
        
        sorted_points = sorted(self.labeled_points, key=sort_key)
        
        # Group points by time to handle simultaneous labels
        time_groups = []
        current_time = None
        current_group = []
        
        for time, stage in sorted_points:
            if current_time is None or abs(time - current_time) < 0.01:  # Same time (within 10ms)
                if current_time is None:
                    current_time = time
                current_group.append((time, stage))
            else:
                # New time group
                if current_group:
                    time_groups.append((current_time, current_group))
                current_time = time
                current_group = [(time, stage)]
        
        # Add the last group
        if current_group:
            time_groups.append((current_time, current_group))
        
        # Generate intervals
        prev_time = 0.0
        
        for group_time, group_stages in time_groups:
            # The first stage in the group (earliest in task_spec order) gets the interval
            if group_time > prev_time:
                first_stage = group_stages[0][1]
                intervals[first_stage].append([prev_time, group_time])
            
            # Any additional stages at the same time get zero-duration intervals
            for i in range(1, len(group_stages)):
                stage = group_stages[i][1]
                intervals[stage].append([group_time, group_time])
            
            prev_time = group_time
        
        return dict(intervals)
    
    def show_labels(self) -> None:
        """Display current labels."""
        print("\nCurrent time points:")
        if not self.labeled_points:
            print("  No points labeled yet.")
        else:
            for time, stage in sorted(self.labeled_points, key=lambda x: x[0]):
                print(f"  {time:.2f}s: {stage}")
        
        print("\nDerived intervals:")
        intervals = self.get_intervals_from_points()
        if not intervals:
            print("  No intervals yet.")
        else:
            for stage, stage_intervals in intervals.items():
                print(f"  {stage}:")
                for start, end in stage_intervals:
                    print(f"    {start:.2f}s - {end:.2f}s")
    
    def reset_labels(self) -> None:
        """Reset all labels."""
        self.labeled_points.clear()
        self.selected_point_index = None
        print("All labels reset.")
    
    def select_next_point(self) -> None:
        """Select the next point for deletion."""
        if not self.labeled_points:
            print("No points to select.")
            return
        
        if self.selected_point_index is None:
            self.selected_point_index = 0
        else:
            self.selected_point_index = (self.selected_point_index + 1) % len(self.labeled_points)
        
        time, stage = self.labeled_points[self.selected_point_index]
        print(f"Selected: point {self.selected_point_index} at {time:.2f}s ({stage})")
    
    def delete_selected_point(self) -> None:
        """Delete the selected point."""
        if self.selected_point_index is not None and 0 <= self.selected_point_index < len(self.labeled_points):
            time, stage = self.labeled_points[self.selected_point_index]
            del self.labeled_points[self.selected_point_index]
            print(f"Deleted point at {time:.2f}s ({stage})")
            
            # Adjust selection
            if len(self.labeled_points) == 0:
                self.selected_point_index = None
            elif self.selected_point_index >= len(self.labeled_points):
                self.selected_point_index = len(self.labeled_points) - 1
        else:
            print("No point selected. Press TAB to select a point first.")
    
    def create_combined_frame(self, frames: List[np.ndarray]) -> np.ndarray:
        """Combine multiple camera frames into a single display frame."""
        # Debug output on first call
        if not hasattr(self, '_frame_combine_logged'):
            print(f"Combining {len(frames)} frames into grid layout")
            self._frame_combine_logged = True
        
        if len(frames) == 1:
            # Even for single frame, make it larger
            frame = frames[0]
            h, w = frame.shape[:2]
            # Scale up by 2x for better visibility
            return cv2.resize(frame, (w * 2, h * 2))
        
        # Resize all frames to same height (much larger for visibility)
        target_height = 720  # Increased from 480 to 720 for better visibility
        resized_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            aspect = w / h
            new_w = int(target_height * aspect)
            resized = cv2.resize(frame, (new_w, target_height))
            resized_frames.append(resized)
        
        # Arrange frames in a grid
        if len(resized_frames) <= 2:
            # Horizontal layout for 1-2 cameras
            combined = np.hstack(resized_frames)
        elif len(resized_frames) <= 4:
            # 2x2 grid for 3-4 cameras
            row1 = np.hstack(resized_frames[:2])
            if len(resized_frames) == 3:
                # Pad with black frame
                black = np.zeros_like(resized_frames[0])
                row2 = np.hstack([resized_frames[2], black])
            else:
                row2 = np.hstack(resized_frames[2:4])
            combined = np.vstack([row1, row2])
        else:
            # 2x3 grid for 5-6 cameras
            row1 = np.hstack(resized_frames[:3])
            remaining = resized_frames[3:]
            # Pad if needed
            while len(remaining) < 3:
                remaining.append(np.zeros_like(resized_frames[0]))
            row2 = np.hstack(remaining[:3])
            combined = np.vstack([row1, row2])
        
        return combined
    
    def add_overlay_text(self, frame: np.ndarray) -> np.ndarray:
        """Add overlay text to frame."""
        frame = frame.copy()
        current_time = self.get_current_time()
        
        # Scale font size based on frame height for consistency
        h = frame.shape[0]
        font_scale = h / 600  # Base scale on 600px height
        
        # Status text
        status = "Playing" if self.is_playing else "Paused"
        text = f"{status} | Time: {current_time:.2f}s / {self.duration_s:.2f}s | Frame: {self.current_frame}/{self.total_frames}"
        cv2.putText(frame, text, (20, int(50 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale * 0.8, (0, 255, 0), max(2, int(2 * font_scale)))
        
        # Show number of labeled points
        text2 = f"Labeled points: {len(self.labeled_points)}"
        cv2.putText(frame, text2, (20, int(100 * font_scale)), cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale * 0.6, (0, 255, 255), max(2, int(2 * font_scale)))
        
        return frame
    
    def create_visualization_panel(self, width: int, height: int) -> np.ndarray:
        """Create a visualization panel showing the timeline and labeled intervals."""
        panel = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Get intervals
        intervals = self.get_intervals_from_points()
        
        # Calculate layout (increase left margin for longer label names with index numbers)
        # Scale margin and fonts based on panel height
        margin = max(250, int(height * 0.5))  # Scale with height
        font_scale = height / 300.0  # Base font scaling
        time_scale = (width - margin - 10) / self.duration_s
        # Ensure minimum spacing between labels - reserve space at top and bottom
        header_space = int(40 * font_scale)
        footer_space = 60
        available_height = height - header_space - footer_space
        stage_height = max(50, available_height // len(self.interval_labels))
        
        # Draw title at the top
        cv2.putText(panel, "Interval Labels Timeline", (10, int(25 * font_scale)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8 * font_scale, (200, 200, 200), max(2, int(2 * font_scale)))
        
        # Draw time axis
        cv2.line(panel, (margin, height - 40), (width - 10, height - 40), (255, 255, 255), 2)
        
        # Draw time markers
        for t in range(0, int(self.duration_s) + 1, max(1, int(self.duration_s / 10))):
            x = int(margin + t * time_scale)
            cv2.line(panel, (x, height - 45), (x, height - 35), (255, 255, 255), 2)
            cv2.putText(panel, f"{t}s", (x - int(20 * font_scale), height - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5 * font_scale, (255, 255, 255), max(1, int(font_scale)))
        
        # Draw stages and intervals
        for stage_idx, stage in enumerate(self.interval_labels):
            # Position y with offset for header
            y = header_space + stage_height * (stage_idx + 1)
            
            # Draw stage label with index number (positioned in the middle of its row)
            label_text = f"{stage_idx} {stage}"
            label_y = y - stage_height // 3  # Position label in upper portion of row
            cv2.putText(panel, label_text, (10, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6 * font_scale, (255, 255, 255), max(2, int(2 * font_scale)))
            
            # Draw stage baseline
            cv2.line(panel, (margin, y), (width - 10, y), (128, 128, 128), 2)
            
            # Draw intervals for this stage
            if stage in intervals:
                color_idx = stage_idx % len(self.stage_colors)
                stage_color = self.stage_colors[color_idx]
                
                for start, end in intervals[stage]:
                    start_x = int(margin + start * time_scale)
                    end_x = int(margin + end * time_scale)
                    
                    # Draw interval line (positioned in the middle between label and baseline)
                    interval_y = y - int(stage_height * 0.5)  # Middle of the row
                    line_thickness = max(5, int(5 * font_scale))
                    cv2.line(panel, (start_x, interval_y), (end_x, interval_y), stage_color, line_thickness)
                    
                    # Draw interval markers (circles at start and end)
                    marker_size = max(4, int(4 * font_scale))
                    cv2.circle(panel, (start_x, interval_y), marker_size, stage_color, -1)
                    cv2.circle(panel, (end_x, interval_y), marker_size, stage_color, -1)
        
        # Draw time points as vertical lines
        for point_idx, (time, stage) in enumerate(self.labeled_points):
            x = int(margin + time * time_scale)
            if margin <= x <= width - 10:
                # Find stage color
                stage_idx = self.interval_labels.index(stage) if stage in self.interval_labels else 0
                color_idx = stage_idx % len(self.stage_colors)
                point_color = self.stage_colors[color_idx]
                
                # Check if this point is selected
                if self.selected_point_index == point_idx:
                    point_color = self.selected_color
                    thickness = max(3, int(3 * font_scale))
                else:
                    thickness = max(2, int(2 * font_scale))
                
                # Draw vertical line for the time point (from header to bottom of stage area)
                top_y = header_space + 5
                bottom_y = height - footer_space
                cv2.line(panel, (x, top_y), (x, bottom_y), point_color, thickness)
                
                # Draw point marker at top (scaled)
                marker_y = header_space + int(10 * font_scale)
                cv2.circle(panel, (x, marker_y), max(5, int(5 * font_scale)), point_color, -1)
                
                # Draw stage label near the point at top (scaled)
                label_x = x + int(8 * font_scale)
                label_y = header_space + int(15 * font_scale)
                cv2.putText(panel, stage[:3], (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5 * font_scale, point_color, max(1, int(font_scale)))
        
        # Draw current time indicator
        current_time = self.get_current_time()
        current_x = int(margin + current_time * time_scale)
        if margin <= current_x <= width - 10:
            # Draw thicker cyan line for current time
            top_y = header_space
            cv2.line(panel, (current_x, top_y), (current_x, height - footer_space), (0, 255, 255), max(3, int(3 * font_scale)))
            # Draw time label at the top in a box for visibility
            time_label_y = header_space - int(5 * font_scale)
            cv2.putText(panel, f"{current_time:.1f}s", (current_x + 5, time_label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7 * font_scale, (0, 255, 255), max(2, int(2 * font_scale)))
        
        # Draw status indicator at the bottom right (scaled)
        status_text = f"Labeled: {len(self.labeled_points)} points"
        status_color = (0, 255, 0)
        status_x = width - int(300 * font_scale)
        status_y = height - 10
        cv2.putText(panel, status_text, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6 * font_scale, status_color, max(2, int(2 * font_scale)))
        
        return panel

    def play(self) -> Dict[str, List[List[float]]]:
        """Start interactive video player and return labeled intervals."""
        window_name = f"{self.title}" if self.title else "Video Player - Interval Labeling"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        # Note: Window will size automatically based on content
        # We've increased the actual frame sizes to make the display larger
        
        clock = cv2.getTickCount()
        
        # Debug: Print number of cameras being used
        print(f"Playing with {len(self.caps)} camera views: {list(self.caps.keys())}")
        
        try:
            while True:
                # Read frames from all videos
                frames = []
                all_frames_valid = True
                
                for video_name, cap in self.caps.items():
                    # Always seek to the current frame to ensure synchronization
                    cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                    ret, frame = cap.read()
                    
                    if not ret:
                        all_frames_valid = False
                        break
                    frames.append(frame)
                
                if not all_frames_valid:
                    # End of video - pause instead of looping
                    self.is_playing = False
                    print(f"Reached end of video at {self.get_current_time():.2f}s")
                    # Reset to last valid frame
                    if self.current_frame > 0:
                        self.current_frame -= 1
                        self.seek_to_time(self.get_current_time())
                    continue
                
                # Create combined display frame
                combined_frame = self.create_combined_frame(frames)
                
                # Add overlay information
                video_frame = self.add_overlay_text(combined_frame)
                
                # Create visualization panel (scale proportionally to video size)
                panel_height = max(400, video_frame.shape[0] // 4)  # At least 400px, or 1/4 of video height
                viz_panel = self.create_visualization_panel(video_frame.shape[1], panel_height)
                
                # Combine video and visualization vertically
                display_frame = np.vstack([video_frame, viz_panel])
                
                cv2.imshow(window_name, display_frame)
                
                # Check if window was closed externally
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    print("\nWindow was closed. Exiting labeler...")
                    break
                
                # Handle keyboard input
                key = cv2.waitKey(1 if self.is_playing else 0) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord(' '):
                    self.is_playing = not self.is_playing
                    print("Playing" if self.is_playing else "Paused")
                elif key == 81 or key == 2:  # Left arrow
                    self.seek_relative(-1.0)
                    self.is_playing = False
                elif key == 83 or key == 3:  # Right arrow
                    self.seek_relative(1.0)
                    self.is_playing = False
                elif key == 82 or key == 0:  # Up arrow
                    # Seek backward by 1 frame
                    self.current_frame = max(0, self.current_frame - 1)
                    self.seek_to_time(self.get_current_time())
                    self.is_playing = False
                elif key == 84 or key == 1:  # Down arrow
                    # Seek forward by 1 frame
                    self.current_frame = min(self.total_frames - 1, self.current_frame + 1)
                    self.seek_to_time(self.get_current_time())
                    self.is_playing = False
                elif key in [ord(str(i)) for i in range(len(self.interval_labels))]:
                    stage_idx = int(chr(key))
                    self.label_current_time(stage_idx)
                elif key == ord('s'):
                    self.show_labels()
                elif key == ord('r'):
                    self.reset_labels()
                elif key == 9:  # Tab
                    self.select_next_point()
                elif key == ord('x') or key == 8:  # x or Backspace
                    self.delete_selected_point()
                
                # Auto advance if playing
                if self.is_playing:
                    new_tick = cv2.getTickCount()
                    elapsed = (new_tick - clock) / cv2.getTickFrequency()
                    if elapsed >= 1.0 / self.fps:
                        self.current_frame += 1
                        if self.current_frame >= self.total_frames:
                            self.current_frame = self.total_frames - 1
                            self.is_playing = False
                            print("Reached end of video")
                        clock = new_tick
        
        except KeyboardInterrupt:
            print("\nCtrl+C detected. Saving labels and exiting gracefully...")
        
        cv2.destroyAllWindows()
        for cap in self.caps.values():
            cap.release()
        
        # Generate intervals from labeled points
        intervals = self.get_intervals_from_points()
        
        print(f"\n✓ Generated {sum(len(stage_intervals) for stage_intervals in intervals.values())} intervals from {len(self.labeled_points)} time points")
        
        return intervals


class EpisodeIntervalLabeler:
    """Manages labeling of time intervals across episodes."""
    
    def __init__(
        self,
        dataset: LeRobotDataset,
        interval_labels: List[str],
        metadata_key: str,
        metadata_file: Path,
        start_episode: int = 0,
        skip_labeled: bool = False,
    ):
        self.dataset = dataset
        self.interval_labels = interval_labels
        self.metadata_key = metadata_key
        self.metadata_file = metadata_file
        self.start_episode = start_episode
        self.skip_labeled = skip_labeled
        
        # Load existing metadata
        self.existing_metadata = self._load_existing_metadata()
        
        print(f"Available cameras: {dataset.meta.camera_keys}")
    
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
    
    def _save_metadata(self, existing: dict, intervals: dict) -> None:
        """Save or update metadata for an episode."""
        existing[self.metadata_key] = intervals
        
        # Update in-memory
        self.existing_metadata[existing["episode_index"]] = existing
        
        # Write all metadata back to file
        self.metadata_file.parent.mkdir(exist_ok=True, parents=True)
        with jsonlines.open(self.metadata_file, "w") as writer:
            for ep_idx in sorted(self.existing_metadata.keys()):
                writer.write(self.existing_metadata[ep_idx])
    
    def _get_video_paths(self, episode_index: int) -> List[Path]:
        """Get all video file paths for an episode."""
        video_paths = []
        
        print(f"\nSearching for videos for episode {episode_index}...")
        print(f"Available camera keys: {self.dataset.meta.camera_keys}")
        
        for camera_key in self.dataset.meta.camera_keys:
            video_path = Path(self.dataset.root) / self.dataset.meta.get_video_file_path(
                episode_index, camera_key
            )
            
            if video_path.exists():
                video_paths.append(video_path)
                print(f"  ✓ Found: {camera_key}")
            else:
                print(f"  ✗ Missing: {camera_key} (path: {video_path})")
        
        print(f"Total videos found: {len(video_paths)}")
        return video_paths
    
    def label_episode(self, episode_index: int) -> tuple[bool, str]:
        """Label a single episode. Returns (is_labeled, reason) tuple."""
        print("\n" + "=" * 80)
        print(f"Episode {episode_index} / {self.dataset.num_episodes - 1}")
        
        # Get episode info
        episode_info = self.dataset.meta.episodes[episode_index]
        print(f"\nEpisode info:")
        print(f"  Tasks: {episode_info['tasks']}")
        print(f"  Length: {episode_info['length']} frames")

        existing = {"episode_index": episode_index}
        
        # Print all existing metadata for this episode
        if episode_index in self.existing_metadata:
            existing = self.existing_metadata[episode_index]
            print(f"\nExisting metadata for this episode:")
            for key, value in existing.items():
                if key == "episode_index":
                    continue
                # Pretty print nested dicts
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for sub_key, sub_value in value.items():
                        print(f"    {sub_key}: {sub_value}")
                else:
                    print(f"  {key}: {value}")
            
            # Ask if user wants to skip before showing videos
            response = input("\nProceed with labeling this episode? (y/n): ").strip().lower()
            if response == 'n' or response == 'no':
                print("Skipping episode")
                return (False, "user_skip")
        else:
            print(f"\nNo existing metadata found for this episode.")
        
        # Check if episode already has interval metadata with our key
        if episode_index in self.existing_metadata:
            existing = self.existing_metadata[episode_index]
            
            # Check if metadata key exists
            if self.metadata_key in existing and existing[self.metadata_key] is not None:
                print(f"\nNote: Episode already has data for metadata key '{self.metadata_key}'")
                
                if self.skip_labeled:
                    print(f"✓ Auto-skipping (--skip-labeled flag is set)")
                    return (False, "interval_labeled")

                response = input(f"Overwrite existing content in metadata_key '{self.metadata_key}'? (y/n): ").strip().lower()
                if response == 'n' or response == 'no':
                    print("Keeping existing labels, skipping episode")
                    return (False, "user_skip")
        
        # Get all video paths
        video_paths = self._get_video_paths(episode_index)
        
        if not video_paths:
            print(f"Error: No videos found for episode {episode_index}")
            return (False, "no_videos")
        
        print(f"\nLoading {len(video_paths)} camera views...")
        
        # Create labeler and run interactive session
        labeler = TimeIntervalLabeler(
            video_paths=video_paths,
            interval_labels=self.interval_labels,
            fps=self.dataset.fps,
            title=f"Episode {episode_index}",
        )
        
        intervals = labeler.play()
        
        if not intervals:
            print("No intervals labeled, skipping episode")
            return (False, "no_intervals")
        
        # Confirm and save
        print("\nInterval labels to save:")
        print(f"  {self.metadata_key}:")
        for label, label_intervals in intervals.items():
            print(f"    {label}: {label_intervals}")
        
        # Wrap intervals under the metadata key
        self._save_metadata(existing, intervals)
        print(f"✓ Interval labels saved for episode {episode_index}")
        return (True, "labeled")
    
    def run(self) -> None:
        """Run the labeling process for all episodes."""
        print("=" * 80)
        print("Episode Interval Labeling Tool")
        print("=" * 80)
        print(f"Dataset: {self.dataset.repo_id}")
        print(f"Total episodes: {self.dataset.num_episodes}")
        print(f"Metadata key: {self.metadata_key}")
        print(f"Interval labels: {', '.join(self.interval_labels)}")
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
            if reason == "interval_labeled":
                episode_index += 1
                continue
            
            # Ask what to do next
            if episode_index < self.dataset.num_episodes - 1:
                print("\nOptions:")
                print("  [y] Continue to next episode")
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
    parser = argparse.ArgumentParser(description="Label time intervals in episodes")
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
        "--interval-labels",
        nargs="+",
        required=True,
        help="List of interval label names (e.g., 'approach' 'grasp' 'lift' 'place')"
    )
    parser.add_argument(
        "--metadata-key",
        type=str,
        required=True,
        help="Metadata key to store interval labels under (e.g., 'task_progress')"
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
        help="Automatically skip episodes that already have all interval labels"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="pyav",
        choices=["pyav", "torchcodec"],
        help="Video backend to use for dataset loading (default: pyav)"
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
    labeler = EpisodeIntervalLabeler(
        dataset=dataset,
        interval_labels=args.interval_labels,
        metadata_key=args.metadata_key,
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

