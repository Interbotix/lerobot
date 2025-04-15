import cv2
from pathlib import Path
from lerobot.common.datasets.video_utils import decode_video_frames
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.utils import load_info
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# ---- CONFIGURATION ----
REPO_ID = "MTX_01/trossen_ai_stationary_test"           # Replace with your actual repo_id
EPISODE_INDEX = 0                          # Choose the episode to play

# ---- LOAD DATASET ----
dataset = LeRobotDataset(
    repo_id=REPO_ID,
    root=HF_LEROBOT_HOME / REPO_ID,
    episodes=[EPISODE_INDEX],
    download_videos=True,
)
print(dataset.num_frames)
print(dataset.fps)
video_path = HF_LEROBOT_HOME / REPO_ID / dataset.meta.get_video_file_path(0, 'cam_high')
print(video_path)
# Load the video frames
decode_video_frames(
    video_path,]
    
)