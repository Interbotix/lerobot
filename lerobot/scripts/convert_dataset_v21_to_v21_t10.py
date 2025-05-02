"""
This script modifies datasets for the LeRobot project by converting joint angles from degrees to radians
and reverting the scaling of gripper states to their original millimeter representation. It updates the
dataset's actions, states, and statistics, and saves the modified dataset locally or pushes it to the
Hugging Face Hub.

Specifically, this script converts the LeRobot v2.1 dataset to LeRobot v2.1 subversion Trossen v1.0.
The v2.1 dataset used degrees for joint angles and applied scaling to the gripper values, while the
Trossen v1.0 subversion uses radians for joint angles and removes the scaling for gripper values.
This conversion is tailored for datasets involving Trossen robotic arms.

Usage:
- Execute the script with the following arguments:
    --repo_id: Repository ID of the dataset to modify.
    --push_to_hub: Flag to push the modified dataset to the Hugging Face Hub.
    --private: Flag to upload the dataset to a private repository on the Hugging Face Hub.
    --tags: Optional tags for the dataset on the Hugging Face Hub.

Example:
        python convert_dataset_v21_to_v21_t10.py --repo_id my_dataset --push_to_hub --private --tags "lerobot" "tutorial"
"""

import argparse
import json

import numpy as np
import torch

from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# This scaling factor was used to prevent the gripper values from vanishing
# to zero in the dataset when converted to integer.
SCALING_FACTOR = 10000


class DatasetModifier:
    def __init__(self, repo_id, root, push_to_hub, private=False, tags=None):
        self.repo_id = repo_id
        self.push_to_hub = push_to_hub
        self.private = private
        self.tags = tags
        self.root = HF_LEROBOT_HOME if root is None else root
        self.dataset_dir = self.root / repo_id
        self.meta_folder = self.dataset_dir / "meta" if self.dataset_dir else None
        self.info_file = self.meta_folder / "info.json" if self.meta_folder else None
        self.episodes_stats_path = self.meta_folder / "episodes_stats.jsonl" if self.meta_folder else None
        self.total_episodes = 0
        self.episodes_stats = []

    def load_metadata(self):
        if not self.info_file or not self.info_file.exists():
            raise FileNotFoundError(f"{self.info_file} not found.")

        with open(self.info_file) as f:
            info_data = json.load(f)
            self.total_episodes = info_data.get("total_episodes", 0)

        if not self.episodes_stats_path or not self.episodes_stats_path.exists():
            raise FileNotFoundError(f"{self.episodes_stats_path} not found.")

        with open(self.episodes_stats_path) as f:
            self.episodes_stats = [json.loads(line) for line in f]

    @staticmethod
    def transform_arm_data(array):
        """
        Transforms arm data by converting joint angles to radians and scaling the gripper values.

        This function splits the input array into two equal parts representing the right and left arms.
        It converts all joint angles (except the gripper values) from degrees to radians and scales
        the gripper values by dividing them by 10,000.
        """
        arm_segment_length = len(array) // 2
        right_arm = np.copy(array[:arm_segment_length])
        left_arm = np.copy(array[arm_segment_length:])

        # Convert to radians except the gripper of each arm
        right_arm[:-1] = np.deg2rad(right_arm[:-1])
        left_arm[:-1] = np.deg2rad(left_arm[:-1])

        # Revert the scaling of the gripper of each arm
        right_arm[-1] /= SCALING_FACTOR
        left_arm[-1] /= SCALING_FACTOR

        return np.concatenate([right_arm, left_arm])

    @staticmethod
    def compute_stats(data_entries):
        data = np.array(data_entries)
        return {
            "min": data.min(axis=0).tolist(),
            "max": data.max(axis=0).tolist(),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "count": [len(data)],
        }

    def update_codebase_subversion(self, new_subversion):
        """
        Updates/Adds the codebase subversion in the info.json file.

        Args:
            new_subversion (str): The new subversion to set for the codebase.
        """
        with open(self.info_file) as f:
            info_data = json.load(f)

        current_version = info_data.get("codebase_version", "unknown")
        print(f"Current codebase version: {current_version}")

        # Insert trossen_subversion just after codebase_version
        updated_info_data = {}
        for key, value in info_data.items():
            updated_info_data[key] = value
            if key == "codebase_version":
                updated_info_data["trossen_subversion"] = new_subversion

        with open(self.info_file, "w") as f:
            json.dump(updated_info_data, f, indent=4)

        print(f"Updated trossen_subversion to: {new_subversion}")

    def modify_dataset(self):
        for episode_index in range(self.total_episodes):
            dataset = LeRobotDataset(
                repo_id=self.repo_id,
                root=self.root if self.root != HF_LEROBOT_HOME else None,
                episodes=[episode_index],
                edit_mode=True,
            )

            modified_actions = []
            modified_states = []

            def modify_entry(entry, actions_list, states_list):
                modified_action = self.transform_arm_data(entry["action"])
                modified_state = self.transform_arm_data(entry["observation.state"])

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
            new_action_stats = self.compute_stats(modified_actions)
            new_state_stats = self.compute_stats(modified_states)

            for ep_stat in self.episodes_stats:
                if ep_stat["episode_index"] == episode_index:
                    ep_stat["stats"]["action"] = new_action_stats
                    ep_stat["stats"]["observation.state"] = new_state_stats
                    break

        # Save updated stats
        with open(self.episodes_stats_path, "w") as f:
            for ep_stat in self.episodes_stats:
                f.write(json.dumps(ep_stat) + "\n")

        print(f"Updated episodes_stats saved to: {self.episodes_stats_path}")

        update_codebase_subversion = "v1.0"

        self.update_codebase_subversion(update_codebase_subversion)

        if self.push_to_hub:
            dataset.push_to_hub(repo_id=self.repo_id, private=self.private, tags=self.tags)
            print(f"Dataset pushed to Hugging Face Hub: {self.repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify dataset and update stats.")
    parser.add_argument("--repo_id", type=str, required=True, help="Repository ID of the dataset.")
    parser.add_argument("--root", type=str, default=None, help="Root directory for the dataset.")
    parser.add_argument("--push_to_hub", action="store_true", help="Flag to load dataset from hub.")
    parser.add_argument(
        "--private", action="store_true", help="Upload on private repository on the Hugging Face hub."
    )
    parser.add_argument("--tags", type=str, nargs="+", help="Add tags to your dataset on the hub.")
    args = parser.parse_args()

    modifier = DatasetModifier(
        repo_id=args.repo_id,
        root=args.root,
        push_to_hub=args.push_to_hub,
        private=args.private,
        tags=args.tags,
    )
    modifier.load_metadata()
    modifier.modify_dataset()
