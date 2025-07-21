# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import logging
import time
from functools import cached_property
from typing import Any

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.widowxai_follower import WidowXAIFollower
from lerobot.robots.widowxai_follower.config_widowxai_follower import WidowXAIFollowerConfig

from ..robot import Robot
from .config_bi_widowxai_follower import BiWidowXAIFollowerConfig

logger = logging.getLogger(__name__)


class BiWidowXAIFollower(Robot):
    """
    [Bimanual WidowX AI Follower Arms](https://www.trossenrobotics.com/widowx-ai) by Trossen Robotics
    """

    config_class = BiWidowXAIFollowerConfig
    name = "bi_widowxai_follower"

    def __init__(self, config: BiWidowXAIFollowerConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = WidowXAIFollowerConfig(
            id=f"{config.id}_left" if config.id else None,
            ip_address=config.left_arm_ip_address,
            max_relative_target=config.left_arm_max_relative_target,
            min_time_to_move_multiplier=config.min_time_to_move_multiplier,
            loop_rate=config.loop_rate,
            cameras={},
        )

        right_arm_config = WidowXAIFollowerConfig(
            id=f"{config.id}_right" if config.id else None,
            ip_address=config.right_arm_ip_address,
            max_relative_target=config.right_arm_max_relative_target,
            min_time_to_move_multiplier=config.min_time_to_move_multiplier,
            loop_rate=config.loop_rate,
            cameras={},
        )

        self.left_arm = WidowXAIFollower(left_arm_config)
        self.right_arm = WidowXAIFollower(right_arm_config)

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joint_ft(self) -> dict[str, type]:
        return (
            {f"left_{motor}.pos": float for motor in self.left_arm.config.joint_names}
            | {f"right_{motor}.pos": float for motor in self.right_arm.config.joint_names}
            | {f"left_{joint_name}.vel": float for joint_name in self.left_arm.config.joint_names}
            | {f"right_{joint_name}.vel": float for joint_name in self.right_arm.config.joint_names}
            | {f"left_{joint_name}.eff": float for joint_name in self.left_arm.config.joint_names}
            | {f"right_{joint_name}.eff": float for joint_name in self.right_arm.config.joint_names}
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._joint_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._joint_ft

    @property
    def is_connected(self) -> bool:
        return (
            self.left_arm.is_connected
            and self.right_arm.is_connected
            and all(cam.is_connected for cam in self.cameras.values())
        )

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration but we check both arms for consistency
        return self.left_arm.is_calibrated and self.right_arm.is_calibrated

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration but we call calibrate on both arms for
        # consistency
        self.left_arm.calibrate()
        self.right_arm.calibrate()

    def configure(self) -> None:
        # Set the arm to position control mode
        self.left_arm.configure()
        self.right_arm.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Add "left_" prefix
        left_obs = self.left_arm.get_observation()
        obs_dict.update({f"left_{key}": val for key, val in left_obs.items()})

        # Add "right_" prefix
        right_obs = self.right_arm.get_observation()
        obs_dict.update({f"right_{key}": val for key, val in right_obs.items()})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        # Remove "left_" prefix
        left_action = {
            key.removeprefix("left_"): value for key, value in action.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_action = {
            key.removeprefix("right_"): value for key, value in action.items() if key.startswith("right_")
        }

        send_action_left = self.left_arm.send_action(left_action)
        send_action_right = self.right_arm.send_action(right_action)

        # Add prefixes back
        prefixed_send_action_left = {f"left_{key}": value for key, value in send_action_left.items()}
        prefixed_send_action_right = {f"right_{key}": value for key, value in send_action_right.items()}

        return {**prefixed_send_action_left, **prefixed_send_action_right}

    def disconnect(self):
        self.left_arm.disconnect()
        self.right_arm.disconnect()

        for cam in self.cameras.values():
            cam.disconnect()
