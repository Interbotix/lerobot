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

from trossen_slate import TrossenSlate

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.robots.bi_widowxai_follower import BiWidowXAIFollower, BiWidowXAIFollowerConfig

from ..robot import Robot
from .config_mobileai import MobileAIConfig

logger = logging.getLogger(__name__)


class MobileAI(Robot):
    """
    [Mobile AI](https://www.trossenrobotics.com/mobile-ai) by Trossen Robotics
    """

    config_class = MobileAIConfig
    name = "bi_widowxai_follower"

    def __init__(self, config: MobileAIConfig):
        super().__init__(config)
        self.config = config

        arms_config = BiWidowXAIFollowerConfig(
            left_arm_ip_address=config.left_arm_ip_address,
            right_arm_ip_address=config.right_arm_ip_address,
            left_arm_max_relative_target=config.left_arm_max_relative_target,
            right_arm_max_relative_target=config.right_arm_max_relative_target,
            min_time_to_move_multiplier=config.min_time_to_move_multiplier,
            loop_rate=config.loop_rate,
            cameras={},
        )

        self.arms = BiWidowXAIFollower(arms_config)
        self.base = TrossenSlate()

        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _joint_ft(self) -> dict[str, type]:
        arms_ft = self.arms._joint_ft
        base_ft = {"x.vel": float, "theta.vel": float}
        return {**arms_ft, **base_ft}

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
        return self.arms.is_connected and all(cam.is_connected for cam in self.cameras.values())

    def connect(self, calibrate: bool = True) -> None:
        self.arms.connect(calibrate=calibrate)
        base_init_success, message = self.base.init_base()
        if not base_init_success:
            raise ConnectionError(f"Failed to connect to Mobile AI base: {message}")

        for cam in self.cameras.values():
            cam.connect()

    @property
    def is_calibrated(self) -> bool:
        # Trossen Arm robots do not require calibration but we check both arms for consistency
        return self.arms.is_calibrated

    def calibrate(self) -> None:
        # Trossen Arm robots do not require calibration but we call calibrate on both arms for
        # consistency
        self.arms.calibrate()

    def configure(self) -> None:
        # Set the arm to position control mode
        self.arms.configure()

    def get_observation(self) -> dict[str, Any]:
        obs_dict = {}

        # Get arm observations
        arms_obs = self.arms.get_observation()
        obs_dict.update(arms_obs)

        # Get base observations
        base_obs = self.base.get_vel()
        obs_dict.update({"x.vel": base_obs[0], "theta.vel": base_obs[1]})

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.async_read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        send_action_arms = self.arms.send_action(
            {k: v for k, v in action.items() if k in self.arms.action_features}
        )
        action_base_x_vel = action.get("x.vel", 0.0)
        action_base_theta_vel = action.get("theta.vel", 0.0)
        self.base.set_cmd_vel(action_base_x_vel, action_base_theta_vel)

        return {**send_action_arms, "x.vel": action_base_x_vel, "theta.vel": action_base_theta_vel}

    def disconnect(self):
        self.arms.disconnect()
        self.base.set_cmd_vel(0.0, 0.0)

        for cam in self.cameras.values():
            cam.disconnect()
