#!/usr/bin/env python

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
from functools import cached_property

from lerobot.teleoperators.widowxai_leader.config_widowxai_leader import WidowXAILeaderConfig
from lerobot.teleoperators.widowxai_leader.widowxai_leader import WidowXAILeader

from ..teleoperator import Teleoperator
from .config_bi_widowxai_leader import BiWidowXAILeaderConfig

logger = logging.getLogger(__name__)


class BiWidowXAILeader(Teleoperator):
    """
    [Bimanual WidowX AI Leader Arms](https://www.trossenrobotics.com/widowx-ai) by Trossen Robotics
    """

    config_class = BiWidowXAILeaderConfig
    name = "bi_widowxai_leader"

    def __init__(self, config: BiWidowXAILeaderConfig):
        super().__init__(config)
        self.config = config

        left_arm_config = WidowXAILeaderConfig(
            id=f"{config.id}_left" if config.id else None,
            ip_address=config.left_arm_ip_address,
        )

        right_arm_config = WidowXAILeaderConfig(
            id=f"{config.id}_right" if config.id else None,
            ip_address=config.right_arm_ip_address,
        )

        self.left_arm = WidowXAILeader(left_arm_config)
        self.right_arm = WidowXAILeader(right_arm_config)

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {f"left_{joint_name}.pos": float for joint_name in self.left_arm.config.joint_names} | {
            f"right_{joint_name}.pos": float for joint_name in self.right_arm.config.joint_names
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        # TODO(lukeschmitt-tr): Implement force feedback
        return {}

    @property
    def is_connected(self) -> bool:
        return self.left_arm.is_connected and self.right_arm.is_connected

    def connect(self, calibrate: bool = True) -> None:
        self.left_arm.connect(calibrate)
        self.right_arm.connect(calibrate)

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
        self.left_arm.configure()
        self.right_arm.configure()

    def get_action(self) -> dict[str, float]:
        action_dict = {}

        # Add "left_" prefix
        left_action = self.left_arm.get_action()
        action_dict.update({f"left_{key}": val for key, val in left_action.items()})

        # Add "right_" prefix
        right_action = self.right_arm.get_action()
        action_dict.update({f"right_{key}": val for key, val in right_action.items()})

        return action_dict

    def send_feedback(self, feedback: dict[str, float]) -> None:
        # Remove "left_" prefix
        left_feedback = {
            key.removeprefix("left_"): value for key, value in feedback.items() if key.startswith("left_")
        }
        # Remove "right_" prefix
        right_feedback = {
            key.removeprefix("right_"): value for key, value in feedback.items() if key.startswith("right_")
        }

        if left_feedback:
            self.left_arm.send_feedback(left_feedback)
        if right_feedback:
            self.right_arm.send_feedback(right_feedback)

    def disconnect(self) -> None:
        self.left_arm.disconnect()
        self.right_arm.disconnect()
