import time
from dataclasses import replace
import torch

import trossen_slate as slate
from lerobot.common.robot_devices.cameras.utils import make_cameras_from_configs
from lerobot.common.robot_devices.motors.utils import MotorsBus, make_motors_buses_from_configs
from lerobot.common.robot_devices.robots.utils import get_arm_id
import threading
import logging

from lerobot.common.robot_devices.robots.configs import TrossenAIMobileConfig
from lerobot.common.robot_devices.robots.configs import ManipulatorRobotConfig
from lerobot.common.robot_devices.robots.utils import get_arm_id
from lerobot.common.robot_devices.utils import RobotDeviceAlreadyConnectedError, RobotDeviceNotConnectedError
from lerobot.common.robot_devices.motors.dynamixel import TorqueMode

def ensure_safe_goal_position(
    goal_pos: torch.Tensor, present_pos: torch.Tensor, max_relative_target: float | list[float]
):
    # Cap relative action target magnitude for safety.
    diff = goal_pos - present_pos
    max_relative_target = torch.tensor(max_relative_target)
    safe_diff = torch.minimum(diff, max_relative_target)
    safe_diff = torch.maximum(safe_diff, -max_relative_target)
    safe_goal_pos = present_pos + safe_diff

    if not torch.allclose(goal_pos, safe_goal_pos):
        logging.warning(
            "Relative goal position magnitude had to be clamped to be safe.\n"
            f"  requested relative goal position target: {diff}\n"
            f"    clamped relative goal position target: {safe_diff}"
        )

    return safe_goal_pos

class TrossenAIMobile():

    def __init__(self, config: TrossenAIMobileConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            self.config = TrossenAIMobileConfig(**kwargs)
        else:
            # Overwrite config arguments using kwargs
            self.config = replace(config, **kwargs)
        self.robot_type = self.config.type
        self.leader_arms = make_motors_buses_from_configs(self.config.leader_arms)
        self.follower_arms = make_motors_buses_from_configs(self.config.follower_arms)
        self.cameras = make_cameras_from_configs(self.config.cameras)
        self.is_connected = False
        self.teleop = None
        self.logs = {}

        self.base = slate.TrossenSlate()
        self.base.init_base()

        self.read_data = False

        self.state_keys = None
        self.action_keys = None

        self.log_data = slate.ChassisData()
        self.update_thread = None

    def get_motor_names(self, arm: dict[str, MotorsBus]) -> list:
        return [f"{arm}_{motor}" for arm, bus in arm.items() for motor in bus.motors]
    
    @property
    def camera_features(self) -> dict:
        cam_ft = {}
        for cam_key, cam in self.cameras.items():
            key = f"observation.images.{cam_key}"
            cam_ft[key] = {
                "shape": (cam.height, cam.width, cam.channels),
                "names": ["height", "width", "channels"],
                "info": None,
            }
        return cam_ft

    @property
    def motor_features(self) -> dict:
        action_names = ['linear_vel', 'angular_vel'] + self.get_motor_names(self.leader_arms)
        state_names = ['odom_x', 'odom_y', 'odom_theta', 'linear_vel', 'angular_vel'] + self.get_motor_names(self.leader_arms)
        return {
            "action": {
                "dtype": "float32",
                "shape": (len(action_names),),
                "names": action_names,
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def features(self):
        return {**self.motor_features, **self.camera_features}

    @property
    def has_camera(self):
        return len(self.cameras) > 0

    @property
    def num_cameras(self):
        return len(self.cameras)

    @property
    def available_arms(self):
        available_arms = []
        for name in self.follower_arms:
            arm_id = get_arm_id(name, "follower")
            available_arms.append(arm_id)
        for name in self.leader_arms:
            arm_id = get_arm_id(name, "leader")
            available_arms.append(arm_id)
        return available_arms

    def teleop_safety_stop(self):
        if self.robot_type in ["trossen_ai_stationary", "trossen_ai_solo"]:
            for arms in self.leader_arms:
                self.leader_arms[arms].write("Reset", 1)
            for arms in self.follower_arms:
                self.follower_arms[arms].write("Reset", 1)
            time.sleep(2)
            for arms in self.leader_arms:
                self.leader_arms[arms].write("Torque_Enable", 0)
            for arms in self.follower_arms:
                self.follower_arms[arms].write("Torque_Enable", 1)

    def connect(self) -> None:
        self.base.init_base()
        self.is_connected = True

        # Start Reading
        self.read_data = True
        self.update_thread = threading.Thread(target=self.update_values)
        self.update_thread.start()

        if self.is_connected:
            raise RobotDeviceAlreadyConnectedError(
                "ManipulatorRobot is already connected. Do not run `robot.connect()` twice."
            )

        if not self.leader_arms and not self.follower_arms and not self.cameras:
            raise ValueError(
                "ManipulatorRobot doesn't have any device to connect. See example of usage in docstring of the class."
            )

        # Connect the arms
        for name in self.follower_arms:
            print(f"Connecting {name} follower arm.")
            self.follower_arms[name].connect()
        for name in self.leader_arms:
            print(f"Connecting {name} leader arm.")
            self.leader_arms[name].connect()
        
        time.sleep(2)

        # We assume that at connection time, arms are in a rest position, and torque can
        # be safely disabled to run calibration and/or set robot preset configurations.
        for name in self.follower_arms:
            self.follower_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)
        for name in self.leader_arms:
            self.leader_arms[name].write("Torque_Enable", TorqueMode.DISABLED.value)

        # Enable torque on all motors of the follower arms
        for name in self.follower_arms:
            print(f"Activating torque on {name} follower arm.")
            self.follower_arms[name].write("Torque_Enable", 1)

        # Check both arms can be read
        for name in self.follower_arms:
            self.follower_arms[name].read("Present_Position")
        for name in self.leader_arms:
            self.leader_arms[name].read("Present_Position")

        # Connect the cameras
        for name in self.cameras:
            self.cameras[name].connect()

        self.is_connected = True

    def update_values(self):
        while self.read_data:
            self.base.update_state()
            self.base.read(self.log_data)

    def get_state(self) -> dict:
        return {
            "odom_x": self.log_data.odom_x,
            "odom_y": self.log_data.odom_y,
            "odom_theta": self.log_data.odom_z,
            "linear_vel": self.log_data.vel_x,
            "angular_vel": self.log_data.vel_z,
        }

    def teleop_step(
        self, record_data=False
    ) -> None | tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "TrossenAIMobile is not connected. You need to run `robot.connect()`."
            )

        # Prepare to assign the position of the leader to the follower
        leader_pos = {}
        for name in self.leader_arms:
            before_lread_t = time.perf_counter()
            leader_pos[name] = self.leader_arms[name].read("Present_Position")
            leader_pos[name] = torch.from_numpy(leader_pos[name])
            self.logs[f"read_leader_{name}_pos_dt_s"] = time.perf_counter() - before_lread_t

        # Send goal position to the follower
        follower_goal_pos = {}
        for name in self.follower_arms:
            before_fwrite_t = time.perf_counter()
            goal_pos = leader_pos[name]

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Used when record_data=True
            follower_goal_pos[name] = goal_pos

            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)
            self.logs[f"write_follower_{name}_goal_pos_dt_s"] = time.perf_counter() - before_fwrite_t

       
        if not record_data:
            return

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        before_read_t = time.perf_counter()
        base_state = self.get_state()
        base_action = [state['linear_vel'], state['angular_vel']]
        self.logs["read_base_dt_s"] = time.perf_counter() - before_read_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state.append(base_state.values())
        state = torch.cat(state)

        # Create action by concatenating follower goal position
        action = []
        for name in self.follower_arms:
            if name in follower_goal_pos:
                action.append(follower_goal_pos[name])
        action.append(base_action)
        action = torch.cat(action)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries
        obs_dict, action_dict = {}, {}
        obs_dict["observation.state"] = state
        action_dict["action"] = action
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        return obs_dict, action_dict

    def capture_observation(self):

        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        # Read follower position
        follower_pos = {}
        for name in self.follower_arms:
            before_fread_t = time.perf_counter()
            follower_pos[name] = self.follower_arms[name].read("Present_Position")
            follower_pos[name] = torch.from_numpy(follower_pos[name])
            self.logs[f"read_follower_{name}_pos_dt_s"] = time.perf_counter() - before_fread_t

        # Read base state
        before_read_t = time.perf_counter()
        base_state = self.get_state()
        self.logs["read_base_dt_s"] = time.perf_counter() - before_read_t

        # Create state by concatenating follower current position
        state = []
        for name in self.follower_arms:
            if name in follower_pos:
                state.append(follower_pos[name])
        state.append(base_state.values())
        state = torch.cat(state)

        # Capture images from cameras
        images = {}
        for name in self.cameras:
            before_camread_t = time.perf_counter()
            images[name] = self.cameras[name].async_read()
            images[name] = torch.from_numpy(images[name])
            self.logs[f"read_camera_{name}_dt_s"] = self.cameras[name].logs["delta_timestamp_s"]
            self.logs[f"async_read_camera_{name}_dt_s"] = time.perf_counter() - before_camread_t

        # Populate output dictionaries and format to pytorch
        obs_dict = {}
        obs_dict["observation.state"] = state
        for name in self.cameras:
            obs_dict[f"observation.images.{name}"] = images[name]

        obs_dict = {}
        obs_dict["observation.state"] = state

        return obs_dict

    def send_action(self, action):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()`."
            )

        from_idx = 0
        to_idx = 0
        action_sent = []
        for name in self.follower_arms:
            # Get goal position of each follower arm by splitting the action vector
            to_idx += len(self.follower_arms[name].motor_names)
            goal_pos = action[from_idx:to_idx]
            from_idx = to_idx

            # Cap goal position when too far away from present position.
            # Slower fps expected due to reading from the follower.
            if self.config.max_relative_target is not None:
                present_pos = self.follower_arms[name].read("Present_Position")
                present_pos = torch.from_numpy(present_pos)
                goal_pos = ensure_safe_goal_position(goal_pos, present_pos, self.config.max_relative_target)

            # Save tensor to concat and return
            action_sent.append(goal_pos)

            # Send goal position to each follower
            goal_pos = goal_pos.numpy().astype(np.float32)
            self.follower_arms[name].write("Goal_Position", goal_pos)

        linear_vel, angular_vel = action.tolist()
        before_write_t = time.perf_counter()
        self.base.set_cmd_vel(linear_vel, angular_vel)
        self.logs["write_base_dt_s"] = time.perf_counter() - before_write_t

        action_sent.append([linear_vel, angular_vel])

        return torch.cat(action_sent)

    def disconnect(self):
        if not self.is_connected:
            raise RobotDeviceNotConnectedError(
                "ManipulatorRobot is not connected. You need to run `robot.connect()` before disconnecting."
            )

        for name in self.follower_arms:
            self.follower_arms[name].disconnect()

        for name in self.leader_arms:
            self.leader_arms[name].disconnect()
        
        time.sleep(2)

        for name in self.cameras:
            self.cameras[name].disconnect()
        
        self.read_data = False
        if self.update_thread is not None:
            self.update_thread.join()
        self.is_connected = False

    def __del__(self):
        self.disconnect()