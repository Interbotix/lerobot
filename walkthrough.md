--- Detected Cameras ---
Camera #0:
  Name: Intel RealSense D405
  Type: RealSense
  Id: 218622270304
  Firmware version: 5.12.14.100
  Usb type descriptor: 3.2
  Physical port: /sys/devices/pci0000:80/0000:80:14.0/usb4/4-8/4-8.1/4-8.1:1.0/video4linux/video18
  Product id: 0B5B
  Product line: D400
  Default stream profile:
    Stream_type: Depth
    Format: z16
    Width: 848
    Height: 480
    Fps: 30
--------------------
Camera #1:
  Name: Intel RealSense D405
  Type: RealSense
  Id: 128422271347
  Firmware version: 5.17.0.9
  Usb type descriptor: 3.2
  Physical port: /sys/devices/pci0000:80/0000:80:14.0/usb4/4-8/4-8.2/4-8.2:1.0/video4linux/video6
  Product id: 0B5B
  Product line: D400
  Default stream profile:
    Stream_type: Depth
    Format: z16
    Width: 848
    Height: 480
    Fps: 30
--------------------
Camera #2:
  Name: Intel RealSense D405
  Type: RealSense
  Id: 130322272628
  Firmware version: 5.17.0.9
  Usb type descriptor: 3.2
  Physical port: /sys/devices/pci0000:80/0000:80:14.0/usb4/4-8/4-8.3/4-8.3:1.0/video4linux/video0
  Product id: 0B5B
  Product line: D400
  Default stream profile:
    Stream_type: Depth
    Format: z16
    Width: 848
    Height: 480
    Fps: 30
--------------------
Camera #3:
  Name: Intel RealSense D405
  Type: RealSense
  Id: 218622274938
  Firmware version: 5.16.0.1
  Usb type descriptor: 3.2
  Physical port: /sys/devices/pci0000:80/0000:80:14.0/usb4/4-8/4-8.4/4-8.4:1.0/video4linux/video12
  Product id: 0B5B
  Product line: D400
  Default stream profile:
    Stream_type: Depth
    Format: z16
    Width: 848
    Height: 480
    Fps: 30
--------------------

```bash
python -m lerobot.teleoperate \
  --robot.type=bi_widowxai_follower \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=bimanual_follower \
    --robot.cameras='{
    top: {"type": "intelrealsense", "serial_number_or_name": 218622270304, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    bottom: {"type": "intelrealsense", "serial_number_or_name": 130322272628, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    right: {"type": "intelrealsense", "serial_number_or_name": 128422271347, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    left: {"type": "intelrealsense", "serial_number_or_name": 218622274938, "width": 640, "height": 480, "fps": 30, "use_depth": True},

    }' \
  --teleop.type=bi_widowxai_leader \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=bimanual_leader \
  --display_data=true

```

```bash
python -m lerobot.teleoperate \
  --robot.type=bi_widowxai_follower \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=bimanual_follower \
  --teleop.type=bi_widowxai_leader \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --teleop_time_s=10.0
  ```


  ```bash
  python -m lerobot.record \
  --robot.type=bi_widowxai_follower \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=bimanual_follower \
    --robot.cameras='{
    top: {"type": "intelrealsense", "serial_number_or_name": 218622270304, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    bottom: {"type": "intelrealsense", "serial_number_or_name": 130322272628, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    right: {"type": "intelrealsense", "serial_number_or_name": 128422271347, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    left: {"type": "intelrealsense", "serial_number_or_name": 218622274938, "width": 640, "height": 480, "fps": 30, "use_depth": True},

    }' \
  --teleop.type=bi_widowxai_leader \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=TrossenRoboticsCommunity/velocity_test \
  --dataset.num_episodes=25\
  --dataset.single_task="Test the maximum velocity of the arm" \
  --dataset.episode_time_s=20 --dataset.reset_time_s=2 --resume=true
  ```


  ```bash
  python -m lerobot.replay \
    --robot.type=bi_widowxai_follower \
    --robot.left_arm_ip_address=192.168.1.5 \
    --robot.right_arm_ip_address=192.168.1.4 \
    --robot.id=bimanual_follower \
    --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube_01 \
    --dataset.episode=0 # choose the episode you want to replay
  ```



  ```bash
  python -m lerobot.scripts.train \
  --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube_01 \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_bimanual_widowxai_test \
  --job_name=pi0_bimanual_widowxai_test \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_pi0
  ```


  ```bash
  python -m lerobot.record  \
    --robot.type=bi_widowxai_follower \
    --robot.left_arm_ip_address=192.168.1.5 \
    --robot.right_arm_ip_address=192.168.1.4 \
    --robot.cameras='{
    top: {"type": "opencv", "index_or_path": 10, "width": 640, "height": 480, "fps": 30},
    bottom: {"type": "opencv", "index_or_path": 16, "width": 640, "height": 480, "fps": 30},
    left: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 22, "width": 640, "height": 480, "fps": 30}
  }' \
  --robot.id=bimanual_follower \
  --display_data=false \
  --dataset.repo_id=TrossenRoboticsCommunity/eval_bimanual_widowxai_25 \
  --dataset.single_task="Grab and handover the red cube to the other arm" \
  --policy.path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube \
  --policy.temporal_ensemble_coeff=1 \
  --policy.temporal_ensemble_coeff=0.1 \
  --dataset.num_image_writer_processes=1 \
  --dataset.reset_time_s=5.0 \
  --robot.min_time_to_move_multiplier=3.0 \
  --resume=true

  ```

        --robot.cameras='{
    top: {"type": "intelrealsense", "serial_number_or_name": 218622270304, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    bottom: {"type": "intelrealsense", "serial_number_or_name": 130322272628, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    right: {"type": "intelrealsense", "serial_number_or_name": 128422271347, "width": 640, "height": 480, "fps": 30, "use_depth": True},
    left: {"type": "intelrealsense", "serial_number_or_name": 218622274938, "width": 640, "height": 480, "fps": 30, "use_depth": True},

    }' \
