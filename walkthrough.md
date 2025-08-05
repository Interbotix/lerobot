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


  ```bash
  python -m lerobot.record \
  --robot.type=bi_widowxai_follower \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=bimanual_follower \
  --robot.id=bimanual_follower \
    --robot.cameras='{
    top: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
    bottom: {"type": "opencv", "index_or_path": 18, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}, 
    left: {"type": "opencv", "index_or_path": 24, "width": 640, "height": 480, "fps": 30},

    }' \
  --teleop.type=bi_widowxai_leader \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube \
  --dataset.num_episodes=25\
  --dataset.single_task="Grab and handover the red cube to the other arm" \
  --dataset.episode_time_s=10 --dataset.reset_time_s=2 --resume=true
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
  --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube \
  --policy.type=pi0 \
  --output_dir=outputs/train/pi0_bimanual_widowxai \
  --job_name=pi0_bimanual_widowxai \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_pi0 --batch_size=2
  ```


  ```bash
  python -m lerobot.record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \ # <- Use your port
  --robot.id=my_blue_follower_arm \ # <- Use your robot id
  --robot.cameras="{ front: {type: opencv, index_or_path: 8, width: 640, height: 480, fps: 30}}" \ # <- Use your cameras
  --dataset.single_task="Grasp a lego block and put it in the bin." \ # <- Use the same task description you used in your dataset recording
  --dataset.repo_id=${HF_USER}/eval_DATASET_NAME_test \  # <- This will be the dataset name on HF Hub
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \

  --policy.path=HF_USER/FINETUNE_MODEL_NAME # <- Use your fine-tuned model
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
  --display_data=true \
  --dataset.repo_id=TrossenRoboticsCommunity/eval_bimanual_widowxai_handover_cube_pi0_new_00 \
  --dataset.num_episodes=2\
  --dataset.single_task="G" \
  --dataset.reset_time_s=2 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=10 \
  --policy.path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_pi0_new \
  --robot.min_time_to_move_multiplier=8.0
  ```


```bash

python src/lerobot/scripts/server/robot_client.py \
    --server_address=127.0.0.1:8080 \ 
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
    --task="transfer red block" \
    --policy_type=pi0 \ 
    --pretrained_name_or_path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_pi0 \
    --policy_device=cuda \ 
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True 
```


  ```bash
  python -m lerobot.record \
  --robot.type=bi_widowxai_follower \
  --robot.left_arm_ip_address=192.168.1.5 \
  --robot.right_arm_ip_address=192.168.1.4 \
  --robot.id=bimanual_follower \
    --robot.cameras='{
    top: {"type": "opencv", "index_or_path": 12, "width": 640, "height": 480, "fps": 30},
    bottom: {"type": "opencv", "index_or_path": 18, "width": 640, "height": 480, "fps": 30},
    right: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}, 
    left: {"type": "opencv", "index_or_path": 24, "width": 640, "height": 480, "fps": 30},

    }' \
  --display_data=false \
  --dataset.repo_id=TrossenRoboticsCommunity/eval_bimanual_widowxai_handover_cube_pi0_00 \
  --teleop.type=bi_widowxai_leader \
  --teleop.left_arm_ip_address=192.168.1.3 \
  --teleop.right_arm_ip_address=192.168.1.2 \
  --teleop.id=bimanual_leader \
  --dataset.num_episodes=2\
  --dataset.single_task="G" \
  --dataset.reset_time_s=2 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=10 \
  --policy.path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_pi0 \
  --robot.min_time_to_move_multiplier=6.0
  ```
