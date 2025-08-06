### Teleoperation

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

### Record Episodes

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

### Replay Episode

```bash
  python -m lerobot.replay \
    --robot.type=bi_widowxai_follower \
    --robot.left_arm_ip_address=192.168.1.5 \
    --robot.right_arm_ip_address=192.168.1.4 \
    --robot.id=bimanual_follower \
    --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube_01 \
    --dataset.episode=0 # choose the episode you want to replay
```

### Train Policy


#### PI0

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
#### SmolVLA

```bash
  python -m lerobot.scripts.train \
  --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube \
  --policy.type=pi0 \
  --output_dir=outputs/train/smolvla_bimanual_widowxai \
  --job_name=smolvla_bimanual_widowxai \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_smolvla--batch_size=2
```

#### ACT

```bash
  python -m lerobot.scripts.train \
  --dataset.repo_id=TrossenRoboticsCommunity/bimanual-widowxai-handover-cube \
  --policy.type=pi0 \
  --output_dir=outputs/train/act_bimanual_widowxai \
  --job_name=act_bimanual_widowxai \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.repo_id=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_act --batch_size=2
```

### Inference (SMOLVLA)


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
  --dataset.repo_id=TrossenRoboticsCommunity/eval_bimanual_widowxai_handover_cube_smolvla \
  --dataset.num_episodes=2\
  --dataset.single_task="G" \
  --dataset.reset_time_s=2 \
  --dataset.episode_time_s=60 \
  --dataset.num_episodes=10 \
  --policy.path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_smolvla \
  --robot.min_time_to_move_multiplier=6.0
```



### Async Inference 

For more information and usage examples, see the [LeRobot Async documentation](https://huggingface.co/docs/lerobot/async).


Start policy server

```bash
python src/lerobot/scripts/server/policy_server.py \
    --host=127.0.0.1 \
    --port=8080 \
```

Start robot client


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
    --policy_type=smolvla \ 
    --pretrained_name_or_path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_smolvla\
    --policy_device=cuda \ 
    --actions_per_chunk=50 \
    --chunk_size_threshold=0.5 \
    --aggregate_fn_name=weighted_average \
    --debug_visualize_queue_size=True 
```

### Server Client (Incase the above command is not working)



```bash
python src/lerobot/scripts/server/robot_client.py --server_address=127.0.0.1:8080 --robot.type=bi_widowxai_follower --robot.left_arm_ip_address=192.168.1.5 --robot.right_arm_ip_address=192.168.1.4 --robot.id=bimanual_follower --robot.cameras='{top: {"type": "opencv", "index_or_path": 12, "width": 640,"height": 480, "fps": 30}, bottom: {"type": "opencv", "index_or_path": 18, "width": 640, "height": 480, "fps": 30},right: {"type": "opencv", "index_or_path": 6, "width": 640, "height": 480, "fps": 30}, left: {"type": "opencv", "index_or_path": 24, "width": 640, "height": 480, "fps": 30}, }'    --task="dummy" --policy_type=smolvla --pretrained_name_or_path=TrossenRoboticsCommunity/bimanual_widowxai_handover_cube_smolvla --policy_device=cuda    --actions_per_chunk=50 --chunk_size_threshold=0.5 --aggregate_fn_name=weighted_average --debug_visualize_queue_size=True 
  ```