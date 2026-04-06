
## Overview
This repository contains the `project` ROS2 Python package for the CPEN 391 final project.

The project explores whether a learning-based driving agent can be built for the F1TENTH car with limited data, limited hardware time, and a safety layer around the learned policy. The original proposal focused on a model-free Reinforcement Learning (RL) driving agent trained from a short collection of real-world data. After more research and early implementation work, the project plan changed to use Behavioural Cloning (BC) as a supervised-learning baseline, then use Soft Actor-Critic (SAC) to refine the learned policy.

The final implementation includes a data preprocessing pipeline, a BC training/inference workflow, SAC training and demo nodes, saved model checkpoints, and a LiDAR safety node that filters learned drive commands before they reach the car.

## Project Package
- Package name: `project`
- Main launch files:
  - `launch/bc_py.py`
  - `launch/sac_train_py.py`
  - `launch/sac_demo_py.py`
- Main executables:
  - `safety_node`
  - `bc_inference_node`
  - `sac_train_node`
  - `sac_demo_node`

## Project Goals
- Train a learning-based driving agent for the F1TENTH car
- Use LiDAR as the main policy input
- Output steering angle and speed commands from the learned model
- Keep the car safe using an independent safety node
- Make the workflow usable for both simulator testing and physical-car deployment

## Did We Meet Our Goals?
- We completed the core pipeline for learning-based driving: ROS bag collection, preprocessing, BC training, SAC training, SAC demo inference, and safety filtering.
- We implemented the BC baseline that was added after the proposal. This became an important step because pure model-free RL would have required more training time and data than was realistic.
- We implemented SAC infrastructure with actor/critic networks, replay buffer, reward function, checkpoint saving/loading, online training, and demo inference.
- We kept safety separate from the learned policy by having BC/SAC publish to `/drive_raw`, while `safety_node` publishes the final `/drive` command.
- The main change from the proposal was that we did not go directly from limited real-world data to a pure model-free RL policy. Instead, we used BC first, then used SAC as a refinement stage.
- We were also able to train the RL model independently in simulation by running multiple simulator instances across different team members’ laptops for over 8 hours, which significantly improved training efficiency.

## What Changed From The Proposal
- Original proposal: train a model-free RL driving agent from a short amount of real-world F1TENTH data.
- Updated plan: collect driving data, train a Behavioural Cloning policy, then use Soft Actor-Critic to refine the policy.
- Reason for change: model-free RL usually needs a large amount of interaction data, while the project had limited physical car time and safety constraints.

## What Went Well
- 
- 
- 
- 
- 

## Issues and Limitations
- The original RL-only plan was difficult to implement on the physical car due to limited access to the track and safety constraints; instead, a BC model was used for warm-up.
- Real-world data collection was also limited by physical car access and limited time on the track.
- 
- 
- 


## System Architecture
### Data Flow
- Data collection records `/scan`, `/drive`, and `/odom` or `/ego_racecar/odom`
- `preprocessing/bag_to_csv.py` converts ROS2 bag data into synchronized CSV rows
- `preprocessing/preprocess.py` cleans, downsamples, augments, and normalizes the dataset
- `bc/train.py` trains the Behavioural Cloning model
- `sac_train_node` trains or refines the SAC policy online
- `bc_inference_node` or `sac_demo_node` publishes learned commands to `/drive_raw`
- `safety_node` reads `/drive_raw`, applies LiDAR safety checks, and publishes final commands to `/drive`

### Nodes and Topics

#### `safety_node`
Inputs:
- `/drive_raw` (`AckermannDriveStamped`)
- `/scan` (`LaserScan`)
- `/odom` or `/ego_racecar/odom` (`Odometry`)

Outputs:
- `/drive` (`AckermannDriveStamped`)
- `/kys` (`Bool`)

Role:
- Computes obstacle distance and time-to-collision using LiDAR and odometry
- Applies partial braking and full braking
- Adds wall-avoidance steering bias when side clearance is low
- Publishes the final safe drive command to `/drive`
- Publishes `/kys` when an emergency stop condition is detected

#### `bc_inference_node`
Inputs:
- `/scan` (`LaserScan`)
- `/kys` (`Bool`)

Outputs:
- `/drive_raw` (`AckermannDriveStamped`)

Role:
- Loads the trained BC model from `bc/bc_model.pth`
- Loads normalization scalers from `processed/scalers.npz`
- Downsamples and normalizes LiDAR scans
- Predicts steering angle and speed from the BC network
- Stops when `/kys` is latched

#### `sac_train_node`
Inputs:
- `/scan` (`LaserScan`)
- `/ego_racecar/odom` (`Odometry`)
- `/kys` (`Bool`)

Outputs:
- `/drive_raw` (`AckermannDriveStamped`)
- `/initialpose` (`PoseWithCovarianceStamped`)

Role:
- Runs online SAC training in the simulator
- Stores transitions in a replay buffer
- Computes rewards using speed, wall clearance, steering smoothness, and crash state
- Saves checkpoints periodically and when a best episode is reached
- Resets the simulator car when `/kys` indicates a crash or emergency stop

#### `sac_demo_node`
Inputs:
- `/scan` (`LaserScan`)
- `/kys` (`Bool`)

Outputs:
- `/drive_raw` (`AckermannDriveStamped`)

Role:
- Loads the best SAC checkpoint from `sac/sac_checkpoint_best.pth`
- Runs the SAC actor in deterministic inference mode
- Publishes learned steering and speed commands to `/drive_raw`
- Stops when `/kys` is latched

## Models
### Behavioural Cloning
- Model file: `bc/model.py`
- Training script: `bc/train.py`
- Checkpoint: `bc/bc_model.pth`
- Architecture: LiDAR input -> 256 ReLU -> 128 ReLU -> 2 outputs
- Outputs: steering angle and speed

### Soft Actor-Critic
- Model file: `sac/model.py`
- Trainer file: `sac/train_sac.py`
- Reward file: `sac/reward.py`
- Checkpoints:
  - `sac/sac_checkpoint.pth`
  - `sac/sac_checkpoint_best.pth`
- Actor: Gaussian policy with tanh squashing to keep actions in `[0, 1]`
- Critics: two Q-networks for SAC training

## Preprocessing
The preprocessing pipeline:
- Converts ROS2 bags to CSV
- Synchronizes scan, drive, and odometry messages by timestamp
- Clips invalid or infinite LiDAR values
- Removes stationary samples
- Downsamples LiDAR readings to reduce the input size
- Mirrors LiDAR scans and steering angles for data augmentation
- Normalizes LiDAR and action values
- Saves scalers to `processed/scalers.npz`

## Parameters
### `safety_node`
- `odom_topic`: odometry topic to read from (default: `/odom`)
- `distance_threshold`: full-brake distance threshold in meters (default: 0.4)
- `ttc_pb1`: TTC threshold for partial braking stage 1 (default: 1.85)
- `ttc_pb2`: TTC threshold for partial braking stage 2 (default: 1.55)
- `ttc_fb`: TTC threshold for full braking (default: 0.8)
- `side_margin`: side clearance margin for wall steering correction (default: 0.7)
- `wall_steer_gain`: gain for wall steering bias (default: 0.35)
- `max_wall_steer_bias`: maximum steering bias from wall avoidance (default: 0.18)

### `bc_inference_node`
- `model_path`: path to BC model weights (default: `bc/bc_model.pth`)
- `scalers_path`: path to normalization scalers (default: `processed/processed_simulator/scalers.npz`)
- `max_speed`: maximum commanded speed (default: 1.0)
- `min_speed`: minimum nonzero commanded speed (default: 0.5)
- `safety_distance`: reserved safety distance parameter (default: 0.3)

### `sac_train_node`
- `bc_weights_path`: optional BC weights path for initializing the SAC actor
- `scalers_path`: path to normalization scalers
- `initial_checkpoint_path`: checkpoint to initialize from
- `checkpoint_path`: checkpoint save path
- `log_path`: training CSV log path
- `max_speed`: maximum commanded speed (default: 2.0)
- `min_speed`: minimum nonzero commanded speed (default: 0.5)
- `deterministic`: whether to use deterministic actions during training (default: false)
- `resume_training`: whether to resume from an existing checkpoint (default: false)
- `lr_actor`: actor learning rate (default: 0.0001)
- `lr_critic`: critic learning rate (default: 0.0003)
- `gamma`: reward discount factor (default: 0.99)
- `tau`: target network update rate (default: 0.005)
- `buffer_size`: replay buffer size (default: 100000)
- `batch_size`: SAC batch size (default: 256)
- `update_every`: training update interval in steps (default: 10)
- `warmup_steps`: warmup steps before regular SAC action sampling (default: 2000)
- `learning_starts`: step count before critic updates begin (default: 3000)
- `actor_learning_starts`: step count before actor updates begin (default: 10000)
- `bc_reg_weight`: BC regularization weight (default: 2.0)
- `bc_reg_decay_steps`: BC regularization decay steps (default: 50000)
- `save_every`: checkpoint save interval in steps (default: 5000)
- `reset_x`: simulator reset x position (default: 0.0)
- `reset_y`: simulator reset y position (default: 0.0)
- `reset_yaw`: simulator reset yaw angle (default: 0.0)

### `sac_demo_node`
- `checkpoint_path`: path to SAC checkpoint
- `scalers_path`: path to normalization scalers
- `max_speed`: maximum commanded speed (default: 1.0)
- `min_speed`: minimum nonzero commanded speed (default: 0.5)

## Testing Strategy
### Data Pipeline Testing
- Recorded ROS2 bag data from `/scan`, `/drive`, and `/odom`
- Converted bag files into CSV files
- Checked timestamp synchronization between scan, drive, and odometry messages
- Verified that LiDAR values were clipped and normalized before training
- Mirrored data to reduce one-direction driving bias

### Behavioural Cloning Testing
- Trained the BC model on processed LiDAR and drive-command data
- Checked train and validation loss during training
- Loaded the saved BC model in `bc_inference_node`
- Verified that the inference node produced steering and speed commands on `/drive_raw`

### SAC Testing
- Initialized SAC from a checkpoint or BC policy when available
- Ran simulator training with replay buffer updates and checkpoint saving
- Used reward terms for survival, speed, wall clearance, steering smoothness, and crash penalty
- Loaded the best SAC checkpoint in `sac_demo_node` for deterministic inference

### Safety Testing
- Verified that learned commands go through `/drive_raw` before reaching `/drive`
- Used TTC and distance thresholds to trigger partial braking or full braking
- Used `/kys` to stop BC/SAC nodes during emergency-stop conditions
- Tested different odometry topic settings for simulator and physical-car use


## Build and Run
```bash
colcon build --packages-select project
source install/local_setup.bash
```

Run Behavioural Cloning inference:
```bash
ros2 launch project bc_py.py
```

Run SAC training in the simulator:
```bash
ros2 launch project sac_train_py.py
```

Run SAC demo inference:
```bash
ros2 launch project sac_demo_py.py
```

## Train Behavioural Cloning
```bash
python bc/train.py --data processed/data.csv --epochs 100 --batch-size 256 --lr 1e-3 --out bc/bc_model.pth
```

## Preprocess Data
```bash
python preprocessing/bag_to_csv.py --bag data/my_bag --output training_data.csv --max-time-diff 50
python preprocessing/preprocess.py
```

## Runtime Parameter Changes
```bash
# Safety node
ros2 param set /safety_node distance_threshold <value>
ros2 param set /safety_node ttc_pb1 <value>
ros2 param set /safety_node ttc_pb2 <value>
ros2 param set /safety_node ttc_fb <value>
ros2 param set /safety_node side_margin <value>
ros2 param set /safety_node wall_steer_gain <value>
ros2 param set /safety_node max_wall_steer_bias <value>

# BC inference node
ros2 param set /bc_inference_node max_speed <value>
ros2 param set /bc_inference_node min_speed <value>

# SAC demo node
ros2 param set /sac_demo_node max_speed <value>
ros2 param set /sac_demo_node min_speed <value>

# SAC training node
ros2 param set /sac_train_node max_speed <value>
ros2 param set /sac_train_node min_speed <value>
ros2 param set /sac_train_node deterministic <true_or_false>
ros2 param set /sac_train_node lr_actor <value>
ros2 param set /sac_train_node lr_critic <value>
ros2 param set /sac_train_node bc_reg_weight <value>
```
