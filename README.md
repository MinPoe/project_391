# Milestone 3 - On-Car Safe Reactive Driving using Computer Vision

## Overview
This milestone 3 package builds on the previous milestones, implementing gap-following and AEB with Computer Vision. This milestone package also implements lap counting with a tunable parameter to control how many laps the car can do.

The milestone1_py launch file runs the old AEB and wall-following nodes:
`ros2 launch milestone2 milestone1_py.py`

The milestone2_py launch file runs the old AEB and gap-following nodes:
`ros2 launch milestone2 milestone2_py.py`

The milestone3_py launch file runs the new AEB, gap-following, and lap counting nodes:
`ros2 launch milestone3 milestone3_py.py`

## Milestone 3 Nodes

### 1) safety_node (AEB)
**Inputs**
- `/camera/depth/image_rect_raw` (Image)
- `/odom` (Odometry)

**Output**
- `/drive` (AckermannDriveStamped)
- `/kys` (Bool)

**Core Idea**
- By splitting the camera feed into vertical chunks, we can calculate time-to-collision (TTC) by treating these chunks like an array, and calculating the estimated distance from the car to each chunk. With the calculated TTC, progressive braking is applied to slow the car down at specific TTC thresholds, stopping the car completely if the TTC is low enough, or if the distance to the nearest object is too low. There are 3 stages total to progressive braking, two of which slow the car and the final stage is a full stop. 

**Key Changes Compared to LiDAR Algorithm**
1) 

### 2) cam_node (Gap Following)
**Input**
- `/camera/color/image_raw` (Image)
- `/kys` (Bool)

**Output**
- `/ranges` (Float32MultiArray)
- `/drive` (AckermannDriveStamped)

**Core Idea**
- This node is multifunctional. It takes in a raw image from the camera and applies different filters to it to create a new image which we can extract information from. From the edited image, this node directs the car to drive into the open space.

**Key Changes Compared to LiDAR Algorithm**
1) 

### 3) lap_counting (Lap Counting)
**Input**
- `/camera/depth/image_rect_raw` (Image)

**Output**
- `/drive` (AckermannDriveStamped)

**Core Idea**
- This node counts the number of laps the car has driven. It has a reference picture indicating the start of a lap, which the node compares to constantly as the car drives. When the node detects a certain cosine similarity to the starting point, the car will recognize a lap as complete. 

**Algorithm**
1) Obtain the current image from the car.
2) Resize and flatten the reference and current image.
3) Compute the dot product of the reference and current image (call this dotProduct).
4) Normalize each of the image vectors (call these normCurrent and normReference).
5) Compute the cosine similarity with the algorithm: dotProduct/(normCurrent * normReference)
6) If the cosine similarity of the 2 images is higher then the threshold, then add a lap to the count.

## Parameters
### safety_node
- 

### cam_node
- 

### lap_counter
- 

## General Testing Strategies
- Camera-based AEB:
  - 
- Camera-based Gap following:
  - 
- Lap Counting:
  - 
- Parameter sweep:
  - 
- Edge cases:
  - 

## Parameter Tuning/Derivation Strategies:
- Camera-based AEB:
  - 
- Camera-based Gap following:
  - 
- Lap Counting:
  - 

## RQT Graph
![ROS graph]()

## How to Run

### Run the milestone 1 nodes (Old AEB + Wall following)
```bash
colcon build
source install/local_setup.bash
ros2 launch milestone2 milestone1_py.py
```

### Run the milestone 2 nodes (Old AEB + Gap following)
```bash
colcon build
source install/local_setup.bash
ros2 launch milestone2 milestone2_py.py
```

### Run the milestone 3 nodes (New AEB + Gap following + Lap Counting)
```bash
colcon build
source install/local_setup.bash
ros2 launch milestone2 milestone3_py.py
```

### Change parameters
```bash
# AEB node

# Gap following node
```
---