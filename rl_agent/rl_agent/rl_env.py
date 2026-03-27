import gymnasium as gym
from gymnasium import spaces
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
import threading
import time
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Bool

# How many LIDAR rays to use (downsampled from 1080)
NUM_LIDAR_RAYS = 181

class F1TenthEnv(gym.Env):
    """
    Gymnasium environment wrapping the F1Tenth ROS2 simulator.
    
    Observation: 108 downsampled LIDAR rays (normalized 0-1)
    Action:      [steering_angle, speed]
    Reward:      speed - collision_penalty - jerk_penalty
    """

    def __init__(self):
        super().__init__()

        self.max_speed = 1.5

        # --- Spaces ---
        # Observations: 108 LIDAR rays, each normalized between 0 and 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(NUM_LIDAR_RAYS,),
            dtype=np.float32
        )
        # Actions: [steering (-0.4 to 0.4 rad), speed (0 to 4 m/s)]
        self.action_space = spaces.Box(
            low=np.array([-0.4, 0.0], dtype=np.float32),
            high=np.array([0.4, 4.0], dtype=np.float32),
        )

        # --- Internal state ---
        self.latest_scan = None         # most recent LIDAR scan
        self.latest_velocity = 0.0      # most recent forward speed
        self.collision_detected = False # True if car has crashed
        self.prev_steering = 0.0        # for jerk penalty
        self.episode_start_time = None

        # --- ROS2 setup ---
        rclpy.init()
        self.node = Node('rl_training_env')

        self.scan_sub = self.node.create_subscription(
            LaserScan, '/scan', self._scan_callback, 10)
        self.odom_sub = self.node.create_subscription(
            Odometry, '/ego_racecar/odom', self._odom_callback, 10)
        self.collision_sub = self.node.create_subscription(
            Bool, '/collision', self._collision_callback, 10)
        self.drive_pub = self.node.create_publisher(
            AckermannDriveStamped, '/drive', 10)
        self.reset_pub = self.node.create_publisher(
            PoseWithCovarianceStamped, '/initialpose', 10)

        # Spin ROS2 in background thread so callbacks fire continuously
        self.ros_thread = threading.Thread(
            target=rclpy.spin, args=(self.node,), daemon=True)
        self.ros_thread.start()

        # Wait for first scan to arrive
        print("Waiting for LIDAR data...")
        while self.latest_scan is None:
            time.sleep(0.1)
        print("LIDAR data received. Environment ready.")

    # ------------------------------------------------------------------
    # ROS2 Callbacks
    # ------------------------------------------------------------------

    def _scan_callback(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        ranges = ranges[:1080]
        indices = np.linspace(0, 1079, 181, dtype=int)
        ranges = ranges[indices]
        self.latest_scan = (ranges / msg.range_max).astype(np.float32)

    def _collision_callback(self, msg: Bool):
        if msg.data:
            if self.episode_start_time is None:
                return
            if time.time() - self.episode_start_time < 0.5:
                return
            self.collision_detected = True

    def _odom_callback(self, msg: Odometry):
        """Store the current forward speed."""
        self.latest_velocity = abs(msg.twist.twist.linear.x)

    # ------------------------------------------------------------------
    # Gymnasium Interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._publish_drive(0.0, 0.0)
        time.sleep(0.2)  # reduced from 0.3

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose.position.x = 0.0
        pose_msg.pose.pose.position.y = 0.0
        pose_msg.pose.pose.position.z = 0.0
        pose_msg.pose.pose.orientation.w = 1.0
        self.reset_pub.publish(pose_msg)

        time.sleep(0.5)  # reduced from 1.0

        self.collision_detected = False
        self.prev_steering = 0.0
        self.episode_start_time = time.time()

        time.sleep(0.2)  # reduced from 0.3
        self.collision_detected = False

        return self._get_observation(), {}

    def step(self, action):
        steering = float(action[0])
        speed = float(action[1])
        speed = float(np.clip(speed, 0.0, self.max_speed))

        self._publish_drive(steering, speed)
        time.sleep(0.1)

        # If collision detected, stop immediately
        if self.collision_detected:
            self._publish_drive(0.0, 0.0)  # stop the car
            return self._get_observation(), -50.0, True, False, {}

        obs = self._get_observation()
        reward = self._compute_reward(steering, speed)
        terminated = self.collision_detected
        self.prev_steering = steering
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_observation(self):
        if self.latest_scan is None:
            return np.zeros(NUM_LIDAR_RAYS, dtype=np.float32)
        return self.latest_scan.copy()
    
    def set_max_speed(self, max_speed):
        self.max_speed = max_speed

    def _compute_reward(self, steering, speed):
        # survival bonus
        reward = 0.5

        # forward speed reward
        reward += self.latest_velocity * 0.5

        # jerk penalty
        reward -= 0.3 * abs(steering - self.prev_steering)

        # wall proximity penalty
        min_clearance = float(np.min(self.latest_scan))
        if min_clearance < 0.1:
            reward -= 2.0 * (0.1 - min_clearance)

        return float(reward)

    def _publish_drive(self, steering, speed):
        """Publish an AckermannDriveStamped command."""
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = float(steering)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)

    def close(self):
        """Clean up ROS2 resources."""
        self.node.destroy_node()
        rclpy.shutdown()