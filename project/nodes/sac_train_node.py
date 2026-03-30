"""SAC training node (simulator).

Drives the car, collects transitions, and trains actor/critics online.
Episode boundaries come from the safety node's /kys topic. When /kys
latches, the episode ends and the car is reset to the starting pose.
Training resumes automatically when the safety node releases /kys.

Launch:
    ros2 launch project sac_train_py.py
"""

import math
import os
import csv
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseWithCovarianceStamped

from project.sac.model import SACActorNet, SACCriticNet
from project.sac.train_sac import SACTrainer
from project.sac.reward import compute_reward

LIDAR_STEP = 6
MAX_RANGE = 10.0


class SACTrainNode(Node):

    def __init__(self):
        super().__init__("sac_train_node")

        # ---- parameters ----
        self.declare_parameter("bc_weights_path", "")
        self.declare_parameter("scalers_path", "")
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("log_path", "")
        self.declare_parameter("max_speed", 2.0)
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("deterministic", False)
        self.declare_parameter("lr_actor", 1e-4)
        self.declare_parameter("lr_critic", 3e-4)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("tau", 0.005)
        self.declare_parameter("buffer_size", 100000)
        self.declare_parameter("batch_size", 256)
        self.declare_parameter("update_every", 10)
        self.declare_parameter("warmup_steps", 2000)
        self.declare_parameter("save_every", 5000)
        self.declare_parameter("reset_x", 0.0)
        self.declare_parameter("reset_y", 0.0)
        self.declare_parameter("reset_yaw", 0.0)

        # ---- read parameters ----
        bc_weights_path = self._str("bc_weights_path")
        scalers_path = self._str("scalers_path")
        self.checkpoint_path = self._str("checkpoint_path")
        self.log_path = self._str("log_path")
        self.max_speed = self._dbl("max_speed")
        self.min_speed = self._dbl("min_speed")
        self.deterministic = self._bool("deterministic")
        lr_actor = self._dbl("lr_actor")
        lr_critic = self._dbl("lr_critic")
        gamma = self._dbl("gamma")
        tau = self._dbl("tau")
        buffer_size = self._int("buffer_size")
        batch_size = self._int("batch_size")
        self.update_every = self._int("update_every")
        self.warmup_steps = self._int("warmup_steps")
        self.save_every = self._int("save_every")
        self.reset_x = self._dbl("reset_x")
        self.reset_y = self._dbl("reset_y")
        self.reset_yaw = self._dbl("reset_yaw")

        # ---- load scalers ----
        scalers = np.load(scalers_path)
        self.lidar_scale = scalers["lidar_scale"].astype(np.float32)
        self.lidar_min = scalers["lidar_min"].astype(np.float32)
        self.action_scale = scalers["action_scale"].astype(np.float32)
        self.action_min = scalers["action_min"].astype(np.float32)
        self.num_lidar = len(self.lidar_scale)
        self.get_logger().info(f"LiDAR features: {self.num_lidar}")

        # ---- build / load networks ----
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actor = SACActorNet(self.num_lidar)
        critic1 = SACCriticNet(self.num_lidar)
        critic2 = SACCriticNet(self.num_lidar)
        self.trainer = SACTrainer(
            actor, critic1, critic2,
            state_dim=self.num_lidar, lr_actor=lr_actor, lr_critic=lr_critic,
            lr_alpha=lr_critic, gamma=gamma, tau=tau,
            buffer_size=buffer_size, batch_size=batch_size, device=device,
        )

        if os.path.isfile(self.checkpoint_path):
            self.get_logger().info(f"Resuming from checkpoint: {self.checkpoint_path}")
            self.trainer.load(self.checkpoint_path)
        elif bc_weights_path and os.path.isfile(bc_weights_path):
            self.get_logger().info(f"Initialising actor from BC: {bc_weights_path}")
            bc_actor = SACActorNet.from_bc(bc_weights_path, self.num_lidar, device=device)
            self.trainer.actor.load_state_dict(bc_actor.state_dict())
        else:
            self.get_logger().warn("No checkpoint or BC weights -- random init")

        self.get_logger().info(
            f"SAC TRAIN ready | deterministic={self.deterministic} device={device}"
        )

        # ---- state tracking ----
        self.prev_state = None
        self.prev_action = None
        self.prev_raw_lidar = None
        self.prev_steering = 0.0
        self.prev_prev_steering = 0.0
        self.current_speed = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count = 0
        self.best_episode_steps = 0
        self.stopped = False

        self._init_log()

        # ---- ROS interface ----
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10)
        self.kys_sub = self.create_subscription(
            Bool, "/kys", self.kys_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10)

    # ---- param helpers ----
    def _str(self, n):
        return self.get_parameter(n).get_parameter_value().string_value

    def _dbl(self, n):
        return self.get_parameter(n).get_parameter_value().double_value

    def _int(self, n):
        return self.get_parameter(n).get_parameter_value().integer_value

    def _bool(self, n):
        return self.get_parameter(n).get_parameter_value().bool_value

    # ------------------------------------------------------------------ #
    #  Scan callback                                                      #
    # ------------------------------------------------------------------ #

    def scan_callback(self, msg: LaserScan):
        if self.stopped:
            self._publish_stop()
            return

        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        # --- preprocess LiDAR ---
        ds = raw_ranges[::LIDAR_STEP]
        ds = np.where(np.isfinite(ds), ds, MAX_RANGE)
        ds = np.clip(ds, 0.0, MAX_RANGE)
        if len(ds) > self.num_lidar:
            ds = ds[:self.num_lidar]
        elif len(ds) < self.num_lidar:
            ds = np.pad(ds, (0, self.num_lidar - len(ds)),
                        constant_values=MAX_RANGE)
        raw_lidar = ds.copy()
        state = ds * self.lidar_scale + self.lidar_min

        # --- store previous transition ---
        if self.prev_state is not None:
            reward = compute_reward(
                self.prev_raw_lidar, self.current_speed,
                self.prev_steering, done=False,
                prev_steering=self.prev_prev_steering,
            )
            self.trainer.store(
                self.prev_state, self.prev_action, reward, state, False)
            self.episode_reward += reward
            self.episode_steps += 1

        # --- select action ---
        if self.step_count < self.warmup_steps:
            action = np.random.uniform(0.0, 1.0, size=2).astype(np.float32)
        else:
            state_t = torch.from_numpy(state.reshape(1, -1)).to(
                self.trainer.device)
            action = (self.trainer.actor.get_action(state_t, self.deterministic)
                      .cpu().numpy()[0])

        # --- denormalise ---
        steering = float((action[0] - self.action_min[0]) / self.action_scale[0])
        speed = float((action[1] - self.action_min[1]) / self.action_scale[1])
        speed = max(self.min_speed, min(speed, self.max_speed))

        # --- publish drive ---
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        # --- bookkeeping ---
        self.prev_state = state
        self.prev_action = action
        self.prev_raw_lidar = raw_lidar
        self.prev_prev_steering = self.prev_steering
        self.prev_steering = steering
        self.step_count += 1

        if self.step_count <= 10:
            self.get_logger().info(
                f"[DRIVE #{self.step_count}] steer={steering:.4f} "
                f"speed={speed:.4f}")

        # --- gradient step ---
        if self.step_count % self.update_every == 0:
            metrics = self.trainer.update()
            if metrics and self.step_count % 200 == 0:
                self.get_logger().info(
                    f"[step {self.step_count}] "
                    f"c1={metrics['critic1_loss']:.4f} "
                    f"c2={metrics['critic2_loss']:.4f} "
                    f"actor={metrics['actor_loss']:.4f} "
                    f"alpha={metrics['alpha']:.4f}")

        # --- periodic checkpoint ---
        if self.step_count % self.save_every == 0:
            self.trainer.save(self.checkpoint_path)
            self.get_logger().info(
                f"Checkpoint saved (step {self.step_count}, "
                f"buffer {len(self.trainer.buffer)})")

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg: Odometry):
        self.current_speed = abs(msg.twist.twist.linear.x)

    def kys_callback(self, msg: Bool):
        if msg.data and not self.stopped:
            self.stopped = True
            self.get_logger().info("Emergency stop — ending episode")
            self._end_episode()
            self._reset_car()
        elif not msg.data and self.stopped:
            self.stopped = False
            self.get_logger().info("Safety released — new episode")

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _publish_stop(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def _end_episode(self):
        if self.prev_state is not None:
            reward = compute_reward(
                self.prev_raw_lidar, 0.0, self.prev_steering, done=True,
                prev_steering=self.prev_prev_steering)
            self.trainer.store(
                self.prev_state, self.prev_action, reward,
                self.prev_state, True)
            self.episode_reward += reward
            self.episode_steps += 1

        self.episode_count += 1
        self.get_logger().info(
            f"Episode {self.episode_count} | "
            f"reward={self.episode_reward:.2f} "
            f"steps={self.episode_steps} "
            f"total={self.step_count} "
            f"buffer={len(self.trainer.buffer)}")

        if self.episode_steps > self.best_episode_steps:
            self.best_episode_steps = self.episode_steps
            best_path = self.checkpoint_path.replace('.pth', '_best.pth')
            self.trainer.save(best_path)
            self.get_logger().info(
                f"NEW BEST (steps={self.episode_steps}, "
                f"reward={self.episode_reward:.2f})")

        self._log_episode()
        self.trainer.save(self.checkpoint_path)

        # reset episode state
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.prev_state = None
        self.prev_action = None

    def _reset_car(self):
        self._publish_stop()

        pose = PoseWithCovarianceStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.pose.position.x = self.reset_x
        pose.pose.pose.position.y = self.reset_y
        pose.pose.pose.orientation.z = math.sin(self.reset_yaw / 2.0)
        pose.pose.pose.orientation.w = math.cos(self.reset_yaw / 2.0)
        self.reset_pub.publish(pose)
        self.get_logger().info("Car reset to starting pose")

    def _init_log(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                csv.writer(f).writerow([
                    "episode", "reward", "steps", "total_steps", "buffer_size"])

    def _log_episode(self):
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                self.episode_count,
                round(self.episode_reward, 4),
                self.episode_steps,
                self.step_count,
                len(self.trainer.buffer)])


def main(args=None):
    rclpy.init(args=args)
    node = SACTrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down — saving checkpoint")
        node.trainer.save(node.checkpoint_path)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
