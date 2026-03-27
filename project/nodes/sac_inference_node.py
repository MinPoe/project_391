"""ROS2 node for SAC online training and inference.

Drives the car using the SAC actor, collects transitions, computes rewards,
and trains the actor/critics from the replay buffer in real time.

Standalone (sim training):
    ros2 run project sac_inference_node --ros-args \
        -p bc_weights_path:=bc/bc_model_sim.pth \
        -p scalers_path:=processed/processed_simulator/scalers.npz \
        -p training:=true -p max_speed:=2.0

Inference only (physical car):
    ros2 run project sac_inference_node --ros-args \
        -p checkpoint_path:=sac/sac_checkpoint.pth \
        -p scalers_path:=processed/processed_physical/scalers.npz \
        -p training:=false -p deterministic:=true -p max_speed:=1.0
"""

import math
import os
import csv
import time
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

# Must match preprocessing constants
LIDAR_STEP = 6
MAX_RANGE = 10.0


class SACInferenceNode(Node):

    def __init__(self):
        super().__init__("sac_inference_node")

        # ---- declare parameters ----
        self.declare_parameter("bc_weights_path", "bc/bc_model_sim.pth")
        self.declare_parameter("scalers_path", "processed/processed_simulator/scalers.npz")
        self.declare_parameter("checkpoint_path", "sac/sac_checkpoint.pth")
        self.declare_parameter("log_path", "sac/training_log.csv")
        self.declare_parameter("max_speed", 2.0)
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("training", True)
        self.declare_parameter("deterministic", False)
        # SAC hyper-parameters
        self.declare_parameter("lr", 3e-4)
        self.declare_parameter("gamma", 0.99)
        self.declare_parameter("tau", 0.005)
        self.declare_parameter("buffer_size", 100000)
        self.declare_parameter("batch_size", 256)
        self.declare_parameter("update_every", 4)
        self.declare_parameter("warmup_steps", 0)
        self.declare_parameter("save_every", 5000)
        # Collision detection + reset
        self.declare_parameter("collision_threshold", 0.3)
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
        self.training = self._bool("training")
        self.deterministic = self._bool("deterministic")
        lr = self._dbl("lr")
        gamma = self._dbl("gamma")
        tau = self._dbl("tau")
        buffer_size = self._int("buffer_size")
        batch_size = self._int("batch_size")
        self.update_every = self._int("update_every")
        self.warmup_steps = self._int("warmup_steps")
        self.save_every = self._int("save_every")
        self.collision_threshold = self._dbl("collision_threshold")
        self.reset_x = self._dbl("reset_x")
        self.reset_y = self._dbl("reset_y")
        self.reset_yaw = self._dbl("reset_yaw")

        # ---- load scalers (same ones BC uses) ----
        scalers = np.load(scalers_path)
        self.lidar_scale = scalers["lidar_scale"].astype(np.float32)
        self.lidar_min = scalers["lidar_min"].astype(np.float32)
        self.action_scale = scalers["action_scale"].astype(np.float32)
        self.action_min = scalers["action_min"].astype(np.float32)
        self.num_lidar = len(self.lidar_scale)
        self.get_logger().info(f"LiDAR features: {self.num_lidar}")

        # ---- build / load networks ----
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if os.path.isfile(self.checkpoint_path):
            self.get_logger().info(f"Resuming from checkpoint: {self.checkpoint_path}")
            actor = SACActorNet(self.num_lidar)
            critic1 = SACCriticNet(self.num_lidar)
            critic2 = SACCriticNet(self.num_lidar)
            self.trainer = SACTrainer(
                actor, critic1, critic2,
                state_dim=self.num_lidar, lr_actor=lr, lr_critic=lr, lr_alpha=lr,
                gamma=gamma, tau=tau, buffer_size=buffer_size,
                batch_size=batch_size, device=device,
            )
            self.trainer.load(self.checkpoint_path)
        else:
            if os.path.isfile(bc_weights_path):
                self.get_logger().info(f"Initialising actor from BC: {bc_weights_path}")
                actor = SACActorNet.from_bc(bc_weights_path, self.num_lidar, device=device)
            else:
                self.get_logger().warn("No BC weights found -- random actor init")
                actor = SACActorNet(self.num_lidar)
            critic1 = SACCriticNet(self.num_lidar)
            critic2 = SACCriticNet(self.num_lidar)
            self.trainer = SACTrainer(
                actor, critic1, critic2,
                state_dim=self.num_lidar, lr_actor=lr, lr_critic=lr, lr_alpha=lr,
                gamma=gamma, tau=tau, buffer_size=buffer_size,
                batch_size=batch_size, device=device,
            )

        self.get_logger().info(
            f"SAC ready | training={self.training} deterministic={self.deterministic} "
            f"device={device}"
        )

        # ---- state tracking ----
        self.prev_state = None
        self.prev_action = None
        self.prev_raw_lidar = None
        self.prev_steering = 0.0
        self.current_speed = 0.0
        self.step_count = 0
        self.episode_reward = 0.0
        self.episode_steps = 0
        self.episode_count = 0
        self.collision_detected = False
        self.episode_start_time = time.time()

        # ---- training log (CSV) ----
        if self.training:
            self._init_log()

        # ---- ROS interface ----
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, "/ego_racecar/odom", self.odom_callback, 10
        )
        self.collision_sub = self.create_subscription(
            Bool, "/collision", self.collision_callback, 10
        )
        self.kys_sub = self.create_subscription(
            Bool, "/kys", self.kys_callback, 10
        )
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/drive", 10
        )
        self.reset_pub = self.create_publisher(
            PoseWithCovarianceStamped, "/initialpose", 10
        )

    # ------------------------------------------------------------------ #
    #  Parameter helpers                                                  #
    # ------------------------------------------------------------------ #

    def _str(self, name):
        return self.get_parameter(name).get_parameter_value().string_value

    def _dbl(self, name):
        return self.get_parameter(name).get_parameter_value().double_value

    def _int(self, name):
        return self.get_parameter(name).get_parameter_value().integer_value

    def _bool(self, name):
        return self.get_parameter(name).get_parameter_value().bool_value

    # ------------------------------------------------------------------ #
    #  Scan callback -- main driving + training loop                      #
    # ------------------------------------------------------------------ #

    def scan_callback(self, msg: LaserScan):
        raw_ranges = np.array(msg.ranges, dtype=np.float32)

        # --- preprocess LiDAR (same as BC) ---
        downsampled = raw_ranges[::LIDAR_STEP]
        downsampled = np.where(np.isfinite(downsampled), downsampled, MAX_RANGE)
        downsampled = np.clip(downsampled, 0.0, MAX_RANGE)
        if len(downsampled) > self.num_lidar:
            downsampled = downsampled[: self.num_lidar]
        elif len(downsampled) < self.num_lidar:
            downsampled = np.pad(
                downsampled, (0, self.num_lidar - len(downsampled)),
                constant_values=MAX_RANGE,
            )
        raw_lidar = downsampled.copy()

        # normalise
        state = downsampled * self.lidar_scale + self.lidar_min

        # --- collision detection (sim /collision topic OR LiDAR fallback) ---
        n = len(raw_ranges)
        forward = raw_ranges[n // 4 : 3 * n // 4]
        forward = forward[np.isfinite(forward)]
        lidar_crash = (
            len(forward) > 0
            and np.min(forward) < self.collision_threshold
        )
        crashed = self.collision_detected or lidar_crash

        if crashed:
            if self.training:
                self._end_episode(done=True)
                self._log_episode()
            self.episode_count += 1
            self.get_logger().info(
                f"CRASH | Episode {self.episode_count} | "
                f"reward={self.episode_reward:.2f} "
                f"steps={self.episode_steps} "
                f"total={self.step_count} "
                f"buffer={len(self.trainer.buffer)}"
            )
            self.episode_reward = 0.0
            self.episode_steps = 0
            self.prev_state = None
            self.prev_action = None
            self._reset_car()
            return

        # --- store previous transition ---
        if self.training and self.prev_state is not None:
            reward = compute_reward(
                self.prev_raw_lidar, self.current_speed,
                self.prev_steering, done=False,
            )
            self.trainer.store(
                self.prev_state, self.prev_action, reward, state, False
            )
            self.episode_reward += reward
            self.episode_steps += 1

        # --- select action (normalised [0, 1], same space as BC) ---
        if self.training and self.step_count < self.warmup_steps:
            action = np.random.uniform(0.0, 1.0, size=2).astype(np.float32)
        else:
            state_t = torch.from_numpy(state.reshape(1, -1)).to(
                self.trainer.device
            )
            action = (
                self.trainer.actor.get_action(state_t, self.deterministic)
                .cpu()
                .numpy()[0]
            )

        # --- denormalise with BC scalers:  X = (X_scaled - min) / scale ---
        steering = float((action[0] - self.action_min[0]) / self.action_scale[0])
        speed = float((action[1] - self.action_min[1]) / self.action_scale[1])
        speed = max(self.min_speed, min(speed, self.max_speed))

        # --- publish ---
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        # --- bookkeeping ---
        self.prev_state = state
        self.prev_action = action
        self.prev_raw_lidar = raw_lidar
        self.prev_steering = steering
        self.step_count += 1

        if self.step_count <= 10:
            self.get_logger().info(
                f"[DRIVE #{self.step_count}] steer={steering:.4f} speed={speed:.4f} "
                f"action=[{action[0]:.3f},{action[1]:.3f}]"
            )

        # --- SAC gradient step ---
        if self.training and self.step_count % self.update_every == 0:
            metrics = self.trainer.update()
            if metrics and self.step_count % 200 == 0:
                self.get_logger().info(
                    f"[step {self.step_count}] "
                    f"c1={metrics['critic1_loss']:.4f} "
                    f"c2={metrics['critic2_loss']:.4f} "
                    f"actor={metrics['actor_loss']:.4f} "
                    f"alpha={metrics['alpha']:.4f}"
                )

        # --- checkpoint ---
        if self.training and self.step_count % self.save_every == 0:
            self.trainer.save(self.checkpoint_path)
            self.get_logger().info(
                f"Checkpoint saved (step {self.step_count}, "
                f"buffer {len(self.trainer.buffer)})"
            )

    # ------------------------------------------------------------------ #
    #  Other callbacks                                                    #
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg: Odometry):
        self.current_speed = abs(msg.twist.twist.linear.x)

    def collision_callback(self, msg: Bool):
        if msg.data and time.time() - self.episode_start_time > 0.5:
            self.collision_detected = True

    def kys_callback(self, msg: Bool):
        """Safety node emergency stop (for physical car)."""
        if msg.data:
            self.collision_detected = True

    # ------------------------------------------------------------------ #
    #  Helpers                                                            #
    # ------------------------------------------------------------------ #

    def _end_episode(self, done: bool):
        if self.training and self.prev_state is not None:
            reward = compute_reward(
                self.prev_raw_lidar, 0.0, self.prev_steering, done=True,
            )
            self.trainer.store(
                self.prev_state, self.prev_action, reward,
                self.prev_state, True,
            )
            self.episode_reward += reward
            self.episode_steps += 1
        if self.training:
            self.trainer.save(self.checkpoint_path)

    def _reset_car(self):
        """Stop car, teleport to start, wait for sim to process."""
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)
        time.sleep(0.2)

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.pose.pose.position.x = self.reset_x
        pose_msg.pose.pose.position.y = self.reset_y
        pose_msg.pose.pose.orientation.z = math.sin(self.reset_yaw / 2.0)
        pose_msg.pose.pose.orientation.w = math.cos(self.reset_yaw / 2.0)
        self.reset_pub.publish(pose_msg)

        time.sleep(0.5)
        self.collision_detected = False
        self.episode_start_time = time.time()

    # ---- CSV training log ----

    def _init_log(self):
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        if not os.path.isfile(self.log_path):
            with open(self.log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode", "reward", "steps", "total_steps", "buffer_size",
                ])

    def _log_episode(self):
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count,
                round(self.episode_reward, 4),
                self.episode_steps,
                self.step_count,
                len(self.trainer.buffer),
            ])


def main(args=None):
    rclpy.init(args=args)
    node = SACInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down -- saving final checkpoint")
        if node.training:
            node.trainer.save(node.checkpoint_path)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
