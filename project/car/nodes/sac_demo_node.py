"""SAC demo node (physical car).

Pure inference — loads the best checkpoint, publishes drive commands,
and respects the safety node's /kys emergency stop. No training,
no weight updates, no checkpoints saved.

Launch:
    ros2 launch project sac_demo_py.py
"""

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from project.sac.model import SACActorNet

LIDAR_STEP = 6
MAX_RANGE = 10.0


class SACDemoNode(Node):

    def __init__(self):
        super().__init__("sac_demo_node")

        # ---- parameters ----
        self.declare_parameter("checkpoint_path", "")
        self.declare_parameter("scalers_path", "")
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("min_speed", 0.5)

        checkpoint_path = self._str("checkpoint_path")
        scalers_path = self._str("scalers_path")
        self.max_speed = self._dbl("max_speed")
        self.min_speed = self._dbl("min_speed")

        # ---- load scalers ----
        scalers = np.load(scalers_path)
        self.lidar_scale = scalers["lidar_scale"].astype(np.float32)
        self.lidar_min = scalers["lidar_min"].astype(np.float32)
        self.action_scale = scalers["action_scale"].astype(np.float32)
        self.action_min = scalers["action_min"].astype(np.float32)
        self.num_lidar = len(self.lidar_scale)

        # ---- load actor (weights only, no critics/optimizer needed) ----
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor = SACActorNet(self.num_lidar).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device,
                          weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()

        self.get_logger().info(
            f"SAC DEMO ready | {self.num_lidar} lidar features | "
            f"device={self.device} | checkpoint={checkpoint_path}")

        # ---- state ----
        self.stopped = False
        self.step_count = 0
        self.prev_steering = 0.0
        self.prev_speed_cmd = 0.0

        # ---- ROS interface ----
        self.scan_sub = self.create_subscription(
            LaserScan, "/scan", self.scan_callback, 10)
        self.kys_sub = self.create_subscription(
            Bool, "/kys", self.kys_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, "/drive", 10)

    def _str(self, n):
        return self.get_parameter(n).get_parameter_value().string_value

    def _dbl(self, n):
        return self.get_parameter(n).get_parameter_value().double_value

    # ------------------------------------------------------------------ #

    def scan_callback(self, msg: LaserScan):
        if self.stopped:
            self._publish_stop()
            return

        raw = np.array(msg.ranges, dtype=np.float32)

        # preprocess
        ds = raw[::LIDAR_STEP]
        ds = np.where(np.isfinite(ds), ds, MAX_RANGE)
        ds = np.clip(ds, 0.0, MAX_RANGE)
        if len(ds) > self.num_lidar:
            ds = ds[:self.num_lidar]
        elif len(ds) < self.num_lidar:
            ds = np.pad(ds, (0, self.num_lidar - len(ds)),
                        constant_values=MAX_RANGE)
        state = ds * self.lidar_scale + self.lidar_min

        # inference
        with torch.no_grad():
            state_t = torch.from_numpy(state.reshape(1, -1)).to(self.device)
            action = self.actor.get_action(state_t, deterministic=True)
            action = action.cpu().numpy()[0]

        # denormalise
        steering = float((action[0] - self.action_min[0]) / self.action_scale[0])
        speed = float((action[1] - self.action_min[1]) / self.action_scale[1])
        steering, speed = self._postprocess_action(steering, speed, ds)

        # publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

        self.prev_steering = steering
        self.prev_speed_cmd = speed

        self.step_count += 1
        if self.step_count <= 10:
            self.get_logger().info(
                f"[DRIVE #{self.step_count}] steer={steering:.4f} "
                f"speed={speed:.4f}")

    def kys_callback(self, msg: Bool):
        if msg.data and not self.stopped:
            self.stopped = True
            self.get_logger().info("Emergency stop latched")
        elif not msg.data and self.stopped:
            self.stopped = False
            self.get_logger().info("Emergency stop released")

    def _publish_stop(self):
        msg = AckermannDriveStamped()
        msg.drive.speed = 0.0
        msg.drive.steering_angle = 0.0
        self.drive_pub.publish(msg)

    def _postprocess_action(self, steering, speed, raw_lidar):
        if not np.isfinite(steering):
            steering = 0.0
        if not np.isfinite(speed):
            speed = self.min_speed

        speed = max(0.0, min(speed, self.max_speed))
        if 0.0 < speed < self.min_speed:
            speed = self.min_speed

        return steering, speed


def main(args=None):
    rclpy.init(args=args)
    node = SACDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
