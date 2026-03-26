"""ROS2 inference node for the Behavioural Cloning model.

Subscribes to /scan (LaserScan), runs the BC model, and publishes
AckermannDriveStamped to /drive.

Usage (standalone):
    ros2 run milestone3 bc_inference_node \
        --ros-args -p model_path:=bc/bc_model_sim.pth \
                   -p scalers_path:=processed/processed_simulator/scalers.npz \
                   -p max_speed:=1.0
"""

import numpy as np
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from project.bc.model import BCNet

# Must match preprocessing constants
LIDAR_STEP = 6
MAX_RANGE = 10.0


class BCInferenceNode(Node):

    def __init__(self):
        super().__init__("bc_inference_node")

        # Parameters
        self.declare_parameter("model_path", "bc/bc_model_sim.pth") # Change this path to either sim or physical
        self.declare_parameter("scalers_path", "processed/processed_simulator/scalers.npz")
        self.declare_parameter("max_speed", 1.0)
        self.declare_parameter("min_speed", 0.5)
        self.declare_parameter("safety_distance", 0.3)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        scalers_path = self.get_parameter("scalers_path").get_parameter_value().string_value
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value
        self.min_speed = self.get_parameter("min_speed").get_parameter_value().double_value
        self.safety_distance = self.get_parameter("safety_distance").get_parameter_value().double_value

        # Load scaler parameters from .npz (portable across numpy versions)
        scalers = np.load(scalers_path)
        self.lidar_scale = scalers["lidar_scale"].astype(np.float32)
        self.lidar_min = scalers["lidar_min"].astype(np.float32)
        self.action_scale = scalers["action_scale"].astype(np.float32)
        self.action_min = scalers["action_min"].astype(np.float32)

        self.num_lidar = len(self.lidar_scale)
        self.get_logger().info(f"LiDAR features: {self.num_lidar}")

        # Load model (CPU is fine for this small network)
        self.device = torch.device("cpu")
        self.model = BCNet(num_lidar_rays=self.num_lidar).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.get_logger().info(f"Loaded model from {model_path} on {self.device}")

        # Emergency stop flag
        self.stopped = False

        # ROS interface
        self.scan_sub = self.create_subscription(LaserScan, "/scan", self.scan_callback, 10)
        self.kys_sub = self.create_subscription(Bool, "/kys", self.kys_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, "/drive", 10)

    def scan_callback(self, msg: LaserScan):
        if self.stopped:
            self._publish_stop()
            return

        ranges = np.array(msg.ranges, dtype=np.float32)

        # LiDAR-based emergency stop: check forward cone for obstacles
        n = len(ranges)
        forward_cone = ranges[n // 4 : 3 * n // 4]
        forward_cone = forward_cone[np.isfinite(forward_cone)]
        if len(forward_cone) > 0 and np.min(forward_cone) < self.safety_distance:
            self.get_logger().warn(
                f"EMERGENCY STOP: obstacle at {np.min(forward_cone):.2f}m"
            )
            self.stopped = True
            self._publish_stop()
            return

        # Downsample raw scan (keep every LIDAR_STEP-th ray)
        downsampled = ranges[::LIDAR_STEP]

        # Clamp infinities / NaN to MAX_RANGE, clip to [0, MAX_RANGE]
        downsampled = np.where(np.isfinite(downsampled), downsampled, MAX_RANGE)
        downsampled = np.clip(downsampled, 0.0, MAX_RANGE)

        # Ensure the length matches what the scaler expects
        if len(downsampled) != self.num_lidar:
            if len(downsampled) > self.num_lidar:
                downsampled = downsampled[: self.num_lidar]
            else:
                downsampled = np.pad(
                    downsampled, (0, self.num_lidar - len(downsampled)),
                    constant_values=MAX_RANGE,
                )

        # Normalize: X_scaled = X * scale + min  (MinMaxScaler formula)
        lidar_norm = downsampled * self.lidar_scale + self.lidar_min

        # Inference
        with torch.no_grad():
            x = torch.from_numpy(lidar_norm.reshape(1, -1)).to(self.device)
            pred = self.model(x).cpu().numpy()[0]  # shape (2,)

        # Denormalize: X = (X_scaled - min) / scale
        steering_angle = float((pred[0] - self.action_min[0]) / self.action_scale[0])
        speed = float((pred[1] - self.action_min[1]) / self.action_scale[1])

        # Clamp speed
        speed = max(self.min_speed, min(speed, self.max_speed))

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def kys_callback(self, msg: Bool):
        if msg.data:
            self.stopped = True
            self.get_logger().info("Emergency stop latched")
        else:
            self.stopped = False

    def _publish_stop(self):
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        self.drive_pub.publish(drive_msg)


def main(args=None):
    rclpy.init(args=args)
    node = BCInferenceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
