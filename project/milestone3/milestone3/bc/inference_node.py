"""ROS2 inference node for the Behavioural Cloning model.

Subscribes to /scan (LaserScan), runs the BC model, and publishes
AckermannDriveStamped to /drive.

Usage (standalone):
    ros2 run milestone3 bc_inference_node \
        --ros-args -p model_path:=bc/bc_model.pth \
                   -p scaler_lidar_path:=processed/processed_simulator/scaler_lidar.pkl \
                   -p scaler_action_path:=processed/processed_simulator/scaler_action.pkl \
                   -p max_speed:=1.0
"""

import numpy as np
import torch
import joblib

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool

from milestone3.bc.model import BCNet

# Must match preprocessing constants
LIDAR_STEP = 6
MAX_RANGE = 10.0


class BCInferenceNode(Node):

    def __init__(self):
        super().__init__("bc_inference_node")

        # Parameters
        self.declare_parameter("model_path", "bc/bc_model.pth")
        self.declare_parameter("scaler_lidar_path", "processed/processed_simulator/scaler_lidar.pkl")
        self.declare_parameter("scaler_action_path", "processed/processed_simulator/scaler_action.pkl")
        self.declare_parameter("max_speed", 1.0)

        model_path = self.get_parameter("model_path").get_parameter_value().string_value
        scaler_lidar_path = self.get_parameter("scaler_lidar_path").get_parameter_value().string_value
        scaler_action_path = self.get_parameter("scaler_action_path").get_parameter_value().string_value
        self.max_speed = self.get_parameter("max_speed").get_parameter_value().double_value

        # Load scalers (MinMaxScaler from preprocessing)
        self.scaler_lidar = joblib.load(scaler_lidar_path)
        self.scaler_action = joblib.load(scaler_action_path)

        # Determine number of LiDAR features from scaler
        self.num_lidar = self.scaler_lidar.n_features_in_
        self.get_logger().info(f"LiDAR features: {self.num_lidar}")

        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

        # Downsample raw scan (keep every LIDAR_STEP-th ray)
        ranges = np.array(msg.ranges, dtype=np.float32)
        downsampled = ranges[::LIDAR_STEP]

        # Clamp infinities / NaN to MAX_RANGE, clip to [0, MAX_RANGE]
        downsampled = np.where(np.isfinite(downsampled), downsampled, MAX_RANGE)
        downsampled = np.clip(downsampled, 0.0, MAX_RANGE)

        # Ensure the length matches what the scaler expects
        if len(downsampled) != self.num_lidar:
            # Truncate or pad to match
            if len(downsampled) > self.num_lidar:
                downsampled = downsampled[: self.num_lidar]
            else:
                downsampled = np.pad(
                    downsampled, (0, self.num_lidar - len(downsampled)),
                    constant_values=MAX_RANGE,
                )

        # Normalize using the same scaler used during training
        lidar_norm = self.scaler_lidar.transform(downsampled.reshape(1, -1))

        # Inference
        with torch.no_grad():
            x = torch.from_numpy(lidar_norm.astype(np.float32)).to(self.device)
            pred = self.model(x).cpu().numpy()  # shape (1, 2)

        # Denormalize action
        action = self.scaler_action.inverse_transform(pred)[0]
        steering_angle = float(action[0])
        speed = float(action[1])

        # Clamp speed
        speed = max(0.0, min(speed, self.max_speed))

        # Publish
        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = steering_angle
        drive_msg.drive.speed = speed
        self.drive_pub.publish(drive_msg)

    def kys_callback(self, msg: Bool):
        if msg.data:
            self.stopped = True
            self.get_logger().info("Emergency stop latched")

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
