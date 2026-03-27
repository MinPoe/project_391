import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from stable_baselines3 import SAC

NUM_LIDAR_RAYS = 181
DEFAULT_MODEL_PATH = os.path.expanduser('~/rl_models/sac_final')


class RLNode(Node):
    """
    Deployment node for the trained SAC policy.
    Mirrors the structure of gap_node.py — subscribes to /scan,
    publishes to /drive, respects the /kys safety signal.
    """
    def __init__(self):
        super().__init__('rl_node')

        # Load trained model
        model_path = self.declare_parameter(
            'model_path', DEFAULT_MODEL_PATH).value
        self.get_logger().info(f'Loading model from {model_path}...')
        self.model = SAC.load(model_path, device='cpu')
        self.get_logger().info('Model loaded successfully.')

        # State
        self.kys = False
        self.prev_steering = 0.0
        self.max_range = 10.0  # will be updated from scan

        # Subscribers & publishers — identical topics to gap_node
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        self.kys_sub = self.create_subscription(
            Bool, '/kys', self.kys_callback, 10)
        self.odom_sub = self.create_subscription(
            Odometry, '/ego_racecar/odom', self.odom_callback, 10)
        self.drive_pub = self.create_publisher(
            AckermannDriveStamped, '/drive', 10)

        self.get_logger().info('RL Node ready.')

    def scan_callback(self, msg: LaserScan):
        """Process LIDAR scan and publish drive command."""
        if self.kys:
            self._publish_drive(0.0, 0.0)
            return

        # Preprocess scan — must match rl_env.py exactly
        self.max_range = msg.range_max
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.where(np.isfinite(ranges), ranges, msg.range_max)
        ranges = np.clip(ranges, msg.range_min, msg.range_max)
        ranges = ranges[:1080]
        indices = np.linspace(0, 1079, 181, dtype=int)
        ranges = ranges[indices]
        ranges = ranges / msg.range_max  # normalize to [0, 1]

        # Run inference — this is fast (~1ms)
        action, _ = self.model.predict(ranges, deterministic=True)
        steering = float(action[0])
        speed = float(action[1])

        # Smooth steering to prevent jerks (simple low-pass filter)
        alpha = 0.7  # tune: higher = smoother but slower response
        steering = alpha * self.prev_steering + (1 - alpha) * steering
        self.prev_steering = steering

        self._publish_drive(steering, speed)

    def kys_callback(self, msg: Bool):
        """Mirror of gap_node kys_callback — respect safety node."""
        if msg.data:
            self.kys = True
            self.get_logger().warn('KYS received — stopping RL node.')

    def odom_callback(self, msg: Odometry):
        """Store velocity (available if needed for future logging)."""
        self.last_velocity = abs(msg.twist.twist.linear.x)

    def _publish_drive(self, steering: float, speed: float):
        msg = AckermannDriveStamped()
        msg.drive.steering_angle = steering
        msg.drive.speed = speed
        self.drive_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = RLNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()