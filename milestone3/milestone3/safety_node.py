import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
import numpy as np

class SafetyNode(Node):
    def __init__(self):
        super().__init__('safety_node')
        
        # Subscribers and Publishers
        self.cam_sub = self.create_subscription(Image, '/camera/depth/image_rect_raw', self.cam_callback, 10)
        self.velocity_subscriber = self.create_subscription(Odometry, '/odom', self.velocity_callback, 10)
        self.speed_publisher = self.create_publisher(AckermannDriveStamped, '/speed', 10)
        self.kys_publisher = self.create_publisher(Bool, '/kys', 10)

        self.kys = False
        self.last_vx = None
        
        
        # Parameters
        self.distance_threshold = self.declare_parameter('distance_threshold', 0.5).value  # meters
        self.ttc_pb1 = self.declare_parameter('ttc_pb1', 1.5).value
        self.ttc_pb2 = self.declare_parameter('ttc_pb2', 0.9).value
        self.ttc_fb = self.declare_parameter('ttc_fb', 0.7).value
        self.pb1_speed_mult = self.declare_parameter('pb1_speed_mult', 0.3).value
        self.pb2_speed_mult = self.declare_parameter('pb2_speed_mult', 0.2).value

    def velocity_callback(self, msg: Odometry) -> None:
        self.last_vx = abs(msg.twist.twist.linear.x)

    def cam_callback(self, msg) -> None:
        if self.kys:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.speed_publisher.publish(drive_msg)
            return
        elif self.last_vx is None:
            return
            
        depth = self.depth_image_to_numpy(msg)
        if depth is None:
            return
        h, w = depth.shape

        # Front ROI (forward cone approximation)
        y0 = int(0.5 * h)
        y1 = int(0.75 * h)
        x0 = int(0.3 * w)
        x1 = int(0.7 * w)
        front_roi = depth[y0:y1, x0:x1]

        # Chunking = virtual forward rays (like LiDAR beams)
        num_chunks = 16
        roi_width = front_roi.shape[1]
        chunk_width = roi_width // num_chunks

        min_ttc = math.inf
        min_distance = math.inf
        vx = self.last_vx  # last known forward velocity
        
        for i in range(num_chunks):
            cx0 = i * chunk_width
            cx1 = (i + 1) * chunk_width if i < num_chunks - 1 else roi_width
            chunk = front_roi[:, cx0:cx1]

            # Remove invalid depth values 
            valid = chunk[chunk > 0]
            if valid.size == 0:
                continue

            # Median depth
            median_depth_mm = np.median(valid)
            d = median_depth_mm / 1000.0

            # Track closest obstacle
            if d < min_distance:
                min_distance = d

            # Compute TTC 
            if vx < 0.1:
                ttc = math.inf
            else:
                ttc = d / vx

            if ttc < min_ttc:
                min_ttc = ttc

        # if min_ttc == math.inf:
        #     drive_msg = AckermannDriveStamped()
        #     drive_msg.drive.speed = 1.0
        #     self.speed_publisher.publish(drive_msg)
        #     return

        # Determine braking stage 
        stage = "NONE"

        if min_distance < self.distance_threshold:
            self.get_logger().info(f"Distance Threshold is: {self.distance_threshold:.2f} -- DEBUG")
            self.get_logger().info(f"Distance Threshold Triggered - KILLING SELF")
            stage = "FB"
        elif min_ttc < self.ttc_fb:
            self.get_logger().info(f"FB TTC Threshold Triggered - KILLING SELF")
            stage = "FB"
        elif min_ttc < self.ttc_pb1:
            stage = "PB1"
        elif min_ttc < self.ttc_pb2:
            stage = "PB2"
    
        if stage == "NONE":
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.6
            self.speed_publisher.publish(drive_msg)

        elif stage == "PB1":
            self.drive_speed = max(0.6, self.pb1_speed_mult * vx)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = self.drive_speed
            self.speed_publisher.publish(drive_msg)
            self.get_logger().info(f"PARTIAL BRAKE 1 - Distance: {min_distance}m, TTC: {min_ttc:.2f}s")

        elif stage == "PB2":
            self.drive_speed = max(0.6, self.pb2_speed_mult * vx)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = self.drive_speed
            self.speed_publisher.publish(drive_msg)
            self.get_logger().info(f"PARTIAL BRAKE 2 - Distance: {min_distance}m, TTC: {min_ttc:.2f}s")

        elif stage == "FB":
            self.kys = True
            kys_msg = Bool()
            kys_msg.data = True
            self.kys_publisher.publish(kys_msg)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            self.speed_publisher.publish(drive_msg)
            self.get_logger().info(f"FULL BRAKE - Distance: {min_distance}m, TTC: {min_ttc:.2f}s")

    def depth_image_to_numpy(self, msg: Image) -> np.ndarray:
        if msg.encoding == '16UC1':
            dtype = np.uint16
        elif msg.encoding == '32FC1':
            dtype = np.float32
        else:
            self.get_logger().warn(f'Unsupported depth encoding: {msg.encoding}')
            return None

        depth = np.frombuffer(msg.data, dtype=dtype)
        if msg.is_bigendian:
            depth = depth.byteswap()

        row_width = msg.step // np.dtype(dtype).itemsize
        depth = depth.reshape((msg.height, row_width))
        depth = depth[:, :msg.width]

        if dtype == np.float32:
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0

        return depth


def main(args=None) -> None:
    rclpy.init(args=args)
    safety_node = SafetyNode()
    rclpy.spin(safety_node)
    safety_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
