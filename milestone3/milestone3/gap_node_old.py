import math
import rclpy
import numpy as np
from milestone3.pid import PID
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from rclpy.parameter import Parameter

"""
Old Gap-Following Node.

Implements a reactive gap-following controller using LIDAR disparity detection.
The node identifies the largest navigable gap in front of the vehicle and steers
toward its center.
"""
class GapNodeOld(Node):
    """
    Reactive gap-following controller.

    Uses LIDAR range data to detect gaps via disparity extension,
    selects a target direction within the largest gap, and applies a PID
    controller to compute steering commands. An external safety signal can
    latch an emergency stop.
    """
    def __init__(self):
        """
        Initialize the gap-following node.
    
        Sets up ROS publishers, subscribers, parameters, and initializes the
        PID controller.
        """
        super().__init__('gap_node_old')

        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.listener_callback, 10)
        self.kys_sub = self.create_subscription(Bool, '/kys', self.kys_callback, 10)
        self.vel_sub = self.create_subscription(Odometry, '/odom', self.velocity_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        #Parameter Declarations & Value Assignment
        self.K_p = self.declare_parameter('K_p', 1.0).value
        self.K_i = self.declare_parameter('K_i', 0.0).value
        self.K_d = self.declare_parameter('K_d', 0.05).value

        self.K_p = self.get_parameter('K_p').get_parameter_value().double_value
        self.K_i = self.get_parameter('K_i').get_parameter_value().double_value
        self.K_d = self.get_parameter('K_d').get_parameter_value().double_value

        self.declare_parameter('target_distance', 1.0)
        self.declare_parameter('max_speed', 1.0)
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('K_speed', 1.0)
        self.declare_parameter('kys_latched', False)

        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.K_speed = self.get_parameter('K_speed').get_parameter_value().double_value

        self.pid = PID(self.K_p, self.K_i, self.K_d)

        self.kys = self.get_parameter('kys_latched').get_parameter_value().bool_value


    def listener_callback(self, msg) -> None:
        """
        Process incoming LIDAR scans and compute a driving command.
    
        Filters raw LIDAR ranges using disparity extension, identifies the
        largest navigable gap, selects a target steering angle, and applies
        PID control to generate steering and speed commands.
    
        Args:
            msg (LaserScan): Incoming LIDAR scan message.
        """
        
        if self.kys:
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.speed = 0.0
            drive_msg.drive.steering_angle = 0.0
            self.publisher_.publish(drive_msg)
            return
        
        
        current_time = self.get_clock().now().nanoseconds / 1e9 # clock in seconds (nano / 10^9)

        clipped_ranges = np.clip(msg.ranges, msg.range_min, 3.5)

        filtered_ranges = self.filter_ranges(clipped_ranges)

        target = self.get_target(filtered_ranges)

        center = len(filtered_ranges) // 2
        angle = (target - center) * msg.angle_increment
        if self.check_corners(filtered_ranges, angle, 0.2):
            angle = 0

        pid_angle = self.pid.pid_err(angle, current_time)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = pid_angle

        # 1.0 - changes how much the car slows on sharp turns, tune as necessary
        speed = self.max_speed - np.abs(pid_angle) * self.K_speed
        if speed < self.min_speed:
            speed = self.min_speed
        
        drive_msg.drive.speed = float(speed)

        self.publisher_.publish(drive_msg)

        # self.get_logger().info('%s' % angle)
        

    def filter_ranges(self, ranges) -> np.ndarray:
        """
        Increase obstacle size in LIDAR ranges for safer gap detection.
    
        Detects disparities and expands nearby
        obstacles to account for vehicle width and safety margin.
    
        Args:
            ranges (np.ndarray): Array of clipped LIDAR range measurements.
    
        Returns:
            np.ndarray: Modified range array with unsafe regions reduced.
        """
        safe_ranges = ranges.copy()

        disparities = []
        for i in range(len(safe_ranges) - 1):
            if np.abs(safe_ranges[i] - safe_ranges[i+1]) > 1:
                disparities.append(i)

        for i in disparities:
            ray1 = ranges[i]
            ray2 = ranges[i+1]

            if ray1 < ray2:
                near = ray1
                direction = 1
                start = i + 1
            else:
                near = ray2
                direction = -1
                start = i

            danger_zone = int(np.arctan2(0.5, near) * 180 * 4 / 3.14)

            #danger_zone = 125

            for j in range(danger_zone):
                k = start + direction * j
                if 0 <= k and k < len(safe_ranges):
                    if safe_ranges[k] > near:
                        safe_ranges[k] = near

            
            # self.get_logger().info('danger: %s' % danger_zone)

        return safe_ranges

    def get_target(self, ranges: np.ndarray) -> int:
        """
        Select the target ray as the center of the largest open gap
        in a forward-facing cone. This keeps the car centered and reduces wall-hugging.
        """
        n = len(ranges)
        center = n // 2

        # Look mostly forward (tune these if needed)
        left = n // 4
        right = 3 * n // 4
        cone = ranges[left:right]

        # Consider anything beyond this as "free space"
        # (Tune: bigger -> more willing to drive between closer walls)
        free_thresh = 1.2

        free = cone > free_thresh

        if not np.any(free):
            # If everything is tight, just go straight
            return center

        # Find contiguous True segments in `free`
        # We'll pick the longest gap; tie-break by being closer to straight ahead.
        best_start = best_end = None
        best_len = -1
        best_center_dist = float("inf")

        i = 0
        while i < len(free):
            if not free[i]:
                i += 1
                continue
            start = i
            while i < len(free) and free[i]:
                i += 1
            end = i - 1  # inclusive

            gap_len = end - start + 1
            gap_center = (start + end) // 2
            dist_to_center = abs((gap_center + left) - center)

            if (gap_len > best_len) or (gap_len == best_len and dist_to_center < best_center_dist):
                best_len = gap_len
                best_start, best_end = start, end
                best_center_dist = dist_to_center

        target_in_cone = (best_start + best_end) // 2
        target = target_in_cone + left
        return int(target)

    def check_corners(self, ranges, angle, min_clearance) -> bool:
        """
        Detect sharp dnagerous corners.
        Check if there is enough space on the turning side.
    
        Args:
            ranges (np.ndarray): Filtered LIDAR ranges.
            angle (float): Steering angle (radians).
            min_clearance (float): Minimum safe distance threshold.
    
        Returns:
            bool: True if a corner dead-end is detected, False otherwise.
        """
        if angle < 0:
            return np.all(ranges[900:1080] < min_clearance)
        else:
            return np.all(ranges[:180] < min_clearance)

    def kys_callback(self, msg) -> None:
        """
        Emergency stop callback. Latches the stop flag when the safety node asserts it.

        Args:
            msg (AckermannDriveStamped): stop command from safety node
        """
        if msg.data:
            self.kys = True
            self.set_parameters([Parameter('kys_latched', Parameter.Type.BOOL, True)])
            self.get_logger().info('kys latched')

    def velocity_callback(self, msg) -> None:
        """
        Store the current forward velocity of the vehicle.

        Args:
            msg (Odometry): Odometry message containing linear velocity.
        """
        self.last_vel = -(msg.twist.twist.linear.x)

def main(args=None) -> None:
    """
    Entry point for the wall-following node.
    """
    rclpy.init(args=args)
    node = GapNodeOld()
    rclpy.spin(gap_node_old)
    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()