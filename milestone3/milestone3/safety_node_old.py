import math
from typing import List
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

"""
    Old Safety Node: 
        Implements Automatic Emergency Braking (AEB) by computing Time-To-Collision (TTC)
        using LIDAR range data and vehicle velocity. The node supports multi-stage braking
        and can latch an emergency stop state.
"""
class SafetyNodeOld(Node):
    """
    Automatic Emergency Braking (AEB) node.

    Subscribes to LIDAR and odometry, computes TTC and minimum distance in a front cone,
    and publishes a stop command when safety thresholds are violated.
    """
    def __init__(self) -> None:
        """
        Initialize the safety node, subscribers, publisher, and parameters.
        """
        super().__init__('safety_node_old')

        # cached data from latest messages
        self.last_ranges = None
        self.last_range_min = None
        self.last_range_max = None
        self.last_angle_min = None
        self.last_angle_increment = None
        self.last_vx = None

        # ROS interfaces
        self.lidar_subscriber = self.create_subscription(LaserScan, '/scan', self.lidar_callback, 10)
        self.velocity_subscriber = self.create_subscription(Odometry, '/odom', self.velocity_callback, 10)
        self.drive_publisher = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.kys_publisher = self.create_publisher(Bool, '/kys', 10)
        self.add_on_set_parameters_callback(self.on_param_change)

        # Parameters 
        self.front_half_angle = self.declare_parameter('front_half_angle', 0.30).value  # radians front cone half
        self.distance_threshold = self.declare_parameter('distance_threshold', 0.01).value  # meters: brake when too close regardless of TTC
        self.kys_latched = self.declare_parameter('kys_latched', False).value
        self.ttc_pb1 = self.declare_parameter('ttc_pb1', 1.0).value # seconds for first stage of progressive braking to engage
        self.ttc_pb2 = self.declare_parameter('ttc_pb2', 0.7).value # seconds for second stage of progressive braking to engage
        self.ttc_fb  = self.declare_parameter('ttc_fb', 0.5).value # seconds for full braking to engage
        self.pb1_speed_mult = self.declare_parameter('pb1_speed_mult', 0.3).value # speed multiplier for first stage of progressive braking
        self.pb2_speed_mult = self.declare_parameter('pb2_speed_mult', 0.2).value # speed multiplier for second stage of progressive braking
      
        self.timer = self.create_timer(0.02, self.timer_callback)


    def lidar_callback(self, msg: LaserScan) -> None:
        """
        Cache the latest LIDAR scan data.

        Args:
            msg (LaserScan): LIDAR scan message
        """
        self.last_ranges = msg.ranges
        self.last_range_min = msg.range_min
        self.last_range_max = msg.range_max
        self.last_angle_min = msg.angle_min
        self.last_angle_increment = msg.angle_increment


    def velocity_callback(self, msg: Odometry) -> None:
        """
        Cache the latest linear velocity.

        Args:
            msg (Odometry): Odometry data
        """
        self.last_vx = -(msg.twist.twist.linear.x)
    

    def timer_callback(self) -> None:
        """
        Computes the minimum TTC and distance to obstacles in a forward direction cone.
        Based on the evaluated safety stage (NONE, PB1, PB2, FB), the node publishes
        braking commands or latches a full emergency stop.
        """
        if self.kys_latched:
            kys_msg = Bool()
            kys_msg.data = True
            self.kys_publisher.publish(kys_msg)
            drive_msg = AckermannDriveStamped()
            drive_msg.drive.steering_angle = 0.0
            drive_msg.drive.speed = 0.0
            self.drive_publisher.publish(drive_msg)
            return

        # guard for missing data
        if (self.last_ranges          is None or
            self.last_range_min       is None or
            self.last_range_max       is None or
            self.last_angle_min       is None or
            self.last_angle_increment is None or
            self.last_vx              is None):
            return

        min_ttc = math.inf
        min_distance = math.inf

        # compute min TTC in a cone
        for i, d in enumerate(self.last_ranges):
            # range filtering
            if d < self.last_range_min or d > self.last_range_max:
                continue

            theta = self.last_angle_min + i * self.last_angle_increment
            if abs(theta) > self.front_half_angle:
                continue
            # track closest obstacle in the front cone
            if d < min_distance:
                min_distance = d
            # range-rate derivative (closing speed along the ray)
            r_dot = -self.last_vx * math.cos(theta)
            if r_dot >= 0.0:
                continue
            ttc = d / (-r_dot)
            if ttc < min_ttc:
                min_ttc = ttc
                
        # Find stage of progressive braking to run
        stage = "NONE"
        # self.get_logger().info(f"min_ttc: {min_ttc}, min_distance: {min_distance}")
        if min_ttc < self.ttc_fb or min_distance < self.distance_threshold:
            stage = "FB"
        elif min_ttc < self.ttc_pb2: 
            stage = "PB2"
        elif min_ttc < self.ttc_pb1:
            stage = "PB1"

        drive_msg = AckermannDriveStamped()
        
        if stage == "NONE": # Don't do anything if AEB shouldn't activate
            return
        elif stage == "PB1": # Publish modified speed to drive
            drive_msg.drive.speed = self.pb1_speed_mult * self.last_vx
            self.drive_publisher.publish(drive_msg)
        elif stage == "PB2": # Publish modified speed to drive
            drive_msg.drive.speed = self.pb2_speed_mult * self.last_vx
            self.drive_publisher.publish(drive_msg)
        elif stage == "FB": # Fully brake and publish latch to stop other nodes from continuing to publish to drive
            kys_msg = Bool()
            kys_msg.data = True
            self.kys_publisher.publish(kys_msg)

            drive_msg.drive.speed = 0.0
            self.drive_publisher.publish(drive_msg)
            self.kys_latched = True
            self.set_parameters([Parameter('kys_latched', Parameter.Type.BOOL, True)])    


    def on_param_change(self, params: List[Parameter]) -> SetParametersResult:
        """
        Handle dynamic parameter updates.

        Updates internal safety thresholds and configuration values when parameters
        are changed at runtime via ROS2 parameter services.

        Args:
            params (List[Parameter]): List of updated parameters.

        Returns:
            SetParametersResult: Result indicating whether the update was successful.
        """
        for p in params:
            if p.name == 'front_half_angle':
                self.front_half_angle = p.value
            elif p.name == 'distance_threshold':
                self.distance_threshold = p.value
            elif p.name == 'kys_latched':
                self.kys_latched = p.value
        return SetParametersResult(successful=True)



def main(args=None) -> None:
    """
    Entry point for the safety node.
    """
    rclpy.init(args=args)
    node = SafetyNodeOld()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()