import rclpy
from milestone3.pid import PID
import numpy as np
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry

"""
    Wall-Follow Node: 
        Follows the right wall, using two lasers -90° (hard-right of the car) and -20° (forwards-right of the car) to gauge both distance and angle to the wall. 
"""
class WallNode(Node):
    """
    Reactive wall-following controller.

    Uses two right-side LIDAR rays to estimate wall angle/distance, then applies PID
    to compute a steering command and adjusts speed based on turn sharpness.
    """
    
    def __init__(self) -> None:
        """
        Initialize the wall-following node.

        Sets up ROS publishers, subscribers, parameters, and initializes
        the PID controller along with internal state variables.
        """
        super().__init__('wall_node')

        # ROS Interfaces
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.listener_callback, 10)
        self.kys_sub = self.create_subscription(Bool, '/kys', self.kys_callback, 10)
        self.vel_sub = self.create_subscription(Odometry, '/odom', self.velocity_callback, 10)
        self.publisher_ = self.create_publisher(AckermannDriveStamped, '/drive', 10)

        # Parameter Declarations & Value Assignment
        self.K_p = self.declare_parameter('K_p', 1.5).value
        self.K_i = self.declare_parameter('K_i', 0.0).value
        self.K_d = self.declare_parameter('K_d', 0.02).value

        self.K_p = self.get_parameter('K_p').get_parameter_value().double_value
        self.K_i = self.get_parameter('K_i').get_parameter_value().double_value
        self.K_d = self.get_parameter('K_d').get_parameter_value().double_value

        self.declare_parameter('target_distance', 1.0)
        self.declare_parameter('max_speed', 3.0)
        self.declare_parameter('min_speed', 0.1)
        self.declare_parameter('K_speed', 1.0)
        self.declare_parameter('kys_latched', False)

        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.K_speed = self.get_parameter('K_speed').get_parameter_value().double_value

        self.pid = PID(self.K_p, self.K_i, self.K_d)

        self.kys = self.get_parameter('kys_latched').get_parameter_value().bool_value

        self.last_vel = None
        self.prev_time = None

    def listener_callback(self, msg: LaserScan) -> None:
        """
        This callback function is continously invoked whenever new LaserScan data is recevied from LIDAR. 
        It computes the distance error relative to the desired wall distance, then applies a PID controller function
        to compute a steering correction, and publishes an AckermannDriveStamped command to control the vehicle.

        Args:
            msg (LaserScan): LIDAR scan message
        """

        current_time = self.get_clock().now().nanoseconds / 1e9 # clock in seconds (nano / 10^9)

        self.pid.K_p = self.get_parameter('K_p').get_parameter_value().double_value
        self.pid.K_i = self.get_parameter('K_i').get_parameter_value().double_value
        self.pid.K_d = self.get_parameter('K_d').get_parameter_value().double_value

        self.target_distance = self.get_parameter('target_distance').get_parameter_value().double_value
        self.max_speed = self.get_parameter('max_speed').get_parameter_value().double_value
        self.min_speed = self.get_parameter('min_speed').get_parameter_value().double_value
        self.K_speed = self.get_parameter('K_speed').get_parameter_value().double_value
        # Allow manual reset via: ros2 param set /wall_node kys_latched false
        self.kys = self.get_parameter('kys_latched').get_parameter_value().bool_value

        if self.prev_time is None:
            dt = 0.0
        else:
            dt = current_time - self.prev_time

        err = get_error(msg.ranges, msg.range_min, msg.range_max, self.target_distance, self.last_vel, dt) # 0.8 is the target distance to the wall, parametrize and adjust as necessary
        pid_err = self.pid.pid_err(err, current_time)

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = pid_err

        # 1.0 - changes how much the car slows on sharp turns, tune as necessary
        speed = self.max_speed - np.abs(pid_err) * self.K_speed # 2.5 is max speed, probs should parametrize and tune as necessary
        if speed < self.min_speed:
            speed = self.min_speed
        
        drive_msg.drive.speed = float(speed)
    
        if self.kys is False:
            self.publisher_.publish(drive_msg)
        else:
            drive_msg.drive.speed = 0.0
            self.publisher_.publish(drive_msg)        
        self.get_logger().info('%s' % pid_err)
        self.prev_time = current_time

    def kys_callback(self, msg: Bool) -> None:
        """
        Emergency stop callback. Latches the stop flag when the safety node asserts it.
        Once latched, the vehicle will remain stopped until manually reset.

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
        

def get_range(range_data, angle) -> float:
    """
        Returns the range data associated with the given angle (-135° to +135°)
        
        Args: 
            range_data - array of distances corresponding to each angle interval (0.25°)
            angle - angle whose distance we're looking for

        Returns: 
            float: Distance measurement corresponding to the given angle.

        Angles:
            0 - straight forward 
            Positive - left of car
            Negative - right of car 

        Angle-to-index conversion: add 135 to offset -135° start, multiply by 4 to convert from intervals (0.25°) to specific indices 
    """

    return range_data[(angle + 135)* 4]


def get_error(range_data, range_min, range_max, dist, vel, dt) -> float:
    """ 
        Calculates the error in distance using two LIDAR rays to gauge car's orientation 

            Two LIDAR rays are used:
                a: distance measured at -20° 
                b: distance measured at -90° 
            Using two measured distances and a geometric projection approach, the vehicle’s
            orientation relative to the wall is estimated. This orientation angle α is
            computed using the following geometric relationship:
            
            α = arctan((a·cosθ − b) / (a·sinθ))
            
            Args:
                range_data: Array of LIDAR range measurements.
                range_min (float): Minimum valid LIDAR range.
                range_max (float): Maximum valid LIDAR range.
                dist (float): Desired distance from the wall.
                vel (float | None): Current forward velocity.
                dt (float): Time step since last update. 
            Returns:
                float: Distance error.
    """
    theta = np.radians(70)
    dist_b = get_range(range_data, -90)
    dist_a = get_range(range_data, -20)
    
    if dist_a < range_min or dist_a > range_max or dist_b < range_min or dist_b > range_max:
        return 0.0

    alpha = np.arctan((dist_a*np.cos(theta) - dist_b) / (dist_a*np.sin(theta)))

    AB = dist_b*np.cos(alpha)

    if vel is None: vel = 0

    CD = AB + vel*dt*np.sin(alpha)

    ret = dist - CD

    if np.abs(ret) < 0.02:
        return 0.0

    return ret

def main(args=None) -> None:
    """
    Entry point for the wall-following node.
    """
    rclpy.init(args=args)
    wall_node = WallNode()
    rclpy.spin(wall_node)
    wall_node.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
