import rclpy
import numpy as np
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from ackermann_msgs.msg import AckermannDriveStamped
from std_msgs.msg import Bool
from milestone3.pid import PID

import cv_bridge

class CamNode(Node):
    def __init__(self):
        super().__init__('cam_node')
        self.cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.cam_callback, 10)
        self.kys_sub = self.create_subscription(Bool, '/kys', self.kys_callback, 10)
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.speed_sub = self.create_subscription(AckermannDriveStamped, '/speed', self.speed_callback, 10)
        self.run = 0
        self.depth_img = None

        self.bridge = cv_bridge.CvBridge()

        self.pid = PID(K_p=1.5, K_i=0.0, K_d=0.05)

        self.kys_latched = False
        self.speed = 0.0

    def cam_callback(self, msg) -> None:

        if self.kys_latched:
            return

        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        path_img, success = cam_filter_path(img)

        if not success:
            self.get_logger().info("No path! ruh roh")
            return

        x, y, straight = get_target(path_img)

        img_w = img.shape[1]

        x_target = x - (img_w / 2)

        angle = - np.arctan2(x_target, y) - 0.2

        self.get_logger().info(f'angle: {angle}')
        
        pid_angle = self.pid.pid_err(angle, self.get_clock().now().nanoseconds * 1e-9)

        if straight:
            pid_angle = 0.0

        drive_msg = AckermannDriveStamped()
        drive_msg.drive.steering_angle = pid_angle

        # speed = max(self.min_speed, self.max_speed - self.K_speed * abs(pid_angle))

        drive_msg.drive.speed = self.speed

        self.drive_pub.publish(drive_msg)


        if self.run == 5:
            cv2.imwrite('img.png', path_img)
            # cv2.imwrite('depth_image.png', self.depth_img)

        self.run += 1
    
    def kys_callback(self, msg) -> None:
        if msg.data:
            self.kys_latched = True

    def speed_callback(self, msg) -> None:
        self.speed = msg.drive.speed

def cam_filter_path(img) -> (np.ndarray, bool):

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((9, 9), np.uint8)
    img = cv2.erode(img, kernel, iterations=2)
    img = cv2.dilate(img, kernel, iterations=2)
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros(img.shape[:2], dtype="uint8")

    success = False

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if y > 200 and area > 10000:
            cv2.drawContours(mask, [contour], -1, 255, -1)
            success = True
            break

    return (mask, success)

def get_target(img, target_row=400) -> (float, float, bool):

    row = img[target_row,:]
    indices = np.argwhere(row > 0)
    if len(indices) == 0:
        return (0, target_row, True)
    return (np.mean(indices), target_row, False)

def main(args=None) -> None:
    rclpy.init(args=args)
    cam_node = CamNode()
    rclpy.spin(cam_node)
    cam_node.destroy_node()
    rclpy.shutdown()
