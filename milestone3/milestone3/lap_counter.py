#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
from std_msgs.msg import Bool

class LapCounter(Node): 
    def __init__(self):
        super().__init__('lap_counter_node')

        self.cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.cam_callback, 10)
        self.kys_publisher = self.create_publisher(Bool, '/kys', 10)

        self.bridge = CvBridge()
        self.kys_latched = False

        self.orb = cv2.ORB_create()
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        self.ref_descriptors = None
        self.near_start = True
        self.lap_count = 0
        self.increment_ready = True

        # lap timer so that sudden changes (sudden back-and-forth near_start values don't increment the lap accidentaly)
        self.lap_time = 20
        self.last_lap_time = 0

        self.create_timer(1, self.log_status)
        self.last_similarity = 0.0

        self.get_logger().info('Lap node started, waiting for first frame...')

    def get_descriptors(self, frame) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.orb.detectAndCompute(gray, None)
        return descriptors

    def get_similarity(self, desc1, desc2) -> float:
        matches = self.bf.match(desc1, desc2)
        if len(matches) == 0:
            return 0.0
        matches = sorted(matches, key=lambda x: x.distance)
        score = sum(m.distance for m in matches[:50]) / 50
        # normalize to 0-1 where 1 = most similar
        return 1 - (score / 100)

    def log_status(self):
        self.get_logger().info(f'Similarity: {self.last_similarity:.3f} | near_start: {self.near_start} | Lap Number: {self.lap_count}')

    def cam_callback(self, msg) -> None: 
        if self.kys_latched:
            return
        
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        descriptors = self.get_descriptors(frame)
        if descriptors is None:
            return

        if self.ref_descriptors is None:
            self.ref_descriptors = descriptors
            self.get_logger().info('Reference image captured')
            return

        similarity = self.get_similarity(self.ref_descriptors, descriptors)
        self.last_similarity = similarity

        if self.near_start:
            if similarity < 0.7:
                self.near_start = False
        else:
            if similarity >= 0.9:
                if time.time() - self.last_lap_time >= self.lap_time:
                    self.near_start = True
                    self.lap_count += 1
                    self.last_lap_time = time.time()
                    self.get_logger().info(f'Lap detected! Similarity: {similarity:.3f}')

                self.get_logger().info(f'Lap Number:  {self.lap_count:.3f}')

        if self.lap_count == 2: # hardcoded lap number to KYS latch
            kys_msg = Bool()
            kys_msg.data = True
            self.kys_publisher.publish(kys_msg)
            self.kys_latched = True


def main(args=None):
    rclpy.init(args=args)
    node = LapCounter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()