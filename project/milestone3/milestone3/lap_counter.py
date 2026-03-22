#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import numpy as np
import time
from std_msgs.msg import Bool
from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter
from typing import List

class LapCounter(Node): 
    """
    Vision-based lap counting controller.

    Uses ORB feature descriptors to compare the current camera frame
    against an initial reference frame captured at startup. When
    similarity exceeds a defined threshold after leaving the start
    region, a lap is counted.

    A time-based debounce mechanism prevents false increments due
    to oscillations near the start location.
    """
    def __init__(self):
        """
        Initialize the lap counter node.

        Sets up ROS publishers/subscribers, declares parameters,
        initializes ORB feature extraction, and prepares internal
        state variables for lap detection and stop latching.
        """
        super().__init__('lap_counter_node')

        self.cam_sub = self.create_subscription(Image, '/camera/color/image_raw', self.cam_callback, 10)
        self.kys_publisher = self.create_publisher(Bool, '/kys', 10)

        self.bridge = CvBridge()
        self.kys_latched = False

        self.lap_max = self.declare_parameter('lap_max', 2).value
        self.add_on_set_parameters_callback(self.on_param_change)

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
        """
        Extract ORB feature descriptors from a frame.

        Args:
            frame (np.ndarray): BGR image frame.

        Returns:
            np.ndarray | None:
                ORB descriptor array if features are found,
                otherwise None.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, descriptors = self.orb.detectAndCompute(gray, None)
        return descriptors

    def get_similarity(self, desc1, desc2) -> float:
        """
        Compute similarity score between two ORB descriptor sets.

        Uses brute-force Hamming matching and averages the best
        50 match distances. Score is normalized to approximately
        [0, 1], where higher values indicate greater similarity.

        Args:
            desc1 (np.ndarray): Reference descriptors.
            desc2 (np.ndarray): Current frame descriptors.

        Returns:
            float: Normalized similarity score.
        """
        matches = self.bf.match(desc1, desc2)
        if len(matches) == 0:
            return 0.0
        matches = sorted(matches, key=lambda x: x.distance)
        score = sum(m.distance for m in matches[:50]) / 50
        # normalize to 0-1 where 1 = most similar
        return 1 - (score / 100)

    def log_status(self) -> None:
        """
        Periodic status logger.

        Logs current similarity score, whether the vehicle is
        considered near the start position, and the current lap count.
        """
        self.get_logger().info(f'Similarity: {self.last_similarity:.3f} | near_start: {self.near_start} | Lap Number: {self.lap_count}')

    def cam_callback(self, msg) -> None: 
        """
        Camera image callback.

        Converts the ROS image to OpenCV format, extracts ORB
        descriptors, and compares them to the reference frame.
        Lap count increments when:

            1. The vehicle leaves the start region (low similarity).
            2. The vehicle re-enters the start region (high similarity).
            3. A minimum time interval has passed since the last lap.

        When lap_count reaches lap_max, a latched stop signal
        is published.

        Args:
            msg (Image): Incoming RGB image message.
        """
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

        if self.lap_count == self.lap_max:
            kys_msg = Bool()
            kys_msg.data = True
            self.kys_publisher.publish(kys_msg)
            self.kys_latched = True

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
            if p.name == 'lap_max':
                self.lap_max = int(p.value)

        return SetParametersResult(successful=True)


def main(args=None) -> None:
    """
    Entry point for the lap counter node.
    """
    rclpy.init(args=args)
    node = LapCounter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
