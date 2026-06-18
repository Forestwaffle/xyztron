#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

from xycar_msgs.msg import XycarMotor
from track_drive.traffic_light_detector import TrafficLightDetector
from track_drive.cone_lidar_driver import ConeLidarDriver


class MainDrivingNode(Node):
    def __init__(self):
        super().__init__('main_driving_node')

        self.get_logger().info("Main driving node started")

        self.bridge = CvBridge()
        self.image = None

        self.motor_msg = XycarMotor()

        self.qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # =====================================================
        # Camera starts first.
        # LiDAR is NOT subscribed here.
        # =====================================================
        self.camera_sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            self.qos
        )

        # =====================================================
        # LiDAR starts only after CONE_DRIVE.
        # =====================================================
        self.lidar_sub = None
        self.lidar_started = False
        self.lidar_ranges = None
        self.lidar_data_received = False

        # Motor publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # Traffic light detector
        self.traffic_light = TrafficLightDetector(show_debug=True)

        # Cone LiDAR driver
        # 실제 라이다 주행 로직은 cone_lidar_driver.py에서 처리
        self.cone_driver = ConeLidarDriver(
            logger=self.get_logger(),
            show_debug=True
        )

        self.cone_driver_started = False

        # Mission state
        self.mission_state = "WAIT_TRAFFIC_LIGHT"

        # Previous states for terminal logging
        self.prev_traffic_light_state = None
        self.prev_mission_state = None

        # Main loop: 20 Hz
        self.timer = self.create_timer(0.05, self.control_loop)

    def camera_callback(self, msg):
        """Store latest front camera image."""
        try:
            self.image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    def start_lidar(self):
        """Start LiDAR subscriber after mission state changes to CONE_DRIVE."""
        if self.lidar_started:
            return

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            self.qos
        )

        self.lidar_started = True
        self.get_logger().info("LiDAR subscriber started after CONE_DRIVE")

    def lidar_callback(self, msg):
        """Store latest LiDAR scan data."""
        self.lidar_ranges = msg.ranges

        if not self.lidar_data_received:
            self.get_logger().info(
                f"LiDAR data received: {len(msg.ranges)} beams"
            )
            self.lidar_data_received = True

    def drive(self, angle, speed):
        """Publish motor command."""
        self.motor_msg.angle = float(angle)
        self.motor_msg.speed = float(speed)
        self.motor_pub.publish(self.motor_msg)

    def close_traffic_light_window(self):
        """Close traffic light debug window safely."""
        try:
            cv2.destroyWindow("Traffic Light Detector")
            cv2.waitKey(1)
        except cv2.error:
            pass

    def log_mission_state_changed(self):
        """Log mission state only when it changes."""
        if self.mission_state != self.prev_mission_state:
            self.get_logger().info(
                f"Mission state changed: {self.mission_state}"
            )
            self.prev_mission_state = self.mission_state

    def log_traffic_light_state_changed(self, state):
        """Log traffic light state only when it changes."""
        if state != self.prev_traffic_light_state:
            self.get_logger().info(
                f"Traffic light state changed: {state}"
            )
            self.prev_traffic_light_state = state

    def control_loop(self):
        """Main mission logic."""
        if self.image is None:
            self.drive(0, 0)
            return

        self.log_mission_state_changed()

        # =====================================================
        # WAIT_TRAFFIC_LIGHT:
        #   1. 처음 실행
        #      - 앞 카메라 구독
        #      - Traffic Light Detector 창 켜짐
        #      - 라이다는 아직 구독하지 않음
        # =====================================================
        if self.mission_state == "WAIT_TRAFFIC_LIGHT":
            if not self.traffic_light.active:
                self.traffic_light.enable()

            state, detected_light, debug_frame = self.traffic_light.process(
                self.image
            )

            self.log_traffic_light_state_changed(state)

            if state == "STOP":
                self.drive(0, 0)

            elif state == "GO":
                # =====================================================
                # 2. 초록불 감지
                #   - Traffic Light Detector 창 닫힘
                #   - mission_state = CONE_DRIVE
                # =====================================================
                self.drive(0, 0)

                self.traffic_light.disable()
                self.close_traffic_light_window()

                self.mission_state = "CONE_DRIVE"

                self.get_logger().info("Traffic light mission complete")

        # =====================================================
        # CONE_DRIVE:
        #   3. CONE_DRIVE 진입
        #      - /scan 라이다 구독 시작
        #      - 원래 LiDAR Debug Viewer 창 켜짐
        #      - 기본 직진
        #      - 가까운 장애물 있으면 정지
        # =====================================================
        elif self.mission_state == "CONE_DRIVE":
            self.start_lidar()

            if not self.cone_driver_started:
                self.cone_driver.start()
                self.cone_driver_started = True

            angle, speed = self.cone_driver.process(self.lidar_ranges)

            self.drive(angle, speed)

        # =====================================================
        # Safety stop
        # =====================================================
        else:
            self.drive(0, 0)


def main(args=None):
    rclpy.init(args=args)

    node = MainDrivingNode()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.drive(0, 0)
        node.cone_driver.stop()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()