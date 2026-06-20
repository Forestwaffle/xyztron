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
        # Camera subscriber
        # =====================================================
        self.camera_sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            self.qos
        )

        # =====================================================
        # LiDAR subscriber
        # 프로그램 시작부터 /scan 계속 구독
        # =====================================================
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            self.qos
        )

        self.lidar_ranges = None
        self.lidar_data_received = False
        self.lidar_callback_count = 0

        # Motor publisher
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # Traffic light detector
        self.traffic_light = TrafficLightDetector(show_debug=True)

        # Cone LiDAR driver
        self.cone_driver = ConeLidarDriver(
            logger=self.get_logger(),
            show_debug=True
        )

        self.cone_driver_started = False

        # Mission state
        self.mission_state = "WAIT_TRAFFIC_LIGHT"

        # Previous states for logging
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

    def lidar_callback(self, msg):
        """
        Store latest LiDAR scan data.

        /scan은 프로그램 시작부터 계속 들어온다.
        self.lidar_ranges에는 항상 최신값만 저장된다.
        """
        self.lidar_ranges = msg.ranges
        self.lidar_callback_count += 1

        if not self.lidar_data_received:
            self.get_logger().info(
                f"LiDAR data received: {len(msg.ranges)} beams"
            )
            self.lidar_data_received = True

        # 라이다 콜백이 계속 들어오는지 확인용
        if self.lidar_callback_count % 20 == 0:
            self.get_logger().info(
                f"LiDAR update #{self.lidar_callback_count} "
                f"| beams:{len(msg.ranges)}"
            )

    def drive(self, angle, speed):
        """Publish motor command."""
        if not rclpy.ok():
            return

        try:
            self.motor_msg.angle = float(angle)
            self.motor_msg.speed = float(speed)
            self.motor_pub.publish(self.motor_msg)
        except Exception as e:
            try:
                self.get_logger().warn(f"Motor publish skipped: {e}")
            except Exception:
                pass

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
        self.log_mission_state_changed()

        # =====================================================
        # WAIT_TRAFFIC_LIGHT:
        #   - 신호등만 판단
        #   - 라이다는 받고 있지만 사용하지 않음
        #   - 차량은 정지
        # =====================================================
        if self.mission_state == "WAIT_TRAFFIC_LIGHT":
            if self.image is None:
                self.drive(0, 0)
                return

            if not self.traffic_light.active:
                self.traffic_light.enable()

            state, detected_light, debug_frame = self.traffic_light.process(
                self.image
            )

            self.log_traffic_light_state_changed(state)

            if state == "STOP":
                self.drive(0, 0)

            elif state == "GO":
                self.drive(0, 0)

                self.traffic_light.disable()
                self.close_traffic_light_window()

                self.mission_state = "CONE_DRIVE"

                self.get_logger().info("Traffic light mission complete")

        # =====================================================
        # CONE_DRIVE:
        #   - 이미 받아둔 최신 라이다 self.lidar_ranges 사용
        #   - STOP/GO 판단은 cone_lidar_driver.py에서 수행
        # =====================================================
        elif self.mission_state == "CONE_DRIVE":
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
        try:
            if rclpy.ok():
                node.drive(0, 0)
        except Exception:
            pass

        try:
            node.cone_driver.stop()
        except Exception:
            pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        try:
            node.destroy_node()
        except Exception:
            pass

        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()