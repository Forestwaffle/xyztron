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
from track_drive.auto_drive import AutoDrive


class MainDrivingNode(Node):
    def __init__(self):
        super().__init__('main_driving_node')

        self.get_logger().info("Main driving node started")

        # =====================================================
        # Basic variables
        # =====================================================
        self.bridge = CvBridge()
        self.image = None

        self.lidar_ranges = None
        self.lidar_data_received = False
        self.lidar_callback_count = 0

        self.motor_msg = XycarMotor()

        self.auto_drive_speed = 8.0

        # =====================================================
        # QoS
        # =====================================================
        self.qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # =====================================================
        # Subscribers
        # =====================================================
        self.camera_sub = self.create_subscription(
            Image,
            '/usb_cam/image_raw/front',
            self.camera_callback,
            self.qos
        )

        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            self.qos
        )

        # =====================================================
        # Publisher
        # =====================================================
        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # =====================================================
        # Mission modules
        # =====================================================
        self.traffic_light = TrafficLightDetector(show_debug=True)

        self.cone_driver = ConeLidarDriver(
            logger=self.get_logger(),
            show_debug=False
        )

        self.cone_driver_started = False

        # AUTO_DRIVE 로직은 auto_drive.py 파일에서 불러와 사용
        self.auto_drive = AutoDrive(
            logger=self.get_logger(),
            show_debug=True
        )

        # =====================================================
        # Mission state
        # =====================================================
        self.mission_state = "WAIT_TRAFFIC_LIGHT"

        self.prev_traffic_light_state = None
        self.prev_mission_state = None

        # =====================================================
        # Main timer: 50 Hz
        # =====================================================
        self.timer = self.create_timer(0.02, self.control_loop)

    # =====================================================
    # Camera callback
    # =====================================================

    def camera_callback(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding='bgr8'
            )
        except Exception as e:
            self.get_logger().error(f"Camera conversion failed: {e}")

    # =====================================================
    # LiDAR callback
    # =====================================================

    def lidar_callback(self, msg):
        # 최신 라이다 값으로 계속 덮어쓰기
        self.lidar_ranges = msg.ranges
        self.lidar_callback_count += 1

        if not self.lidar_data_received:
            self.get_logger().info(
                f"LiDAR data received: {len(msg.ranges)} beams"
            )
            self.lidar_data_received = True

        # 로그 너무 많이 찍히지 않게 20번마다 출력
        if self.lidar_callback_count % 20 == 0:
            self.get_logger().info(
                f"LiDAR update #{self.lidar_callback_count} "
                f"| beams:{len(msg.ranges)}"
            )

    # =====================================================
    # Motor publish
    # =====================================================

    def drive(self, angle, speed):
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

    # =====================================================
    # Utility
    # =====================================================

    def close_traffic_light_window(self):
        try:
            cv2.destroyWindow("Traffic Light Detector")
            cv2.waitKey(1)
        except cv2.error:
            pass

    def log_mission_state_changed(self):
        if self.mission_state != self.prev_mission_state:
            self.get_logger().info(
                f"Mission state changed: {self.mission_state}"
            )
            self.prev_mission_state = self.mission_state

    def log_traffic_light_state_changed(self, state):
        if state != self.prev_traffic_light_state:
            self.get_logger().info(
                f"Traffic light state changed: {state}"
            )
            self.prev_traffic_light_state = state

    # =====================================================
    # Main control loop
    # =====================================================

    def control_loop(self):
        self.log_mission_state_changed()

        # =====================================================
        # Mission 1: WAIT_TRAFFIC_LIGHT
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
                # 초록불을 봐도 바로 급출발하지 않고 일단 정지 명령
                self.drive(0, 0)

                # 신호등 미션 종료
                self.traffic_light.disable()
                self.close_traffic_light_window()

                # 라바콘 주행으로 전환
                self.mission_state = "CONE_DRIVE"

                self.get_logger().info(
                    "Traffic light mission complete -> CONE_DRIVE"
                )

        # =====================================================
        # Mission 2: CONE_DRIVE
        # =====================================================
        elif self.mission_state == "CONE_DRIVE":
            if self.lidar_ranges is None:
                self.get_logger().warn("Waiting for LiDAR data...")
                self.drive(0, 0)
                return

            if not self.cone_driver_started:
                self.cone_driver.start()
                self.cone_driver_started = True

            angle, speed = self.cone_driver.process(self.lidar_ranges)

            # left > 53m 또는 left invalid ?
            #   └─ 예 → AUTO_DRIVE
            if self.cone_driver.is_complete():
                self.get_logger().info(
                    "Cone mission complete. Switching to AUTO_DRIVE"
                )

                self.mission_state = "AUTO_DRIVE"

                # AUTO_DRIVE 진입 순간부터 별도 AutoDrive 로직 실행
                self.auto_drive.start()
                angle, speed = self.auto_drive.process(self.image)
                self.drive(angle, speed)
                return

            self.drive(angle, speed)

        # =====================================================
        # Mission 3: AUTO_DRIVE
        # =====================================================
        elif self.mission_state == "AUTO_DRIVE":
            # 별도 파일 auto_drive.py에 있는 AutoDrive 로직 실행
            # 현재 AutoDrive 기능: 정지 + 전방 카메라 디버그 창 표시
            self.auto_drive.start()
            angle, speed = self.auto_drive.process(self.image)
            self.drive(angle, speed)

        # =====================================================
        # Safety fallback
        # =====================================================
        else:
            self.get_logger().warn(
                f"Unknown mission state: {self.mission_state}"
            )
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
            node.auto_drive.stop()
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