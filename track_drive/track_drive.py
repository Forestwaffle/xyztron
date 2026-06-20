#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from xycar_msgs.msg import XycarMotor


class MainDrivingNode(Node):
    def __init__(self):
        super().__init__('main_driving_node')

        self.get_logger().info("Main driving node started")
        self.get_logger().info("Mode: LiDAR INFO ONLY")
        self.get_logger().info("LiDAR log range: index 80~100")

        # =====================================================
        # QoS
        # Unity / ROS-TCP 쪽은 RELIABLE 사용
        # =====================================================
        self.qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE
        )

        # =====================================================
        # LiDAR subscriber
        # 프로그램 시작하자마자 /scan 구독
        # =====================================================
        self.lidar_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            self.qos
        )

        # =====================================================
        # Motor publisher
        # 현재는 INFO 확인용이므로 계속 정지
        # =====================================================
        self.motor_msg = XycarMotor()

        self.motor_pub = self.create_publisher(
            XycarMotor,
            '/xycar_motor',
            10
        )

        # =====================================================
        # LiDAR data
        # =====================================================
        self.lidar_ranges = None
        self.lidar_data_received = False
        self.lidar_callback_count = 0

        # =====================================================
        # Log range
        # 이전과 동일하게 80~100만 출력
        # =====================================================
        self.log_index_min = 80
        self.log_index_max = 100

        # INFO 출력 주기
        # 0.25초마다 한 번 출력
        self.log_interval_sec = 0.25

        # 정지 명령 주기
        # 20Hz
        self.control_timer = self.create_timer(
            0.05,
            self.control_loop
        )

        # 라이다 INFO 출력 주기
        self.info_timer = self.create_timer(
            self.log_interval_sec,
            self.print_lidar_info
        )

    def lidar_callback(self, msg):
        """
        /scan 콜백.
        라이다 전체 360개 값을 계속 최신값으로 저장.
        """
        self.lidar_ranges = msg.ranges
        self.lidar_callback_count += 1

        if not self.lidar_data_received:
            self.get_logger().info(
                f"LiDAR data received: {len(msg.ranges)} beams"
            )
            self.lidar_data_received = True

    def control_loop(self):
        """
        현재는 라이다 확인용.
        차량은 계속 정지.
        """
        self.drive(0.0, 0.0)

    def drive(self, angle, speed):
        """
        Motor command publish.
        """
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

    def print_lidar_info(self):
        """
        index 80~100 라이다 값만 INFO로 출력.
        """
        if self.lidar_ranges is None:
            self.get_logger().info("LiDAR INFO | no data yet")
            return

        total_beams = len(self.lidar_ranges)

        start_index = max(0, self.log_index_min)
        end_index = min(total_beams - 1, self.log_index_max)

        raw_values_text = self.make_range_values_text(
            self.lidar_ranges,
            start_index,
            end_index
        )

        closest = self.get_closest_in_range(
            self.lidar_ranges,
            start_index,
            end_index
        )

        if closest is None:
            self.get_logger().info(
                f"LiDAR INFO "
                f"| cb:{self.lidar_callback_count} "
                f"| beams:{total_beams} "
                f"| range:[{start_index}~{end_index}] "
                f"| closest:invalid "
                f"| values:[{raw_values_text}]"
            )
            return

        closest_distance = closest["distance"]
        closest_index = closest["index"]
        closest_angle = closest["angle_deg"]
        closest_direction = closest["direction"]

        self.get_logger().info(
            f"LiDAR INFO "
            f"| cb:{self.lidar_callback_count} "
            f"| beams:{total_beams} "
            f"| range:[{start_index}~{end_index}] "
            f"| closest_dist:{closest_distance:.2f}m "
            f"| closest_index:{closest_index} "
            f"| closest_angle:{closest_angle:.2f}deg "
            f"| direction:{closest_direction} "
            f"| values:[{raw_values_text}]"
        )

    def make_range_values_text(self, lidar_ranges, start_index, end_index):
        """
        index 80~100 값을 문자열로 생성.
        """
        texts = []

        for index in range(start_index, end_index + 1):
            distance = lidar_ranges[index]

            if not math.isfinite(distance):
                texts.append(f"{index:03d}:inf")
            else:
                texts.append(f"{index:03d}:{distance:.2f}")

        return " ".join(texts)

    def get_closest_in_range(self, lidar_ranges, start_index, end_index):
        """
        지정 범위 안에서 가장 가까운 유효값 찾기.
        """
        candidates = []

        for index in range(start_index, end_index + 1):
            distance = lidar_ranges[index]

            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            angle_deg = self.index_to_angle_deg(index)
            direction = self.angle_to_direction(angle_deg)

            candidates.append(
                {
                    "distance": float(distance),
                    "index": int(index),
                    "angle_deg": float(angle_deg),
                    "direction": direction
                }
            )

        if len(candidates) == 0:
            return None

        candidates.sort(key=lambda item: item["distance"])

        return candidates[0]

    def index_to_angle_deg(self, index):
        """
        라이다 index를 차량 기준 각도로 변환.

        기준:
            index 90 = 전방 0도
            index 80 = 왼쪽 -10도
            index 100 = 오른쪽 +10도
        """
        angle = float(index) - 90.0

        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    def angle_to_direction(self, angle_deg):
        """
        각도를 방향 문자열로 변환.
        """
        if angle_deg is None:
            return "UNKNOWN"

        if -30.0 <= angle_deg <= 30.0:
            return "FRONT"

        if 30.0 < angle_deg <= 150.0:
            return "RIGHT"

        if -150.0 <= angle_deg < -30.0:
            return "LEFT"

        return "BACK"


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
                node.drive(0.0, 0.0)
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