#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        라바콘 두 줄 사이 중앙으로 전진

    입력:
        /scan LaserScan.ranges

    처리:
        전방 0.5m~4.0m 점만 사용
        좌우 -2.0m~2.0m 범위만 사용
        왼쪽 라바콘 점 평균
        오른쪽 라바콘 점 평균
        중앙점 계산

    출력:
        angle: 중앙점 방향 조향
        speed: 4.0 또는 5.0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # LiDAR ROI parameters
        # =====================================================
        self.front_min = 0.5
        self.front_max = 4.0
        self.side_limit = 2.0
        self.center_ignore_width = 0.20

        # 라바콘 두 줄 사이 예상 폭
        # 한쪽 라바콘만 보일 때 중앙점 보정용
        self.lane_width = 1.2

        # 조향 제어 파라미터
        self.lookahead_y = 2.0
        self.steering_gain = 1.5
        self.max_angle = 50.0

        # 속도 제어 파라미터
        self.fast_speed = 5.0
        self.slow_speed = 4.0
        self.slow_angle_threshold = 25.0

        # 현재 출력값
        self.angle = 0.0
        self.speed = 0.0

        # Debug viewer objects
        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.all_points_scatter = None
        self.left_points_scatter = None
        self.right_points_scatter = None
        self.target_point_scatter = None
        self.center_line_plot = None

        # Logging
        self.warned_no_lidar = False
        self.log_counter = 0

    def start(self):
        """
        CONE_DRIVE 상태에 처음 들어왔을 때 한 번 호출.
        """
        if self.show_debug:
            self.init_lidar_viewer()

        self.log_info("ConeLidarDriver started")

    def stop(self):
        """
        종료 시 정지 및 디버그 창 닫기.
        """
        self.angle = 0.0
        self.speed = 0.0

        if self.viewer_ready:
            try:
                plt.close(self.fig)
            except Exception:
                pass

        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.all_points_scatter = None
        self.left_points_scatter = None
        self.right_points_scatter = None
        self.target_point_scatter = None
        self.center_line_plot = None

        self.log_info("ConeLidarDriver stopped")

    def process(self, lidar_ranges):
        """
        목표:
            라바콘 두 줄 사이 중앙으로 전진

        입력:
            /scan LaserScan.ranges

        처리:
            1. 라이다 ranges를 x, y 좌표로 변환
            2. 전방 0.5m~4.0m 점만 사용
            3. 좌우 -2.0m~2.0m 범위만 사용
            4. 왼쪽 라바콘 점 평균 계산
            5. 오른쪽 라바콘 점 평균 계산
            6. 중앙점 계산
            7. 중앙점 방향으로 조향각 계산

        출력:
            angle: 중앙점 방향 조향
            speed: 4.0 또는 5.0
        """
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            return 0.0, 0.0

        self.warned_no_lidar = False

        # 1. LaserScan.ranges -> x, y 좌표 변환
        points = self.lidar_ranges_to_xy(lidar_ranges)

        # 2. 전방 ROI 필터링
        roi_points = self.filter_front_roi(points)

        # 3. 왼쪽 / 오른쪽 라바콘 후보 분리
        left_points, right_points = self.split_left_right(roi_points)

        # 4. 중앙 목표점 계산
        target = self.calculate_center_target(left_points, right_points)

        if self.show_debug:
            self.update_lidar_viewer(
                all_points=points,
                left_points=left_points,
                right_points=right_points,
                target=target
            )

        # 중앙점을 못 찾으면 안전 정지
        if target is None:
            self.angle = 0.0
            self.speed = 0.0

            self.print_debug(
                left_points=left_points,
                right_points=right_points,
                target=None
            )

            return self.angle, self.speed

        target_x, target_y = target

        # 5. 중앙점 방향으로 조향각 계산
        self.angle = self.calculate_steering_angle(target_x, target_y)

        # 6. 속도 계산: 조향이 크면 4.0, 작으면 5.0
        self.speed = self.calculate_speed(self.angle)

        self.print_debug(
            left_points=left_points,
            right_points=right_points,
            target=target
        )

        return self.angle, self.speed

    def lidar_ranges_to_xy(self, lidar_ranges):
        """
        LaserScan.ranges를 차량 기준 x, y 좌표로 변환.

        좌표계:
            y축 + : 차량 전방
            x축 - : 차량 왼쪽
            x축 + : 차량 오른쪽

        현재 가정:
            index 90 = 전방
            index 0 = 왼쪽
            index 180 = 오른쪽
        """
        points = []

        for i, distance in enumerate(lidar_ranges):
            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            # index 90을 전방 0도로 사용
            angle_deg = i - 90
            angle_rad = math.radians(angle_deg)

            x = distance * math.sin(angle_rad)
            y = distance * math.cos(angle_rad)

            points.append((x, y))

        if len(points) == 0:
            return np.empty((0, 2), dtype=float)

        return np.array(points, dtype=float)

    def filter_front_roi(self, points):
        """
        처리:
            전방 0.5m~4.0m 점만 사용
            좌우 -2.0m~2.0m 범위만 사용
        """
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=float)

        x = points[:, 0]
        y = points[:, 1]

        mask = (
            (y >= self.front_min) &
            (y <= self.front_max) &
            (x >= -self.side_limit) &
            (x <= self.side_limit)
        )

        return points[mask]

    def split_left_right(self, roi_points):
        """
        처리:
            왼쪽 라바콘 점 평균을 위해 왼쪽 점 분리
            오른쪽 라바콘 점 평균을 위해 오른쪽 점 분리

        가운데 근처 점은 노이즈로 보고 무시.
        """
        if roi_points is None or len(roi_points) == 0:
            empty = np.empty((0, 2), dtype=float)
            return empty, empty

        x = roi_points[:, 0]

        left_mask = x < -self.center_ignore_width
        right_mask = x > self.center_ignore_width

        left_points = roi_points[left_mask]
        right_points = roi_points[right_mask]

        return left_points, right_points

    def calculate_center_target(self, left_points, right_points):
        """
        처리:
            왼쪽 라바콘 점 평균
            오른쪽 라바콘 점 평균
            중앙점 계산
        """
        has_left = left_points is not None and len(left_points) > 0
        has_right = right_points is not None and len(right_points) > 0

        if not has_left and not has_right:
            return None

        # 양쪽 라바콘이 모두 보이는 경우
        if has_left and has_right:
            left_x = float(np.mean(left_points[:, 0]))
            right_x = float(np.mean(right_points[:, 0]))

            center_x = (left_x + right_x) / 2.0
            center_y = self.lookahead_y

            return center_x, center_y

        # 왼쪽 라바콘만 보이는 경우
        if has_left and not has_right:
            left_x = float(np.mean(left_points[:, 0]))

            # 왼쪽 라바콘에서 오른쪽으로 lane_width/2 만큼 떨어진 곳을 중앙으로 가정
            center_x = left_x + (self.lane_width / 2.0)
            center_y = self.lookahead_y

            return center_x, center_y

        # 오른쪽 라바콘만 보이는 경우
        if has_right and not has_left:
            right_x = float(np.mean(right_points[:, 0]))

            # 오른쪽 라바콘에서 왼쪽으로 lane_width/2 만큼 떨어진 곳을 중앙으로 가정
            center_x = right_x - (self.lane_width / 2.0)
            center_y = self.lookahead_y

            return center_x, center_y

        return None

    def calculate_steering_angle(self, target_x, target_y):
        """
        출력:
            angle: 중앙점 방향 조향

        target_x가 오른쪽이면 angle 양수,
        target_x가 왼쪽이면 angle 음수.
        """
        if target_y <= 0.0:
            return 0.0

        raw_angle = math.degrees(math.atan2(target_x, target_y))
        steer_angle = raw_angle * self.steering_gain

        steer_angle = max(-self.max_angle, min(self.max_angle, steer_angle))

        return steer_angle

    def calculate_speed(self, angle):
        """
        출력:
            speed: 4.0 또는 5.0
        """
        if abs(angle) >= self.slow_angle_threshold:
            return self.slow_speed

        return self.fast_speed

    def init_lidar_viewer(self):
        """
        디버깅용 LiDAR Viewer 생성.
        """
        if self.viewer_ready:
            return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("LiDAR Cone Drive Debug Viewer")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-1, 5)
        self.ax.grid(True)

        self.ax.set_xlabel("x: left(-) / right(+)")
        self.ax.set_ylabel("y: front(+)")

        # 전체 라이다 점
        self.all_points_scatter = self.ax.scatter([], [], s=5, c='gray')

        # 왼쪽 라바콘 후보 점
        self.left_points_scatter = self.ax.scatter([], [], s=20, c='blue')

        # 오른쪽 라바콘 후보 점
        self.right_points_scatter = self.ax.scatter([], [], s=20, c='orange')

        # 중앙 목표점
        self.target_point_scatter = self.ax.scatter([], [], s=80, c='red')

        # 차량 중심
        self.ax.plot(0, 0, 'ko')

        # 전방 방향 표시
        self.ax.plot([0, 0], [0, 1.0], 'k-')
        self.ax.text(0.1, 1.0, "FRONT")

        # ROI 박스 표시
        roi_x = [
            -self.side_limit,
            self.side_limit,
            self.side_limit,
            -self.side_limit,
            -self.side_limit
        ]

        roi_y = [
            self.front_min,
            self.front_min,
            self.front_max,
            self.front_max,
            self.front_min
        ]

        self.ax.plot(roi_x, roi_y, 'g--')

        # 중앙 목표선
        self.center_line_plot, = self.ax.plot([], [], 'r-')

        plt.ion()
        plt.show(block=False)

        self.viewer_ready = True
        self.log_info("LiDAR cone debug viewer started")

    def update_lidar_viewer(self, all_points, left_points, right_points, target):
        """
        라이다 디버깅 뷰어 업데이트.
        """
        if not self.viewer_ready:
            self.init_lidar_viewer()

        self.all_points_scatter.set_offsets(self.make_offsets(all_points))
        self.left_points_scatter.set_offsets(self.make_offsets(left_points))
        self.right_points_scatter.set_offsets(self.make_offsets(right_points))

        if target is not None:
            target_x, target_y = target
            self.target_point_scatter.set_offsets(
                np.array([[target_x, target_y]], dtype=float)
            )

            self.center_line_plot.set_data(
                [0.0, target_x],
                [0.0, target_y]
            )
        else:
            self.target_point_scatter.set_offsets(np.empty((0, 2)))
            self.center_line_plot.set_data([], [])

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def make_offsets(self, points):
        """
        matplotlib scatter용 좌표 배열 생성.
        """
        if points is None or len(points) == 0:
            return np.empty((0, 2), dtype=float)

        return points

    def print_debug(self, left_points, right_points, target):
        """
        약 1초마다 터미널 디버깅 출력.
        """
        self.log_counter += 1

        if self.log_counter % 20 != 0:
            return

        left_count = 0 if left_points is None else len(left_points)
        right_count = 0 if right_points is None else len(right_points)

        if target is None:
            self.log_warn(
                f"CONE LOG | left:{left_count} right:{right_count} "
                f"| target:None | angle:0.00 speed:0.00"
            )
            return

        target_x, target_y = target

        self.log_info(
            f"CONE LOG | left:{left_count} right:{right_count} "
            f"| target_x:{target_x:.2f} target_y:{target_y:.2f} "
            f"| angle:{self.angle:.2f} speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)