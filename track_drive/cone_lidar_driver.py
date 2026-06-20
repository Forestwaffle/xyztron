#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    LiDAR Debug Viewer 기준:
        index 0   = 정면
        index 90  = 오른쪽
        index 180 = 뒤쪽
        index 270 = 왼쪽

    이번 로직:
        정면 기준 좌/우 전방을 본다.

        left  : 0~60
        right : 300~359

    중앙 찾기:
        left_median과 right_median을 비교한다.

        left_median < right_median:
            왼쪽이 더 가까움
            오른쪽으로 조향

        right_median < left_median:
            오른쪽이 더 가까움
            왼쪽으로 조향

        두 값이 비슷하면:
            중앙에 있다고 보고 직진
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 5.0
        self.slow_speed = 3.0

        # 조향 gain
        # 값이 클수록 더 많이 꺾음
        self.steering_gain = 25.0

        # 최대 조향 제한
        self.max_angle = 50.0

        # 중앙이라고 판단할 좌우 차이
        self.center_deadband = 0.15

        # 너무 가까우면 긴급 정지
        self.emergency_stop_distance = 0.30

        # 조향 방향 보정값
        # 현재 가정:
        #   angle 양수 = 오른쪽 조향
        #   angle 음수 = 왼쪽 조향
        #
        # 실제 차량이 반대로 꺾이면 -1.0으로 바꾸면 됨.
        self.steering_sign = 1.0

        # =====================================================
        # Sector ranges
        # =====================================================
        # 오른쪽/왼쪽 범위 바꿈
        self.left_indices = list(range(0, 61))        # 0~60
        self.right_indices = list(range(300, 360))    # 300~359

        # =====================================================
        # Logging settings
        # =====================================================
        # 20Hz 기준:
        # 20 = 약 1초
        # 10 = 약 0.5초
        # 5  = 약 0.25초
        self.log_interval_count = 5

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"
        self.decision = "INIT"

        # =====================================================
        # Median values
        # =====================================================
        self.left_median = None
        self.right_median = None

        self.left_count = 0
        self.right_count = 0

        self.center_error = 0.0

        # =====================================================
        # Viewer objects
        # =====================================================
        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.lidar_points = None

        self.warned_no_lidar = False
        self.log_counter = 0

    def start(self):
        if self.show_debug:
            self.init_lidar_viewer()

        self.log_info("ConeLidarDriver started")

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"
        self.decision = "STOP"

        if self.viewer_ready:
            try:
                plt.close(self.fig)
            except Exception:
                pass

        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.lidar_points = None

        self.log_info("ConeLidarDriver stopped")

    def process(self, lidar_ranges):
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            self.clear_all_info()
            self.set_stop_state("NO_LIDAR")
            self.print_debug()
            return self.angle, self.speed

        self.warned_no_lidar = False

        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # =====================================================
        # 1. 왼쪽 / 오른쪽 중앙값 계산
        # =====================================================
        self.left_median, self.left_count = self.calculate_sector_median(
            lidar_ranges,
            self.left_indices
        )

        self.right_median, self.right_count = self.calculate_sector_median(
            lidar_ranges,
            self.right_indices
        )

        # =====================================================
        # 2. 유효값 없으면 정지
        # =====================================================
        if self.left_median is None or self.right_median is None:
            self.center_error = 0.0
            self.set_stop_state("INVALID_MEDIAN")
            self.print_debug()
            return self.angle, self.speed

        # =====================================================
        # 3. 너무 가까우면 긴급 정지
        # =====================================================
        if (
            self.left_median < self.emergency_stop_distance or
            self.right_median < self.emergency_stop_distance
        ):
            self.center_error = self.right_median - self.left_median
            self.set_stop_state("EMERGENCY_CLOSE")
            self.print_debug()
            return self.angle, self.speed

        # =====================================================
        # 4. 중앙 찾기
        # =====================================================
        # error > 0:
        #   right_median이 더 큼
        #   left가 더 가까움
        #   오른쪽으로 조향
        #
        # error < 0:
        #   left_median이 더 큼
        #   right가 더 가까움
        #   왼쪽으로 조향
        self.center_error = self.right_median - self.left_median

        if abs(self.center_error) <= self.center_deadband:
            self.set_go_state(
                decision="CENTER_GO",
                angle=0.0,
                speed=self.forward_speed
            )

        else:
            raw_angle = self.center_error * self.steering_gain
            angle = self.clamp(
                raw_angle * self.steering_sign,
                -self.max_angle,
                self.max_angle
            )

            # 차이가 크면 조금 느리게
            if abs(self.center_error) >= 1.0:
                speed = self.slow_speed
            else:
                speed = self.forward_speed

            self.set_go_state(
                decision="CENTER_STEER",
                angle=angle,
                speed=speed
            )

        self.print_debug()
        return self.angle, self.speed

    # =====================================================
    # State setters
    # =====================================================

    def set_go_state(self, decision, angle, speed):
        self.state = "GO"
        self.decision = decision
        self.angle = float(angle)
        self.speed = float(speed)

    def set_stop_state(self, decision):
        self.state = "STOP"
        self.decision = decision
        self.angle = 0.0
        self.speed = 0.0

    # =====================================================
    # LiDAR calculation
    # =====================================================

    def clear_all_info(self):
        self.left_median = None
        self.right_median = None

        self.left_count = 0
        self.right_count = 0

        self.center_error = 0.0

    def calculate_sector_median(self, lidar_ranges, indices):
        valid_values = []

        for index in indices:
            if index < 0 or index >= len(lidar_ranges):
                continue

            distance = lidar_ranges[index]

            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            valid_values.append(float(distance))

        if len(valid_values) == 0:
            return None, 0

        median_value = float(np.median(valid_values))

        return median_value, len(valid_values)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    # =====================================================
    # LiDAR viewer
    # =====================================================

    def init_lidar_viewer(self):
        if self.viewer_ready:
            return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("LiDAR Debug Viewer")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        self.lidar_points = self.ax.scatter([], [], s=5)

        # 차량 중심
        self.ax.plot(0, 0, 'ro')

        # 전방 방향 표시
        self.ax.plot([0, 0], [0, 2], 'r-')

        plt.ion()
        plt.show(block=False)

        self.viewer_ready = True
        self.log_info("LiDAR debug viewer started")

    def update_lidar_viewer(self, lidar_ranges):
        if not self.viewer_ready:
            self.init_lidar_viewer()

        valid = np.array([
            d if math.isfinite(d) else np.nan
            for d in lidar_ranges
        ], dtype=float)

        if len(valid) == 0:
            return

        angles = np.deg2rad(np.arange(len(valid)) - 90)

        x = -valid * np.cos(angles)
        y = -valid * np.sin(angles)

        indices = np.arange(len(valid))

        colors = np.full(len(valid), 'gray', dtype=object)

        # left: 0~60
        colors[(indices >= 0) & (indices <= 60)] = 'green'

        # right: 300~359
        colors[(indices >= 300) & (indices < 360)] = 'orange'

        # 정면 index 0 근처 강조
        colors[(indices >= 350) & (indices < 360)] = 'red'
        colors[(indices >= 0) & (indices <= 10)] = 'red'

        # 후방 참고
        colors[(indices >= 150) & (indices <= 210)] = 'blue'

        valid_mask = np.isfinite(x) & np.isfinite(y)

        self.lidar_points.set_offsets(
            np.c_[x[valid_mask], y[valid_mask]]
        )

        self.lidar_points.set_color(
            colors[valid_mask]
        )

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    # =====================================================
    # Debug log
    # =====================================================

    def format_distance(self, value):
        if value is None:
            return "invalid"

        return f"{value:.2f}m"

    def print_debug(self):
        self.log_counter += 1

        if self.log_counter % self.log_interval_count != 0:
            return

        self.log_info(
            f"LiDAR CENTER | "
            f"state:{self.state} | "
            f"decision:{self.decision} | "
            f"left:{self.format_distance(self.left_median)} | "
            f"right:{self.format_distance(self.right_median)} | "
            f"error:{self.center_error:.2f} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)