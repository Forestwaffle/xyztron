#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    LiDAR index 기준:
        index 0   = 왼쪽
        index 90  = 정면
        index 180 = 오른쪽

    판단 범위:
        left  : 0~59
        front : 60~119
        right : 120~180

    현재 로직:
        1. 왼쪽 / 정면 / 오른쪽 중앙값 계산
        2. 왼쪽, 오른쪽은 INFO 출력만 함
        3. 정면 중앙값이 front_stop_distance 이하이면 STOP
        4. 정면 중앙값이 front_stop_distance보다 크면 GO

    정지 조건:
        front_median <= 0.80m 이면 STOP
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 5.0

        # 정면 중앙값 정지 기준
        self.front_stop_distance = 0.80

        # =====================================================
        # Sector ranges
        # =====================================================
        self.left_indices = list(range(0, 60))        # 0~59
        self.front_indices = list(range(60, 120))     # 60~119
        self.right_indices = list(range(120, 181))    # 120~180

        # =====================================================
        # Logging settings
        # =====================================================
        # control_loop 20Hz 기준:
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
        self.front_median = None
        self.right_median = None

        self.left_count = 0
        self.front_count = 0
        self.right_count = 0

        # =====================================================
        # Viewer objects
        # =====================================================
        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.lidar_points = None

        # Logging
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
        # 1. 왼쪽 / 정면 / 오른쪽 중앙값 계산
        # =====================================================
        self.left_median, self.left_count = self.calculate_sector_median(
            lidar_ranges,
            self.left_indices
        )

        self.front_median, self.front_count = self.calculate_sector_median(
            lidar_ranges,
            self.front_indices
        )

        self.right_median, self.right_count = self.calculate_sector_median(
            lidar_ranges,
            self.right_indices
        )

        # =====================================================
        # 2. STOP / GO 판단
        # 정면 중앙값만 판단에 사용
        # =====================================================
        if self.front_median is None:
            self.set_stop_state("FRONT_INVALID")

        elif self.front_median <= self.front_stop_distance:
            self.set_stop_state("FRONT_STOP")

        else:
            self.set_go_state("FRONT_SAFE_GO")

        self.print_debug()

        return self.angle, self.speed

    # =====================================================
    # State setters
    # =====================================================

    def set_go_state(self, decision):
        self.state = "GO"
        self.decision = decision
        self.angle = 0.0
        self.speed = self.forward_speed

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
        self.front_median = None
        self.right_median = None

        self.left_count = 0
        self.front_count = 0
        self.right_count = 0

    def calculate_sector_median(self, lidar_ranges, indices):
        """
        특정 구간의 중앙값 계산.

        제외 없음:
            - 특정 index 제외 없음
            - 0.12m 고정값 제외 없음
            - 0.30m 이하 제거 없음

        단, inf / nan / 0 이하 값은 중앙값 계산이 불가능하므로 제외.
        """
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
        colors = np.full(len(valid), 'b', dtype=object)

        colors[(indices >= 0) & (indices < 45)] = 'r'
        colors[(indices >= 45) & (indices < 90)] = 'g'
        colors[(indices >= 90) & (indices < 270)] = 'b'
        colors[(indices >= 270) & (indices < 315)] = 'orange'
        colors[(indices >= 315) & (indices < 360)] = 'purple'

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
            f"LiDAR MEDIAN | "
            f"state:{self.state} | "
            f"front:{self.format_distance(self.front_median)} | "
            f"right:{self.format_distance(self.right_median)} | "
            f"left:{self.format_distance(self.left_median)} | "
            f"front_stop:{self.front_stop_distance:.2f}m"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)