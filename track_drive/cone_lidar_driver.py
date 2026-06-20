#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    LiDAR Debug Viewer 기준 방향:
        index 0   = 정면
        index 90  = 오른쪽
        index 180 = 뒤쪽
        index 270 = 왼쪽

    판단 범위:
        front : 330~359 + 0~30
        right : 60~120
        left  : 240~300

    로직:
        1. 정면 / 오른쪽 / 왼쪽 중앙값 계산
        2. 셋 중 하나라도 1.00m보다 작으면 STOP
        3. 아니면 GO
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        self.forward_speed = 5.0

        # =====================================================
        # Stop distance
        # 셋 중 하나라도 1m보다 작으면 정지
        # =====================================================
        self.front_stop_distance = 1.00
        self.right_stop_distance = 1.00
        self.left_stop_distance = 1.00

        # =====================================================
        # Sector ranges
        # =====================================================
        self.front_indices = (
            list(range(330, 360)) + list(range(0, 31))
        )                                                   # 330~359 + 0~30

        self.right_indices = list(range(60, 121))            # 60~120

        self.left_indices = list(range(240, 301))            # 240~300

        # =====================================================
        # Logging settings
        # =====================================================
        self.log_interval_count = 5

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"
        self.decision = "INIT"
        self.obstacle_detected = False

        # =====================================================
        # Median values
        # =====================================================
        self.front_median = None
        self.right_median = None
        self.left_median = None

        self.front_count = 0
        self.right_count = 0
        self.left_count = 0

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
        # Median calculation
        # =====================================================
        self.front_median, self.front_count = self.calculate_sector_median(
            lidar_ranges,
            self.front_indices
        )

        self.right_median, self.right_count = self.calculate_sector_median(
            lidar_ranges,
            self.right_indices
        )

        self.left_median, self.left_count = self.calculate_sector_median(
            lidar_ranges,
            self.left_indices
        )

        # =====================================================
        # STOP / GO
        # 1m보다 작으면 정지
        # =====================================================
        stop_reasons = []

        if self.front_median is not None and self.front_median < self.front_stop_distance:
            stop_reasons.append("FRONT")

        if self.right_median is not None and self.right_median < self.right_stop_distance:
            stop_reasons.append("RIGHT")

        if self.left_median is not None and self.left_median < self.left_stop_distance:
            stop_reasons.append("LEFT")

        if len(stop_reasons) > 0:
            self.set_stop_state("STOP_BY_" + "_".join(stop_reasons))
        else:
            self.set_go_state("GO")

        self.print_debug()

        return self.angle, self.speed

    # =====================================================
    # State setters
    # =====================================================

    def set_go_state(self, decision):
        self.state = "GO"
        self.decision = decision
        self.obstacle_detected = False
        self.angle = 0.0
        self.speed = self.forward_speed

    def set_stop_state(self, decision):
        self.state = "STOP"
        self.decision = decision
        self.obstacle_detected = True
        self.angle = 0.0
        self.speed = 0.0

    # =====================================================
    # LiDAR calculation
    # =====================================================

    def clear_all_info(self):
        self.front_median = None
        self.right_median = None
        self.left_median = None

        self.front_count = 0
        self.right_count = 0
        self.left_count = 0

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

        # 정면: 빨강
        colors[(indices >= 330) & (indices < 360)] = 'red'
        colors[(indices >= 0) & (indices <= 30)] = 'red'

        # 오른쪽: 주황
        colors[(indices >= 60) & (indices <= 120)] = 'orange'

        # 왼쪽: 초록
        colors[(indices >= 240) & (indices <= 300)] = 'green'

        # 뒤쪽: 파랑
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
            f"LiDAR MEDIAN | "
            f"state:{self.state} | "
            f"front:{self.format_distance(self.front_median)} | "
            f"right:{self.format_distance(self.right_median)} | "
            f"left:{self.format_distance(self.left_median)}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)