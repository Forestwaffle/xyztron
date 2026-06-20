#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np


class ConeLidarDriver:
    """
    LiDAR Debug Viewer 기준:
        index 0   = 정면
        index 90  = 오른쪽/왼쪽 중 실제 뷰어 기준 확인 필요
        index 180 = 뒤쪽
        index 270 = 반대쪽

    현재 로직:
        left  : 0~60
        right : 300~359

    상태:
        GO:
            left 또는 right 중앙값이 10m 밑이면 TURN으로 전환

        TURN:
            왼쪽 최대 조향으로 전진
            left 중앙값이 53m 넘으면 FORWARD로 전환

        FORWARD:
            정지하지 않고 계속 직진

    LiDAR debug viewer:
        사용하지 않음
    """

    def __init__(self, logger=None, show_debug=False):
        self.logger = logger

        # 디버그 창 강제 비활성화
        self.show_debug = False

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 8.0
        self.turn_speed = 7.0

        # 왼쪽 최대 조향
        # 실제 차량이 반대로 꺾이면 +100.0으로 변경
        self.left_turn_angle = -100.0

        # =====================================================
        # Distance thresholds
        # =====================================================
        # GO -> TURN 전환 기준
        self.turn_start_distance = 10.00

        # TURN -> FORWARD 전환 기준
        self.turn_finish_left_distance = 53.00

        # =====================================================
        # Sector ranges
        # =====================================================
        self.left_indices = list(range(0, 61))        # 0~60
        self.right_indices = list(range(300, 360))    # 300~359

        # =====================================================
        # Logging settings
        # =====================================================
        self.log_interval_count = 1

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        self.state = "GO"
        self.decision = "INIT"
        self.obstacle_detected = False

        # =====================================================
        # Median values
        # =====================================================
        self.left_median = None
        self.right_median = None

        self.left_count = 0
        self.right_count = 0

        self.warned_no_lidar = False
        self.log_counter = 0

    def start(self):
        self.log_info("ConeLidarDriver started without LiDAR debug viewer")

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"
        self.decision = "STOP"
        self.obstacle_detected = True

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
        # 2. 상태 머신
        # =====================================================

        if self.state == "GO":
            turn_reasons = []

            if self.left_median is not None and self.left_median < self.turn_start_distance:
                turn_reasons.append("LEFT")

            if self.right_median is not None and self.right_median < self.turn_start_distance:
                turn_reasons.append("RIGHT")

            if len(turn_reasons) > 0:
                self.set_turn_state("TURN_BY_" + "_".join(turn_reasons))
            else:
                self.set_go_state("GO_STRAIGHT")

        elif self.state == "TURN":
            if self.left_median is not None and self.left_median > self.turn_finish_left_distance:
                self.set_forward_state("LEFT_OVER_53_FORWARD")
            else:
                self.set_turn_state("TURN_LEFT_MAX")

        elif self.state == "FORWARD":
            self.set_forward_state("FORWARD_KEEP")

        else:
            self.set_stop_state("UNKNOWN_STATE_STOP")

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

    def set_turn_state(self, decision):
        self.state = "TURN"
        self.decision = decision
        self.obstacle_detected = True
        self.angle = self.left_turn_angle
        self.speed = self.turn_speed

    def set_forward_state(self, decision):
        self.state = "FORWARD"
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
        self.left_median = None
        self.right_median = None

        self.left_count = 0
        self.right_count = 0

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
            f"LiDAR TURN | "
            f"state:{self.state} | "
            f"decision:{self.decision} | "
            f"start_under:{self.turn_start_distance:.2f}m | "
            f"finish_left_over:{self.turn_finish_left_distance:.2f}m | "
            f"left:{self.format_distance(self.left_median)} | "
            f"right:{self.format_distance(self.right_median)} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)