#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    LiDAR Debug Viewer 기준:
        index 0   = 정면
        index 90  = 오른쪽/왼쪽 중 실제 뷰어 기준 확인 필요
        index 180 = 뒤쪽
        index 270 = 반대쪽

    현재 범위:
        left  : 0~60
        right : 300~359

    상태 흐름:
        GO:
            left < 10 또는 right < 10 이면 TURN

        TURN:
            왼쪽 최대 조향 -100으로 주행
            left > 60 이면 TURN2

        TURN2:
            정지하지 않고 직진
            right < 9 이면 TURN3

        TURN3:
            왼쪽 조향 -30으로 1초 주행
            이후 FINISH_STOP

        FINISH_STOP:
            정지 유지
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 8.0
        self.turn_speed = 7.0
        self.turn2_speed = 8.0
        self.turn3_speed = 7.0

        # TURN 상태: 왼쪽 최대 조향
        # 실제 차량이 반대로 꺾이면 +100.0으로 변경
        self.left_turn_angle = -100.0

        # TURN3 상태: 마지막 1초 동안 약한 왼쪽 조향
        self.turn3_left_angle = -10.0

        # =====================================================
        # Distance thresholds
        # =====================================================
        # GO -> TURN
        self.turn_start_distance = 10.00

        # TURN -> TURN2
        self.turn_finish_left_distance = 60.00

        # TURN2 -> TURN3
        self.turn2_right_trigger_distance = 9.00

        # TURN3 duration
        self.turn3_duration_sec = 0.50

        # =====================================================
        # Sector ranges
        # =====================================================
        self.left_indices = list(range(0, 61))        # 0~60
        self.right_indices = list(range(300, 360))    # 300~359

        # =====================================================
        # Logging settings
        # =====================================================
        # 1 = 매 제어 루프마다 출력
        # 5 = 약 0.25초마다 출력, control_loop 20Hz 기준
        self.log_interval_count = 1

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        self.state = "GO"
        self.decision = "INIT"
        self.obstacle_detected = False

        # TURN3 시간 측정용
        self.turn3_start_time = None
        self.turn3_elapsed = 0.0

        # =====================================================
        # Median values
        # =====================================================
        self.left_median = None
        self.right_median = None

        self.left_count = 0
        self.right_count = 0

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
        self.state = "FINISH_STOP"
        self.decision = "STOP"
        self.obstacle_detected = True

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
            self.set_finish_stop_state("NO_LIDAR")
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
        # 2. 상태 머신
        # =====================================================

        # -----------------------------------------------------
        # GO
        # left 또는 right가 10m 밑이면 TURN
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # TURN
        # 왼쪽 최대 조향 -100
        # left > 60 이면 TURN2
        # -----------------------------------------------------
        elif self.state == "TURN":
            if self.left_median is not None and self.left_median > self.turn_finish_left_distance:
                self.set_turn2_state("LEFT_OVER_60_GO_STRAIGHT")
            else:
                self.set_turn_state("TURN_LEFT_MAX")

        # -----------------------------------------------------
        # TURN2
        # 정지가 아니라 직진
        # right < 9 이면 TURN3
        # -----------------------------------------------------
        elif self.state == "TURN2":
            if self.right_median is not None and self.right_median < self.turn2_right_trigger_distance:
                self.set_turn3_state("RIGHT_UNDER_9_TURN_LEFT_1SEC")
            else:
                self.set_turn2_state("TURN2_STRAIGHT")

        # -----------------------------------------------------
        # TURN3
        # -30으로 1초 주행 후 정지
        # -----------------------------------------------------
        elif self.state == "TURN3":
            now = time.monotonic()

            if self.turn3_start_time is None:
                self.turn3_start_time = now

            self.turn3_elapsed = now - self.turn3_start_time

            if self.turn3_elapsed >= self.turn3_duration_sec:
                self.set_finish_stop_state("TURN3_1SEC_DONE_STOP")
            else:
                self.set_turn3_state("TURN3_LEFT_MINUS_30_1SEC")

        # -----------------------------------------------------
        # FINISH_STOP
        # 정지 유지
        # -----------------------------------------------------
        elif self.state == "FINISH_STOP":
            self.set_finish_stop_state("FINISH_STOP_KEEP")

        # -----------------------------------------------------
        # Unknown state
        # -----------------------------------------------------
        else:
            self.set_finish_stop_state("UNKNOWN_STATE_STOP")

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

    def set_turn2_state(self, decision):
        self.state = "TURN2"
        self.decision = decision
        self.obstacle_detected = False
        self.angle = 0.0
        self.speed = self.turn2_speed

    def set_turn3_state(self, decision):
        if self.state != "TURN3":
            self.turn3_start_time = time.monotonic()
            self.turn3_elapsed = 0.0

        self.state = "TURN3"
        self.decision = decision
        self.obstacle_detected = True

        # 마지막 TURN3는 -30으로 1초 주행
        self.angle = self.turn3_left_angle
        self.speed = self.turn3_speed

    def set_finish_stop_state(self, decision):
        self.state = "FINISH_STOP"
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

        # 뒤쪽 참고
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
            f"LiDAR STATE | "
            f"state:{self.state} | "
            f"decision:{self.decision} | "
            f"turn_start_under:{self.turn_start_distance:.2f}m | "
            f"turn_finish_left_over:{self.turn_finish_left_distance:.2f}m | "
            f"turn2_right_under:{self.turn2_right_trigger_distance:.2f}m | "
            f"turn3_elapsed:{self.turn3_elapsed:.2f}s | "
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