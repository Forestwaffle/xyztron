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

    이번 로직:
        1. 왼쪽 구간 중앙값 계산
            - index 0~59

        2. 정면 구간 중앙값 계산
            - index 60~119

        3. 오른쪽 구간 중앙값 계산
            - index 120~180

        4. INFO에 세 중앙값 출력

        5. 세 중앙값 중 하나라도 stop_median_distance 이하이면 STOP

        6. 아니면 GO

    제외 없음:
        - 특정 index 제외 없음
        - 0.12m 고정값 제외 없음
        - 0.30m 이하 제거 없음

    단, inf / nan / 0 이하 값은 중앙값 계산이 불가능하므로 계산에서만 제외.
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 5.0

        # 중앙값 정지 기준
        # 주의:
        #   오른쪽 120~180 구간에 0.12m 고정값이 많으면
        #   이 값을 너무 높게 잡으면 계속 STOP 될 수 있음.
        #
        # 처음 테스트는 0.10으로 시작.
        # 너무 안 멈추면 0.15, 0.20, 0.30 순서로 올리기.
        self.stop_median_distance = 0.10

        # =====================================================
        # Sector ranges
        # =====================================================
        # 사용자가 요청한 단순 구간:
        #   0~60, 60~120, 120~180
        #
        # 중복 방지를 위해 실제 코드에서는:
        #   left  : 0~59
        #   front : 60~119
        #   right : 120~180
        self.left_indices = list(range(0, 60))        # 0~59
        self.front_indices = list(range(60, 120))     # 60~119
        self.right_indices = list(range(120, 181))    # 120~180

        # =====================================================
        # Logging settings
        # =====================================================
        # control_loop 20Hz 기준:
        #   20 = 약 1초
        #   10 = 약 0.5초
        #   5  = 약 0.25초
        #   1  = 매 루프
        self.log_interval_count = 5

        # 각 구간에서 가장 가까운 값 몇 개 출력할지
        self.near_value_count = 5

        # 전체 raw 360개 출력 여부
        self.log_all_lidar_values = False
        self.all_value_chunk_size = 60

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
        self.left_median = None
        self.front_median = None
        self.right_median = None

        self.left_count = 0
        self.front_count = 0
        self.right_count = 0

        self.left_points = []
        self.front_points = []
        self.right_points = []

        # =====================================================
        # Global closest info
        # 전체 방향에서 가장 가까운 값.
        # 판단에는 사용하지 않고 INFO 확인용.
        # =====================================================
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

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
            self.print_debug(lidar_ranges=None)
            return self.angle, self.speed

        self.warned_no_lidar = False

        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # =====================================================
        # 1. 전체 방향 최단값 계산
        # INFO용. STOP/GO 판단에는 사용하지 않음.
        # =====================================================
        self.update_global_closest(lidar_ranges)

        # =====================================================
        # 2. 왼쪽 / 정면 / 오른쪽 중앙값 계산
        # =====================================================
        self.left_median, self.left_count, self.left_points = self.calculate_sector_median(
            lidar_ranges,
            self.left_indices
        )

        self.front_median, self.front_count, self.front_points = self.calculate_sector_median(
            lidar_ranges,
            self.front_indices
        )

        self.right_median, self.right_count, self.right_points = self.calculate_sector_median(
            lidar_ranges,
            self.right_indices
        )

        # =====================================================
        # 3. STOP / GO 판단
        # 세 중앙값 중 하나라도 기준 이하이면 STOP
        # =====================================================
        stop_reasons = []

        if self.left_median is not None and self.left_median <= self.stop_median_distance:
            stop_reasons.append("LEFT_MEDIAN_LOW")

        if self.front_median is not None and self.front_median <= self.stop_median_distance:
            stop_reasons.append("FRONT_MEDIAN_LOW")

        if self.right_median is not None and self.right_median <= self.stop_median_distance:
            stop_reasons.append("RIGHT_MEDIAN_LOW")

        if len(stop_reasons) > 0:
            self.set_stop_state("+".join(stop_reasons))
        else:
            self.set_go_state("MEDIAN_SAFE_GO")

        self.print_debug(lidar_ranges=lidar_ranges)

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
        self.left_median = None
        self.front_median = None
        self.right_median = None

        self.left_count = 0
        self.front_count = 0
        self.right_count = 0

        self.left_points = []
        self.front_points = []
        self.right_points = []

        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

    def calculate_sector_median(self, lidar_ranges, indices):
        """
        특정 구간의 중앙값 계산.

        제외 정책:
            - 특정 index 제외 없음
            - 가까운 값 제거 없음
            - 단, inf/nan/0 이하는 계산 불가능하므로 제외

        반환:
            median_value, valid_count, nearest_points
        """
        valid_values = []
        points = []

        for index in indices:
            if index < 0 or index >= len(lidar_ranges):
                continue

            distance = lidar_ranges[index]

            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            distance = float(distance)
            valid_values.append(distance)

            angle_deg = self.index_to_angle_deg(index)
            direction = self.angle_to_direction(angle_deg)

            points.append(
                {
                    "distance": distance,
                    "index": int(index),
                    "angle_deg": float(angle_deg),
                    "direction": direction
                }
            )

        if len(valid_values) == 0:
            return None, 0, []

        median_value = float(np.median(valid_values))

        points.sort(key=lambda item: item["distance"])
        nearest_points = points[:self.near_value_count]

        return median_value, len(valid_values), nearest_points

    def update_global_closest(self, lidar_ranges):
        closest = None

        for index, distance in enumerate(lidar_ranges):
            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            distance = float(distance)

            angle_deg = self.index_to_angle_deg(index)
            direction = self.angle_to_direction(angle_deg)

            point = {
                "distance": distance,
                "index": int(index),
                "angle_deg": float(angle_deg),
                "direction": direction
            }

            if closest is None:
                closest = point
            elif point["distance"] < closest["distance"]:
                closest = point

        if closest is None:
            self.closest_distance = None
            self.closest_index = None
            self.closest_angle_deg = None
            self.closest_direction = "UNKNOWN"
            return

        self.closest_distance = closest["distance"]
        self.closest_index = closest["index"]
        self.closest_angle_deg = closest["angle_deg"]
        self.closest_direction = closest["direction"]

    def index_to_angle_deg(self, index):
        """
        기준:
            index 90 = 정면 0도
            index 0 = 왼쪽 -90도
            index 180 = 오른쪽 +90도
            index 270 = 후방 180도 근처
        """
        angle = float(index) - 90.0

        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    def angle_to_direction(self, angle_deg):
        if angle_deg is None:
            return "UNKNOWN"

        if -30.0 <= angle_deg <= 30.0:
            return "FRONT"

        if 30.0 < angle_deg <= 150.0:
            return "RIGHT"

        if -150.0 <= angle_deg < -30.0:
            return "LEFT"

        return "BACK"

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

    def make_points_text(self, points):
        if len(points) == 0:
            return "none"

        texts = []

        for number, point in enumerate(points, start=1):
            text = (
                f"#{number} "
                f"i:{point['index']} "
                f"d:{point['distance']:.2f}m "
                f"a:{point['angle_deg']:.1f}deg "
                f"{point['direction']}"
            )
            texts.append(text)

        return " | ".join(texts)

    def format_distance(self, value):
        if value is None:
            return "invalid"

        return f"{value:.2f}m"

    def format_angle(self, value):
        if value is None:
            return "invalid"

        return f"{value:.2f}deg"

    def format_lidar_value(self, value):
        if value is None:
            return "None"

        if math.isnan(value):
            return "nan"

        if math.isinf(value):
            return "inf"

        return f"{value:.2f}"

    def log_all_ranges(self, lidar_ranges):
        if lidar_ranges is None:
            self.log_info("LIDAR RAW | None")
            return

        total = len(lidar_ranges)

        self.log_info(
            f"LIDAR RAW ALL START | total_beams:{total}"
        )

        start = 0

        while start < total:
            end = min(start + self.all_value_chunk_size, total)

            values = []

            for index in range(start, end):
                value_text = self.format_lidar_value(lidar_ranges[index])
                values.append(f"{index:03d}:{value_text}")

            self.log_info(
                f"LIDAR RAW {start:03d}-{end - 1:03d} | "
                + " ".join(values)
            )

            start = end

        self.log_info("LIDAR RAW ALL END")

    def print_debug(self, lidar_ranges=None):
        self.log_counter += 1

        if self.log_counter % self.log_interval_count != 0:
            return

        left_points_text = self.make_points_text(self.left_points)
        front_points_text = self.make_points_text(self.front_points)
        right_points_text = self.make_points_text(self.right_points)

        self.log_info(
            f"LIDAR STATE:{self.state} "
            f"| decision:{self.decision} "
            f"| stop_median:{self.stop_median_distance:.2f}m "
            f"| left_median:{self.format_distance(self.left_median)} "
            f"| left_count:{self.left_count} "
            f"| left_range:[0~59] "
            f"| front_median:{self.format_distance(self.front_median)} "
            f"| front_count:{self.front_count} "
            f"| front_range:[60~119] "
            f"| right_median:{self.format_distance(self.right_median)} "
            f"| right_count:{self.right_count} "
            f"| right_range:[120~180] "
            f"| global_min:{self.format_distance(self.closest_distance)} "
            f"| global_index:{self.closest_index} "
            f"| global_angle:{self.format_angle(self.closest_angle_deg)} "
            f"| global_dir:{self.closest_direction} "
            f"| left_near:[{left_points_text}] "
            f"| front_near:[{front_points_text}] "
            f"| right_near:[{right_points_text}] "
            f"| obstacle:{self.obstacle_detected} "
            f"| angle_cmd:{self.angle:.2f} "
            f"| speed_cmd:{self.speed:.2f}"
        )

        if self.log_all_lidar_values:
            self.log_all_ranges(lidar_ranges)

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)