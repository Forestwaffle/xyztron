#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        1. 기본은 직진
        2. 전방에 벽/장애물이 가까워지면 정지
        3. 전체 방향에서 가장 가까운 값도 INFO에 표시
        4. STOP/GO 판단은 전방 구간만 사용
        5. 전체 라이다 360개 값을 모두 INFO 로그로 출력

    입력:
        /scan LaserScan.ranges

    출력:
        angle:
            기본 0.0

        speed:
            GO: 5.0
            STOP: 0.0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Drive parameters
        # =====================================================
        self.forward_speed = 5.0

        # 전방 장애물 정지 거리
        self.front_stop_distance = 0.80

        # =====================================================
        # Front sector
        # =====================================================
        # 기준:
        #   index 90 = 전방
        #
        # 전방 벽 판단용 범위
        self.front_index_min = 70
        self.front_index_max = 110

        # =====================================================
        # INFO 출력 설정
        # =====================================================
        # control_loop가 20Hz라면:
        #   20 = 약 1초
        #   10 = 약 0.5초
        #   5  = 약 0.25초
        #   1  = 거의 매 루프
        self.log_interval_count = 5

        # 전체 라이다 360개 값을 모두 로그에 찍을지 여부
        self.log_all_lidar_values = True

        # 360개 값을 한 줄에 모두 찍으면 너무 길어서 여러 줄로 나눔
        self.all_value_chunk_size = 60

        # 전체 방향 가까운 값 표시 개수
        self.near_value_count = 5

        # 전방 가까운 값 표시 개수
        self.front_value_count = 5

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"
        self.obstacle_detected = False

        # =====================================================
        # Global closest info
        # 전체 방향에서 가장 가까운 값
        # 판단에는 사용하지 않고 INFO용으로만 사용
        # =====================================================
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

        self.near_points = []

        # =====================================================
        # Front closest info
        # 실제 STOP/GO 판단에 사용
        # =====================================================
        self.front_distance = None
        self.front_index = None
        self.front_angle_deg = None
        self.front_direction = "UNKNOWN"

        self.front_points = []

        # =====================================================
        # Original LiDAR viewer objects
        # =====================================================
        self.viewer_ready = False
        self.fig = None
        self.ax = None
        self.lidar_points = None

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
        self.state = "STOP"

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
        """
        기본 로직:
            - 전체 방향 가까운 값은 INFO용으로 계산
            - STOP/GO 판단은 전방 구간만 사용
            - front_distance <= front_stop_distance 이면 STOP
            - 아니면 GO
        """
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            self.set_stop_state()
            self.clear_all_info()
            self.print_debug(lidar_ranges=None)
            return self.angle, self.speed

        self.warned_no_lidar = False

        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # =====================================================
        # 1. 전체 방향 가까운 값 계산
        # 판단용이 아니라 INFO 확인용
        # =====================================================
        self.near_points = self.get_near_points(
            lidar_ranges,
            count=self.near_value_count,
            index_min=None,
            index_max=None
        )

        if len(self.near_points) > 0:
            closest = self.near_points[0]

            self.closest_distance = closest["distance"]
            self.closest_index = closest["index"]
            self.closest_angle_deg = closest["angle_deg"]
            self.closest_direction = closest["direction"]
        else:
            self.closest_distance = None
            self.closest_index = None
            self.closest_angle_deg = None
            self.closest_direction = "UNKNOWN"

        # =====================================================
        # 2. 전방 구간 가까운 값 계산
        # 실제 STOP/GO 판단용
        # =====================================================
        self.front_points = self.get_near_points(
            lidar_ranges,
            count=self.front_value_count,
            index_min=self.front_index_min,
            index_max=self.front_index_max
        )

        if len(self.front_points) > 0:
            front = self.front_points[0]

            self.front_distance = front["distance"]
            self.front_index = front["index"]
            self.front_angle_deg = front["angle_deg"]
            self.front_direction = front["direction"]
        else:
            self.front_distance = None
            self.front_index = None
            self.front_angle_deg = None
            self.front_direction = "UNKNOWN"

        # =====================================================
        # 3. STOP/GO 판단
        # 전방 값만 사용
        # =====================================================
        if self.front_distance is None:
            # 전방 유효값이 없으면 감지 물체 없음으로 보고 GO
            self.set_go_state()

        elif self.front_distance <= self.front_stop_distance:
            self.set_stop_state()

        else:
            self.set_go_state()

        self.print_debug(lidar_ranges=lidar_ranges)

        return self.angle, self.speed

    def set_go_state(self):
        self.state = "GO"
        self.obstacle_detected = False
        self.angle = 0.0
        self.speed = self.forward_speed

    def set_stop_state(self):
        self.state = "STOP"
        self.obstacle_detected = True
        self.angle = 0.0
        self.speed = 0.0

    def clear_all_info(self):
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"
        self.near_points = []

        self.front_distance = None
        self.front_index = None
        self.front_angle_deg = None
        self.front_direction = "UNKNOWN"
        self.front_points = []

    def get_near_points(self, lidar_ranges, count=5, index_min=None, index_max=None):
        """
        라이다 값 중 가까운 값 여러 개 반환.

        index_min, index_max:
            None이면 전체 방향
            값이 있으면 해당 index 구간만 검사

        반환 형식:
            [
                {
                    "distance": 0.42,
                    "index": 90,
                    "angle_deg": 0.0,
                    "direction": "FRONT"
                },
                ...
            ]
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return []

        valid_candidates = []

        start = 0
        end = len(lidar_ranges) - 1

        if index_min is not None:
            start = max(0, int(index_min))

        if index_max is not None:
            end = min(len(lidar_ranges) - 1, int(index_max))

        for index in range(start, end + 1):
            distance = lidar_ranges[index]

            if not math.isfinite(distance):
                continue

            if distance <= 0.0:
                continue

            angle_deg = self.index_to_angle_deg(index)
            direction = self.angle_to_direction(angle_deg)

            valid_candidates.append(
                {
                    "distance": float(distance),
                    "index": int(index),
                    "angle_deg": float(angle_deg),
                    "direction": direction
                }
            )

        if len(valid_candidates) == 0:
            return []

        valid_candidates.sort(key=lambda item: item["distance"])

        return valid_candidates[:count]

    def index_to_angle_deg(self, index):
        """
        라이다 index를 차량 기준 각도로 변환.

        기준:
            index 90 = 전방 0도
            index 0 = 왼쪽 -90도
            index 180 = 오른쪽 +90도
            index 270 = 후방 180도 근처

        반환:
            -180도 ~ +180도
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

    # =====================================================
    # Original LiDAR Viewer
    # =====================================================

    def init_lidar_viewer(self):
        """
        원래 LiDAR Debug Viewer 형태로 생성.
        """
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
        """
        기존 lidar_viewer.py 방식으로 LiDAR Viewer 업데이트.
        """
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
        """
        INFO에 표시할 가까운 값 문자열 생성.
        """
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
        """
        라이다 raw value를 로그용 문자열로 변환.
        """
        if value is None:
            return "None"

        if math.isnan(value):
            return "nan"

        if math.isinf(value):
            return "inf"

        return f"{value:.2f}"

    def log_all_ranges(self, lidar_ranges):
        """
        전체 라이다 값을 index:value 형태로 모두 INFO 출력.

        예:
            LIDAR RAW 000-059 | 000:7.60 001:7.61 ...
            LIDAR RAW 060-119 | 060:3.12 061:3.10 ...
        """
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
        """
        터미널 INFO 출력.

        출력 내용:
            - STOP/GO 상태
            - 전방 판단값
            - 전체 방향 최단값
            - 전방 가까운 값 목록
            - 전체 가까운 값 목록
            - 전체 라이다 360개 raw 값
        """
        self.log_counter += 1

        if self.log_counter % self.log_interval_count != 0:
            return

        near_values_text = self.make_points_text(self.near_points)
        front_values_text = self.make_points_text(self.front_points)

        front_range_text = f"[{self.front_index_min}~{self.front_index_max}]"

        self.log_info(
            f"LIDAR STATE:{self.state} "
            f"| decision:FRONT_ONLY "
            f"| front_min:{self.format_distance(self.front_distance)} "
            f"| front_index:{self.front_index} "
            f"| front_angle:{self.format_angle(self.front_angle_deg)} "
            f"| front_dir:{self.front_direction} "
            f"| front_stop:{self.front_stop_distance:.2f}m "
            f"| front_range:{front_range_text} "
            f"| global_min:{self.format_distance(self.closest_distance)} "
            f"| global_index:{self.closest_index} "
            f"| global_angle:{self.format_angle(self.closest_angle_deg)} "
            f"| global_dir:{self.closest_direction} "
            f"| front_values:[{front_values_text}] "
            f"| global_near_values:[{near_values_text}] "
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