#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        1. 기본은 직진
        2. 전 방향 중 가장 가까운 물체를 찾음
        3. 가장 가까운 거리가 stop_distance 이하이면 STOP
        4. 아니면 GO
        5. 가장 가까운 곳의 각도와 거리, STOP/GO 상태를 INFO로 출력
        6. 가까운 값 여러 개를 INFO에 같이 출력

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

        # 전 방향에서 이 거리보다 가까운 물체가 있으면 STOP
        self.stop_distance = 0.60

        # INFO 출력 주기
        # control_loop가 20Hz이므로:
        #   20 = 약 1초마다 출력
        #   10 = 약 0.5초마다 출력
        #   5  = 약 0.25초마다 출력
        self.log_interval_count = 5

        # INFO에 표시할 가까운 값 개수
        self.near_value_count = 5

        # =====================================================
        # Current output
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0
        self.state = "STOP"

        # =====================================================
        # Closest point info
        # =====================================================
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

        # 가까운 값 여러 개
        self.near_points = []

        self.obstacle_detected = False

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
            - 기본은 GO
            - 라이다 전체 방향에서 가장 가까운 거리 계산
            - closest_distance <= stop_distance 이면 STOP
            - 아니면 GO
        """
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            self.set_stop_state()
            self.clear_closest_info()
            self.print_debug()
            return self.angle, self.speed

        self.warned_no_lidar = False

        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        self.near_points = self.get_near_points(
            lidar_ranges,
            count=self.near_value_count
        )

        if len(self.near_points) == 0:
            self.clear_closest_info()

            # 라이다 값이 없으면 안전 정지
            self.set_stop_state()
            self.print_debug()
            return self.angle, self.speed

        closest = self.near_points[0]

        self.closest_distance = closest["distance"]
        self.closest_index = closest["index"]
        self.closest_angle_deg = closest["angle_deg"]
        self.closest_direction = closest["direction"]

        if self.closest_distance <= self.stop_distance:
            self.set_stop_state()
        else:
            self.set_go_state()

        self.print_debug()

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

    def clear_closest_info(self):
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"
        self.near_points = []

    def get_near_points(self, lidar_ranges, count=5):
        """
        라이다 전체 방향에서 가까운 값 여러 개 반환.

        반환 형식:
            [
                {
                    "distance": 0.42,
                    "index": 180,
                    "angle_deg": 90.0,
                    "direction": "RIGHT"
                },
                ...
            ]
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return []

        valid_candidates = []

        for index, distance in enumerate(lidar_ranges):
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

        index 90 = 전방 0도

        반환:
            -180도 ~ +180도
            음수: 왼쪽
            양수: 오른쪽
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

    def make_near_points_text(self):
        """
        INFO에 표시할 가까운 값 문자열 생성.
        """
        if len(self.near_points) == 0:
            return "none"

        texts = []

        for number, point in enumerate(self.near_points, start=1):
            text = (
                f"#{number} "
                f"i:{point['index']} "
                f"d:{point['distance']:.2f}m "
                f"a:{point['angle_deg']:.1f}deg "
                f"{point['direction']}"
            )

            texts.append(text)

        return " | ".join(texts)

    def print_debug(self):
        """
        터미널 INFO 출력.

        기존보다 자주 출력:
            log_interval_count = 5
            20Hz 기준 약 0.25초마다 출력
        """
        self.log_counter += 1

        if self.log_counter % self.log_interval_count != 0:
            return

        near_values_text = self.make_near_points_text()

        if self.closest_distance is None:
            self.log_info(
                f"LIDAR STATE:{self.state} "
                f"| closest:invalid "
                f"| near_values:[{near_values_text}] "
                f"| obstacle:{self.obstacle_detected} "
                f"| angle_cmd:{self.angle:.2f} "
                f"| speed_cmd:{self.speed:.2f}"
            )
            return

        self.log_info(
            f"LIDAR STATE:{self.state} "
            f"| closest_dist:{self.closest_distance:.2f}m "
            f"| closest_index:{self.closest_index} "
            f"| closest_angle:{self.closest_angle_deg:.2f}deg "
            f"| direction:{self.closest_direction} "
            f"| stop_distance:{self.stop_distance:.2f}m "
            f"| near_values:[{near_values_text}] "
            f"| obstacle:{self.obstacle_detected} "
            f"| angle_cmd:{self.angle:.2f} "
            f"| speed_cmd:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)