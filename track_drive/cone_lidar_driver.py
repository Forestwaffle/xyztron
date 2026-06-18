#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        1. 기본은 직진
        2. 전 방향 중 어느 곳이든 가까운 물체가 있으면 정지
        3. 가장 가까운 곳의 각도와 거리, STOP/GO 상태를 INFO로 출력
        4. 라이다 값이 실제로 갱신되는지 scan_delta로 확인

    입력:
        /scan LaserScan message

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
        self.stop_distance = 0.6

        # 너무 작은 값은 센서 노이즈 / 자기 차체 반사일 수 있어서 무시
        self.min_valid_distance = 0.20

        # scan 변화량 판단 기준
        self.scan_change_threshold = 0.01

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

        # =====================================================
        # Scan update debug info
        # =====================================================
        self.prev_scan = None
        self.scan_delta_max = None
        self.scan_delta_mean = None
        self.changed_beams = 0
        self.valid_beams = 0

        self.callback_count = 0
        self.prev_callback_count = -1
        self.callback_changed = False

        self.scan_stamp = None
        self.prev_scan_stamp = None
        self.stamp_changed = False

        # 대표 sample 값
        self.sample_text = ""

        # 장애물 감지 여부
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

    def process(self, lidar_msg=None, callback_count=0):
        """
        기본 로직:
            - 기본은 GO
            - 전체 방향에서 가장 가까운 유효 거리 계산
            - closest_distance <= stop_distance 이면 STOP
            - 아니면 GO

        추가 진단:
            - callback_count 변화 확인
            - header stamp 변화 확인
            - scan_delta_max / changed_beams 확인
            - 여러 index sample 출력
        """
        self.callback_count = callback_count
        self.callback_changed = (
            self.callback_count != self.prev_callback_count
        )
        self.prev_callback_count = self.callback_count

        if lidar_msg is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            self.set_stop_state()
            self.clear_scan_info()
            self.print_debug()
            return self.angle, self.speed

        self.warned_no_lidar = False

        lidar_ranges = lidar_msg.ranges

        # stamp 확인
        self.scan_stamp = self.get_stamp_sec(lidar_msg)
        self.stamp_changed = (
            self.scan_stamp != self.prev_scan_stamp
        )
        self.prev_scan_stamp = self.scan_stamp

        # 원래 LiDAR Viewer 방식으로 화면 업데이트
        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # scan 변화량 확인
        self.update_scan_change_info(lidar_ranges)

        # sample 값 확인
        self.sample_text = self.make_sample_text(lidar_ranges)

        # 기본값은 GO
        self.set_go_state()

        # 가장 가까운 점 계산
        closest = self.get_closest_point_info(lidar_ranges)

        if closest is None:
            self.closest_distance = None
            self.closest_index = None
            self.closest_angle_deg = None
            self.closest_direction = "UNKNOWN"

            # 유효한 거리값이 전혀 없으면 일단 GO
            # 더 안전하게 하려면 set_stop_state()로 바꿔도 됨
            self.set_go_state()
            self.print_debug()
            return self.angle, self.speed

        (
            self.closest_distance,
            self.closest_index,
            self.closest_angle_deg,
            self.closest_direction
        ) = closest

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

    def clear_scan_info(self):
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

        self.scan_delta_max = None
        self.scan_delta_mean = None
        self.changed_beams = 0
        self.valid_beams = 0
        self.sample_text = ""

    def get_stamp_sec(self, lidar_msg):
        """
        LaserScan header stamp를 초 단위 float로 반환.
        """
        try:
            stamp = lidar_msg.header.stamp
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        except Exception:
            return None

    def update_scan_change_info(self, lidar_ranges):
        """
        이전 scan과 현재 scan의 차이를 계산.

        scan_delta_max가 계속 0.00이면
        실제 ranges 값이 거의 안 바뀌는 상태.
        """
        current = np.array([
            d if math.isfinite(d) else np.nan
            for d in lidar_ranges
        ], dtype=float)

        if self.prev_scan is None:
            self.prev_scan = current.copy()
            self.scan_delta_max = None
            self.scan_delta_mean = None
            self.changed_beams = 0
            self.valid_beams = int(np.sum(np.isfinite(current)))
            return

        if len(current) != len(self.prev_scan):
            self.prev_scan = current.copy()
            self.scan_delta_max = None
            self.scan_delta_mean = None
            self.changed_beams = 0
            self.valid_beams = int(np.sum(np.isfinite(current)))
            return

        valid_mask = np.isfinite(current) & np.isfinite(self.prev_scan)
        self.valid_beams = int(np.sum(valid_mask))

        if self.valid_beams == 0:
            self.scan_delta_max = None
            self.scan_delta_mean = None
            self.changed_beams = 0
            self.prev_scan = current.copy()
            return

        diff = np.abs(current[valid_mask] - self.prev_scan[valid_mask])

        self.scan_delta_max = float(np.max(diff))
        self.scan_delta_mean = float(np.mean(diff))
        self.changed_beams = int(np.sum(diff > self.scan_change_threshold))

        self.prev_scan = current.copy()

    def make_sample_text(self, lidar_ranges):
        """
        대표 index들의 거리값을 문자열로 생성.

        기존 기준:
            index 90 = 전방
            index 0 = 왼쪽
            index 180 = 오른쪽
            index 270 = 후방 근처
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return ""

        sample_indices = [
            0, 45, 90, 135, 180, 225, 270, 315
        ]

        texts = []

        for index in sample_indices:
            if index >= len(lidar_ranges):
                continue

            value = lidar_ranges[index]

            if not math.isfinite(value):
                texts.append(f"{index}:inf")
            else:
                texts.append(f"{index}:{value:.2f}")

        return " ".join(texts)

    def get_closest_point_info(self, lidar_ranges):
        """
        라이다 전체 방향에서 가장 가까운 거리, index, 각도, 방향 문자열 반환.

        기준:
            index 90 = 전방 0도
            index 0 = 왼쪽
            index 180 = 오른쪽
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return None

        valid_candidates = []

        for index, distance in enumerate(lidar_ranges):
            if not math.isfinite(distance):
                continue

            if distance <= self.min_valid_distance:
                continue

            valid_candidates.append((distance, index))

        if not valid_candidates:
            return None

        closest_distance, closest_index = min(
            valid_candidates,
            key=lambda item: item[0]
        )

        closest_angle_deg = self.index_to_angle_deg(closest_index)
        closest_direction = self.angle_to_direction(closest_angle_deg)

        return (
            closest_distance,
            closest_index,
            closest_angle_deg,
            closest_direction
        )

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

    def format_float(self, value, suffix=""):
        if value is None:
            return "None"

        return f"{value:.3f}{suffix}"

    def print_debug(self):
        """
        약 1초마다 터미널 INFO 출력.
        """
        self.log_counter += 1

        if self.log_counter % 20 != 0:
            return

        if self.closest_distance is None:
            self.log_info(
                f"LIDAR STATE:{self.state} "
                f"| cb:{self.callback_count} "
                f"| cb_changed:{self.callback_changed} "
                f"| stamp:{self.format_float(self.scan_stamp)} "
                f"| stamp_changed:{self.stamp_changed} "
                f"| closest:invalid "
                f"| delta_max:{self.format_float(self.scan_delta_max, 'm')} "
                f"| delta_mean:{self.format_float(self.scan_delta_mean, 'm')} "
                f"| changed_beams:{self.changed_beams}/{self.valid_beams} "
                f"| samples:[{self.sample_text}] "
                f"| speed_cmd:{self.speed:.2f}"
            )
            return

        self.log_info(
            f"LIDAR STATE:{self.state} "
            f"| cb:{self.callback_count} "
            f"| cb_changed:{self.callback_changed} "
            f"| stamp:{self.format_float(self.scan_stamp)} "
            f"| stamp_changed:{self.stamp_changed} "
            f"| closest_dist:{self.closest_distance:.2f}m "
            f"| closest_index:{self.closest_index} "
            f"| closest_angle:{self.closest_angle_deg:.2f}deg "
            f"| direction:{self.closest_direction} "
            f"| stop_distance:{self.stop_distance:.2f}m "
            f"| delta_max:{self.format_float(self.scan_delta_max, 'm')} "
            f"| delta_mean:{self.format_float(self.scan_delta_mean, 'm')} "
            f"| changed_beams:{self.changed_beams}/{self.valid_beams} "
            f"| samples:[{self.sample_text}] "
            f"| speed_cmd:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)