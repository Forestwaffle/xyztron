#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        1. 일단 직진
        2. 전 방향 중 어느 곳이든 가까운 물체가 있으면 정지
        3. 가장 가까운 곳의 각도와 거리, STOP/GO 상태를 INFO로 출력

    입력:
        /scan LaserScan.ranges

    처리:
        라이다 전체 ranges 확인
        유효한 거리값 중 최소 거리 계산
        최소 거리의 index와 angle 계산
        최소 거리가 stop_distance 이하이면 STOP
        아니면 GO

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
        # Simple drive parameters
        # =====================================================
        # 기본 직진 속도
        self.forward_speed = 5.0

        # 전 방향에서 이 거리보다 가까운 물체가 있으면 정지
        # 너무 자주 멈추면 0.4~0.5
        # 너무 늦게 멈추면 0.8~1.2
        self.stop_distance = 0.6

        # 너무 작은 값은 센서 노이즈/자기 자신 반사 가능성이 있으므로 무시
        self.min_valid_distance = 0.05

        # 현재 출력값
        self.angle = 0.0
        self.speed = 0.0

        # 현재 상태
        self.state = "STOP"

        # 가장 가까운 점 정보
        self.closest_distance = None
        self.closest_index = None
        self.closest_angle_deg = None
        self.closest_direction = "UNKNOWN"

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

    def process(self, lidar_ranges):
        """
        목표:
            1. 일단 직진
            2. 전 방향 중 어느 곳이든 가까운 물체가 있으면 정지
            3. 가장 가까운 곳의 각도와 거리, STOP/GO 상태를 INFO로 출력

        출력:
            angle = 0.0
            speed = 5.0 또는 0.0
        """
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            # LiDAR가 아직 안 들어오면 안전 정지
            self.angle = 0.0
            self.speed = 0.0
            self.state = "STOP"
            self.obstacle_detected = True

            self.closest_distance = None
            self.closest_index = None
            self.closest_angle_deg = None
            self.closest_direction = "UNKNOWN"

            self.print_debug()
            return self.angle, self.speed

        self.warned_no_lidar = False

        # 원래 LiDAR Viewer 방식으로 화면 업데이트
        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # 기본값은 GO 직진
        self.angle = 0.0
        self.speed = self.forward_speed
        self.state = "GO"
        self.obstacle_detected = False

        # 전 방향에서 가장 가까운 점 계산
        closest = self.get_closest_point_info(lidar_ranges)

        if closest is None:
            # 유효한 라이다 값이 없으면 일단 GO
            # 완전 안전 정지를 원하면 여기 speed를 0.0으로 바꾸면 됨
            self.closest_distance = None
            self.closest_index = None
            self.closest_angle_deg = None
            self.closest_direction = "UNKNOWN"

            self.state = "GO"
            self.obstacle_detected = False
            self.angle = 0.0
            self.speed = self.forward_speed

            self.print_debug()
            return self.angle, self.speed

        (
            self.closest_distance,
            self.closest_index,
            self.closest_angle_deg,
            self.closest_direction
        ) = closest

        # 가까운 물체가 있으면 STOP
        if self.closest_distance <= self.stop_distance:
            self.state = "STOP"
            self.obstacle_detected = True
            self.angle = 0.0
            self.speed = 0.0
        else:
            self.state = "GO"
            self.obstacle_detected = False
            self.angle = 0.0
            self.speed = self.forward_speed

        self.print_debug()

        return self.angle, self.speed

    def get_closest_point_info(self, lidar_ranges):
        """
        라이다 전체 방향에서 가장 가까운 거리, index, 각도, 방향 문자열 반환.

        angle 기준:
            0도     = 전방
            음수    = 왼쪽
            양수    = 오른쪽
            ±180도  = 후방 근처

        기존 viewer 기준:
            index 90 = 전방
            index 0  = 왼쪽
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

        기존 사용 기준:
            index 90 = 전방 0도

        반환:
            -180도 ~ +180도 범위
            음수: 왼쪽
            양수: 오른쪽
        """
        angle = float(index) - 90.0

        # -180 ~ +180 범위로 정규화
        while angle > 180.0:
            angle -= 360.0

        while angle < -180.0:
            angle += 360.0

        return angle

    def angle_to_direction(self, angle_deg):
        """
        각도를 사람이 보기 쉬운 방향 문자열로 변환.
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

        형태:
            - 고정 스케일 -10~10
            - scatter 점 표시
            - 차량 중심 빨간 점
            - 전방 빨간 선
        """
        if self.viewer_ready:
            return

        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.ax.set_title("LiDAR Debug Viewer")
        self.ax.set_aspect('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)

        # 원래 방식: scatter 하나만 사용
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

        # 기존 viewer 방식:
        # index 0 = left
        # index 90 = front
        angles = np.deg2rad(np.arange(len(valid)) - 90)

        x = -valid * np.cos(angles)
        y = -valid * np.sin(angles)

        indices = np.arange(len(valid))
        colors = np.full(len(valid), 'b', dtype=object)

        # 원래 색상 구간
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

    def print_debug(self):
        """
        약 1초마다 터미널 INFO 출력.

        출력 예시:
            LIDAR STATE:GO | closest_dist:1.25m | closest_angle:-42.00deg | direction:LEFT
            LIDAR STATE:STOP | closest_dist:0.42m | closest_angle:18.00deg | direction:FRONT
        """
        self.log_counter += 1

        # process()가 20Hz이므로 20번마다 1회 출력
        if self.log_counter % 20 != 0:
            return

        if self.closest_distance is None:
            self.log_info(
                f"LIDAR STATE:{self.state} "
                f"| closest:invalid "
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