#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt


class ConeLidarDriver:
    """
    목표:
        1. 일단 직진
        2. 어느 이상 가까운 곳이 있으면 정지

    입력:
        /scan LaserScan.ranges

    처리:
        전방 일정 각도 범위 안의 라이다 거리 확인
        stop_distance 이하의 물체가 있으면 정지
        가까운 물체가 없으면 직진

    출력:
        angle:
            기본 0.0

        speed:
            장애물 없음: 5.0
            장애물 가까움: 0.0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        # =====================================================
        # Simple stop logic parameters
        # =====================================================
        # 기본 직진 속도
        self.forward_speed = 5.0

        # 이 거리보다 가까운 물체가 있으면 정지
        self.stop_distance = 1.0

        # 전방 검사 범위
        # 기존 viewer 기준:
        #   index 90 = 전방
        # 따라서 60~120이면 전방 약 ±30도
        self.front_index_min = 60
        self.front_index_max = 120

        # 현재 출력값
        self.angle = 0.0
        self.speed = 0.0

        # 현재 전방 최소 거리
        self.front_min_distance = None
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
            2. 어느 이상 가까운 곳이 있으면 정지

        입력:
            /scan LaserScan.ranges

        처리:
            전방 index 60~120 범위의 유효 거리값만 확인
            최소 거리가 stop_distance 이하이면 정지
            아니면 직진

        출력:
            angle = 0.0
            speed = 5.0 또는 0.0
        """
        if lidar_ranges is None:
            if not self.warned_no_lidar:
                self.log_warn("No LiDAR data yet")
                self.warned_no_lidar = True

            # 라이다 데이터가 없으면 안전 정지
            self.angle = 0.0
            self.speed = 0.0
            return self.angle, self.speed

        self.warned_no_lidar = False

        # 원래 LiDAR Viewer 방식으로 화면 업데이트
        if self.show_debug:
            self.update_lidar_viewer(lidar_ranges)

        # 전방 가까운 물체 확인
        self.front_min_distance = self.get_front_min_distance(lidar_ranges)

        if self.front_min_distance is None:
            # 전방 거리값이 전부 invalid면 안전 정지
            self.obstacle_detected = False
            self.angle = 0.0
            self.speed = 0.0
        elif self.front_min_distance <= self.stop_distance:
            # 가까운 물체가 있으면 정지
            self.obstacle_detected = True
            self.angle = 0.0
            self.speed = 0.0
        else:
            # 가까운 물체가 없으면 직진
            self.obstacle_detected = False
            self.angle = 0.0
            self.speed = self.forward_speed

        self.print_debug()

        return self.angle, self.speed

    def get_front_min_distance(self, lidar_ranges):
        """
        전방 범위 안에서 가장 가까운 거리 반환.

        기존 좌표 가정:
            index 90 = 전방

        검사 범위:
            index 60~120
        """
        if lidar_ranges is None or len(lidar_ranges) == 0:
            return None

        max_index = len(lidar_ranges) - 1

        start = max(0, self.front_index_min)
        end = min(max_index, self.front_index_max)

        front_candidates = [
            d for d in lidar_ranges[start:end + 1]
            if math.isfinite(d) and d > 0.0
        ]

        if not front_candidates:
            return None

        return min(front_candidates)

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
        약 1초마다 터미널 디버깅 출력.
        """
        self.log_counter += 1

        if self.log_counter % 20 != 0:
            return

        if self.front_min_distance is None:
            self.log_warn(
                "CONE LOG | front:invalid "
                f"| obstacle:{self.obstacle_detected} "
                f"| angle:{self.angle:.2f} speed:{self.speed:.2f}"
            )
            return

        self.log_info(
            f"CONE LOG | front_min:{self.front_min_distance:.2f} m "
            f"| stop_distance:{self.stop_distance:.2f} m "
            f"| obstacle:{self.obstacle_detected} "
            f"| angle:{self.angle:.2f} speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)