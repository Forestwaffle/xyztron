#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import math
import cv2
import numpy as np


class YellowLineFollower:
    """
    노란색 중앙선을 따라가는 전용 클래스.

    처리 방식:
        1. 카메라 이미지에서 도로 하단 ROI만 사용
        2. HSV 색상공간에서 노란색 마스크 추출
        3. 노란색 픽셀들로 직선 기울기 추정
        4. lookahead 지점의 노란선 x좌표 예측
        5. 화면 중심과 노란선 위치 차이 + 선 기울기로 조향각 계산
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        self.window_name = "Yellow Line Follower"
        self.mask_window_name = "Yellow Mask"

        # =====================================================
        # Driving parameters
        # =====================================================
        self.speed = 8.0
        self.max_angle = 50.0

        # 조향 튜닝값
        self.kp_position = 0.28
        self.kp_slope = 0.65

        # 조향 방향이 반대로 움직이면 -1.0으로 변경
        self.steer_sign = 1.0

        # =====================================================
        # ROI parameters
        # =====================================================
        # 화면 위쪽은 버리고 도로 하단만 사용
        self.roi_y_ratio = 0.50

        # ROI 안에서 어느 y 위치를 목표로 볼지
        # 0.75면 ROI의 아래쪽 75% 지점
        self.lookahead_y_ratio = 0.75

        # =====================================================
        # Yellow HSV threshold
        # =====================================================
        self.lower_yellow = np.array([15, 70, 70])
        self.upper_yellow = np.array([45, 255, 255])

        self.min_yellow_area = 250

        self.last_angle = 0.0
        self.last_speed = 0.0
        self.warned_no_yellow = False

    def process(self, image):
        if image is None:
            self.last_angle = 0.0
            self.last_speed = 0.0
            return 0.0, 0.0, "NO IMAGE"

        height, width = image.shape[:2]

        # =====================================================
        # 1. 도로 하단 ROI만 사용
        # =====================================================
        roi_y1 = int(height * self.roi_y_ratio)
        roi_y2 = height

        roi = image[roi_y1:roi_y2, :].copy()
        roi_height, roi_width = roi.shape[:2]

        # =====================================================
        # 2. 노란색 마스크 생성
        # =====================================================
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        yellow_mask = cv2.inRange(
            hsv,
            self.lower_yellow,
            self.upper_yellow
        )

        kernel = np.ones((5, 5), np.uint8)

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        yellow_area = cv2.countNonZero(yellow_mask)

        if yellow_area < self.min_yellow_area:
            self.last_angle = 0.0
            self.last_speed = 0.0

            if not self.warned_no_yellow:
                self.log_warn("YellowLineFollower: yellow line not detected")
                self.warned_no_yellow = True

            if self.show_debug:
                self.show_debug_image(
                    image=image,
                    mask=yellow_mask,
                    roi_y1=roi_y1,
                    target_x=None,
                    target_y=None,
                    line_angle_deg=0.0,
                    error_x=0.0,
                    mode_text="NO YELLOW - STOP"
                )

            return 0.0, 0.0, "NO YELLOW - STOP"

        self.warned_no_yellow = False

        # =====================================================
        # 3. 노란색 픽셀 좌표 추출
        # =====================================================
        ys, xs = np.where(yellow_mask > 0)

        if len(xs) < 20:
            self.last_angle = 0.0
            self.last_speed = 0.0
            return 0.0, 0.0, "YELLOW TOO SMALL"

        points = np.column_stack((xs, ys)).astype(np.float32)

        # =====================================================
        # 4. 노란색 선의 기울기 추정
        # =====================================================
        line = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = line.flatten()
        vx = float(vx)
        vy = float(vy)
        x0 = float(x0)
        y0 = float(y0)

        if abs(vy) < 1e-5:
            self.last_angle = 0.0
            self.last_speed = 0.0
            return 0.0, 0.0, "BAD LINE"

        # =====================================================
        # 5. lookahead 지점에서 노란선 위치 예측
        # =====================================================
        target_y = int(roi_height * self.lookahead_y_ratio)
        target_x = int(x0 + (target_y - y0) * vx / vy)

        target_x = max(0, min(target_x, roi_width - 1))

        image_center_x = roi_width // 2

        # 노란선이 화면 중심 기준 오른쪽이면 +
        # 왼쪽이면 -
        error_x = target_x - image_center_x

        # 수직선을 기준으로 한 기울기 각도
        # 오른쪽으로 기울면 +, 왼쪽으로 기울면 -
        line_angle_deg = math.degrees(math.atan2(vx, vy))

        # =====================================================
        # 6. 조향각 계산
        # =====================================================
        angle = (
            self.kp_position * error_x
            + self.kp_slope * line_angle_deg
        )

        angle = self.steer_sign * angle
        angle = self.clamp(angle, -self.max_angle, self.max_angle)

        speed = self.speed

        self.last_angle = angle
        self.last_speed = speed

        mode_text = "YELLOW FOLLOW"

        if self.show_debug:
            self.show_debug_image(
                image=image,
                mask=yellow_mask,
                roi_y1=roi_y1,
                target_x=target_x,
                target_y=target_y,
                line_angle_deg=line_angle_deg,
                error_x=error_x,
                mode_text=mode_text
            )

        return angle, speed, mode_text

    def show_debug_image(
        self,
        image,
        mask,
        roi_y1,
        target_x,
        target_y,
        line_angle_deg,
        error_x,
        mode_text
    ):
        debug_frame = image.copy()
        height, width = debug_frame.shape[:2]

        image_center_x = width // 2

        # ROI 영역 표시
        cv2.rectangle(
            debug_frame,
            (0, roi_y1),
            (width - 1, height - 1),
            (255, 255, 0),
            2
        )

        # 화면 중심선
        cv2.line(
            debug_frame,
            (image_center_x, roi_y1),
            (image_center_x, height - 1),
            (255, 255, 255),
            2
        )

        # 목표점 표시
        if target_x is not None and target_y is not None:
            target_y_image = roi_y1 + target_y

            cv2.circle(
                debug_frame,
                (target_x, target_y_image),
                8,
                (0, 255, 255),
                -1
            )

            cv2.line(
                debug_frame,
                (image_center_x, target_y_image),
                (target_x, target_y_image),
                (0, 255, 255),
                2
            )

        cv2.putText(
            debug_frame,
            "MISSION: AUTO_DRIVE",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"MODE: {mode_text}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"ERROR_X: {error_x:.1f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"LINE_ANGLE: {line_angle_deg:.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"ANGLE: {self.last_angle:.1f}  SPEED: {self.last_speed:.1f}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(self.window_name, debug_frame)
        cv2.imshow(self.mask_window_name, mask)
        cv2.waitKey(1)

    def stop(self):
        self.last_angle = 0.0
        self.last_speed = 0.0

        try:
            cv2.destroyWindow(self.window_name)
            cv2.destroyWindow(self.mask_window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(value, max_value))

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 전체 주행 로직.

    동작:
        1. 시작 후 0.5초 동안 angle=10, speed=8
        2. 그 다음 1.0초 동안 angle=0, speed=8
        3. 이후 YellowLineFollower 클래스로 노란색 중앙선 추종
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug

        self.angle = 0.0
        self.speed = 0.0

        # =====================================================
        # Initial fixed motion
        # =====================================================
        self.first_angle = 10.0
        self.first_speed = 8.0
        self.first_duration = 0.5

        self.forward_angle = 0.0
        self.forward_speed = 8.0
        self.forward_duration = 1.0

        # =====================================================
        # Runtime state
        # =====================================================
        self.started = False
        self.start_time = None
        self.log_count = 0

        # =====================================================
        # Yellow line follower
        # =====================================================
        self.yellow_follower = YellowLineFollower(
            logger=self.logger,
            show_debug=self.show_debug
        )

    def start(self):
        if self.started:
            return

        self.started = True
        self.start_time = time.time()

        self.log_info(
            "AutoDrive started: first turn -> straight -> yellow follow"
        )

    def process(self, image):
        if not self.started:
            self.start()

        elapsed_time = time.time() - self.start_time

        if elapsed_time < self.first_duration:
            self.angle = self.first_angle
            self.speed = self.first_speed
            mode_text = "FIRST TURN"

        elif elapsed_time < self.first_duration + self.forward_duration:
            self.angle = self.forward_angle
            self.speed = self.forward_speed
            mode_text = "GO STRAIGHT"

        else:
            self.angle, self.speed, mode_text = self.yellow_follower.process(
                image
            )

        self.log_count += 1

        if self.log_count % 25 == 1:
            self.log_info(
                f"AUTO_DRIVE | "
                f"elapsed:{elapsed_time:.2f}s | "
                f"mode:{mode_text} | "
                f"angle:{self.angle:.2f} | "
                f"speed:{self.speed:.2f}"
            )

        return self.angle, self.speed

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.started = False
        self.start_time = None

        try:
            self.yellow_follower.stop()
        except Exception:
            pass

        self.log_info("AutoDrive stopped")

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)