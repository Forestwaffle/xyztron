#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 노란색 점선 중심 추종 로직.

    목표:
        차량 중심이 노란색 선의 중심을 따라가도록 조향한다.

    방식:
        1. 전방 카메라에서 노란색만 마스킹
        2. 아래쪽 ROI에서 노란색 픽셀 추출
        3. x = a*y + b 형태의 직선 모델로 노란선 중심 피팅
        4. 점선이 끊겨서 안 보이는 순간에는 이전 직선 모델로 예측
        5. lookahead_y 위치에서 예상 노란선 x 좌표 계산
        6. 화면 중심과 노란선 x 좌표 오차로 조향
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Yellow Line Follow"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        # 천천히 테스트용 속도
        self.line_detected_speed = 8.0
        self.no_line_speed = 5.0

        self.max_angle = 100.0

        # =====================================================
        # Control parameters
        # =====================================================
        # 화면 중심을 목표 차량 중심으로 사용
        self.target_center_x_ratio = 0.50

        # 노란선을 예측할 y 위치.
        # 0.80이면 화면 아래에서 조금 앞쪽을 봄.
        self.lookahead_y_ratio = 0.80

        # 더 가까운 위치. 선의 진행 방향 예측용.
        self.near_y_ratio = 0.95

        # 조향 민감도
        self.center_gain = 0.28

        # 기울기 기반 예측 보정
        self.heading_gain = 0.35

        # 조향 방향 보정.
        # 차가 반대로 꺾이면 -1.0을 1.0으로 변경.
        self.steering_sign = -1.0

        # 조향 부드럽게 만들기
        self.smoothing_alpha = 0.65
        self.prev_angle = 0.0

        # =====================================================
        # Detection parameters
        # =====================================================
        # 아래쪽 ROI만 사용
        self.roi_y_start_ratio = 0.42

        # 노란색 HSV threshold
        self.yellow_lower = np.array([15, 70, 90])
        self.yellow_upper = np.array([40, 255, 255])

        # 노란색 픽셀 최소 개수
        self.min_yellow_pixels = 80

        # 이전 모델 유지 프레임 수
        self.max_prediction_frames = 8
        self.prediction_count = 0

        # 직선 모델 x = a*y + b
        self.last_fit = None

        # 최근 검출값 smoothing
        self.fit_alpha = 0.70

        # =====================================================
        # Debug / state
        # =====================================================
        self.started = False
        self.warned_no_image = False
        self.log_count = 0

        self.line_detected = False
        self.using_prediction = False

        self.last_line_x = None
        self.last_near_x = None
        self.last_target_x = None
        self.last_error = None
        self.last_heading_error = None
        self.last_pixel_count = 0
        self.last_mask = None

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info(
            "AutoDrive started: yellow dashed line center follow "
            f"| speed:{self.line_detected_speed:.1f}"
        )

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.prev_angle = 0.0
        self.started = False

        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

        self.log_info("AutoDrive stopped")

    # =====================================================
    # Main process
    # =====================================================

    def process(self, image):
        if image is None:
            if not self.warned_no_image:
                self.log_warn("AUTO_DRIVE: waiting for camera image...")
                self.warned_no_image = True

            self.angle = 0.0
            self.speed = self.no_line_speed
            return self.angle, self.speed

        self.warned_no_image = False

        result = self.detect_yellow_line(image)

        if result is None:
            # 노란색 선도 없고 예측 모델도 없으면 천천히 직진
            self.line_detected = False
            self.using_prediction = False
            self.last_line_x = None
            self.last_near_x = None
            self.last_target_x = None
            self.last_error = None
            self.last_heading_error = None
            self.last_pixel_count = 0

            self.angle = 0.0
            self.speed = self.no_line_speed
            self.prev_angle = 0.0

        else:
            line_x, near_x, target_x, detected_now, pixel_count = result

            self.line_detected = detected_now
            self.using_prediction = not detected_now

            self.last_line_x = line_x
            self.last_near_x = near_x
            self.last_target_x = target_x
            self.last_pixel_count = pixel_count

            # 중심 오차:
            # 노란선 예상 위치와 차량 화면 중심 차이
            center_error = target_x - line_x
            self.last_error = center_error

            # 기울기 예측 오차:
            # 가까운 위치와 lookahead 위치의 x 차이로 선의 진행 방향 반영
            heading_error = near_x - line_x
            self.last_heading_error = heading_error

            control_error = center_error + self.heading_gain * heading_error

            raw_angle = (
                self.steering_sign
                * self.center_gain
                * control_error
            )

            raw_angle = self.clamp(
                raw_angle,
                -self.max_angle,
                self.max_angle
            )

            smoothed_angle = (
                self.smoothing_alpha * self.prev_angle
                + (1.0 - self.smoothing_alpha) * raw_angle
            )

            self.angle = self.clamp(
                smoothed_angle,
                -self.max_angle,
                self.max_angle
            )

            self.speed = self.line_detected_speed
            self.prev_angle = self.angle

        if self.show_debug:
            self.show_debug_view(image)

        self.print_log()

        return self.angle, self.speed

    # =====================================================
    # Yellow line detection / prediction
    # =====================================================

    def detect_yellow_line(self, image):
        height, width = image.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        roi = image[roi_y1:height, 0:width].copy()

        if roi.size == 0:
            return None

        yellow_mask = self.create_yellow_mask(roi)

        # 점선 조각 노이즈 정리
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((7, 7), np.uint8)

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_OPEN,
            open_kernel
        )

        yellow_mask = cv2.morphologyEx(
            yellow_mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        self.last_mask = yellow_mask

        ys, xs = yellow_mask.nonzero()

        pixel_count = len(xs)

        detected_now = pixel_count >= self.min_yellow_pixels

        fit = None

        if detected_now:
            # ROI 좌표를 전체 이미지 좌표로 변환
            full_ys = ys + roi_y1
            full_xs = xs

            try:
                # 직선 피팅: x = a*y + b
                new_fit = np.polyfit(full_ys, full_xs, 1)

                if self.last_fit is None:
                    fit = new_fit
                else:
                    # 직선 모델 smoothing
                    fit = (
                        self.fit_alpha * self.last_fit
                        + (1.0 - self.fit_alpha) * new_fit
                    )

                self.last_fit = fit
                self.prediction_count = 0

            except Exception:
                fit = None

        else:
            # 점선이 끊겨서 현재 프레임에서 안 보이면 이전 모델로 예측
            if (
                self.last_fit is not None
                and self.prediction_count < self.max_prediction_frames
            ):
                fit = self.last_fit
                self.prediction_count += 1
                detected_now = False
            else:
                self.last_fit = None
                self.prediction_count = 0
                return None

        if fit is None:
            return None

        a = fit[0]
        b = fit[1]

        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        line_x = a * lookahead_y + b
        near_x = a * near_y + b

        # 화면 밖으로 너무 벗어난 예측 제거
        if line_x < -width * 0.2 or line_x > width * 1.2:
            return None

        if near_x < -width * 0.2 or near_x > width * 1.2:
            return None

        line_x = self.clamp(line_x, 0.0, float(width - 1))
        near_x = self.clamp(near_x, 0.0, float(width - 1))

        target_x = width * self.target_center_x_ratio

        return (
            float(line_x),
            float(near_x),
            float(target_x),
            detected_now,
            int(pixel_count)
        )

    def create_yellow_mask(self, image):
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = cv2.GaussianBlur(hsv, (5, 5), 0)

        yellow_mask = cv2.inRange(
            hsv,
            self.yellow_lower,
            self.yellow_upper
        )

        return yellow_mask

    # =====================================================
    # Debug view
    # =====================================================

    def show_debug_view(self, image):
        debug_frame = image.copy()
        height, width = debug_frame.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        # ROI 표시
        cv2.rectangle(
            debug_frame,
            (0, roi_y1),
            (width - 1, height - 1),
            (255, 255, 0),
            2
        )

        # lookahead / near line
        cv2.line(
            debug_frame,
            (0, lookahead_y),
            (width, lookahead_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            debug_frame,
            (0, near_y),
            (width, near_y),
            (120, 120, 120),
            2
        )

        # 차량 목표 중심
        if self.last_target_x is not None:
            target_x = int(self.last_target_x)

            cv2.line(
                debug_frame,
                (target_x, roi_y1),
                (target_x, height),
                (255, 255, 255),
                2
            )

        # 예측된 노란선 위치
        if self.last_line_x is not None:
            line_x = int(self.last_line_x)

            cv2.circle(
                debug_frame,
                (line_x, lookahead_y),
                10,
                (0, 255, 255),
                -1
            )

            cv2.line(
                debug_frame,
                (line_x, roi_y1),
                (line_x, height),
                (0, 255, 255),
                2
            )

        if self.last_near_x is not None:
            near_x = int(self.last_near_x)

            cv2.circle(
                debug_frame,
                (near_x, near_y),
                8,
                (0, 180, 255),
                -1
            )

        # 피팅된 선 시각화
        if self.last_fit is not None:
            a = self.last_fit[0]
            b = self.last_fit[1]

            y1 = roi_y1
            y2 = height - 1

            x1 = int(a * y1 + b)
            x2 = int(a * y2 + b)

            x1 = int(self.clamp(x1, 0, width - 1))
            x2 = int(self.clamp(x2, 0, width - 1))

            cv2.line(
                debug_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                4
            )

        if self.using_prediction:
            status = "YELLOW PREDICTION"
            status_color = (0, 165, 255)
        elif self.line_detected:
            status = "YELLOW DETECTED"
            status_color = (0, 255, 0)
        else:
            status = "NO YELLOW LINE"
            status_color = (0, 0, 255)

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
            status,
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2
        )

        cv2.putText(
            debug_frame,
            f"ANGLE:{self.angle:.2f} SPEED:{self.speed:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if self.last_error is not None:
            cv2.putText(
                debug_frame,
                f"LINE_X:{self.last_line_x:.1f} "
                f"TARGET:{self.last_target_x:.1f} "
                f"ERR:{self.last_error:.1f}",
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                debug_frame,
                f"HEAD:{self.last_heading_error:.1f} "
                f"PIXELS:{self.last_pixel_count} "
                f"PRED:{self.prediction_count}",
                (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow(self.window_name, debug_frame)
        cv2.waitKey(1)

    # =====================================================
    # Utility
    # =====================================================

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(max_value, value))

    def print_log(self):
        self.log_count += 1

        if self.log_count % 20 != 0:
            return

        line_x_text = (
            "None"
            if self.last_line_x is None
            else f"{self.last_line_x:.1f}"
        )

        err_text = (
            "None"
            if self.last_error is None
            else f"{self.last_error:.1f}"
        )

        self.log_info(
            f"AUTO_DRIVE | "
            f"detected:{self.line_detected} | "
            f"prediction:{self.using_prediction} | "
            f"line_x:{line_x_text} | "
            f"err:{err_text} | "
            f"pixels:{self.last_pixel_count} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)