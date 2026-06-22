#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 노란색 점선 추종 로직.

    목표:
        차량 중심이 노란색 점선의 중심을 따라가도록 주행한다.

    방식:
        1. 전방 카메라 이미지를 Bird's Eye View로 변환
        2. 화면 아래 절반 ROI만 사용
        3. 노란색만 HSV로 검출
        4. 노란색 픽셀을 x = a*y + b 직선으로 피팅
        5. 노란색 점선이 끊기면 이전 직선 모델로 잠깐 예측
        6. lookahead_y 위치에서 노란선의 예상 x 좌표 계산
        7. 차량 중심과 노란선 x 좌표 오차로 조향
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Yellow Dashed Line Follow"

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        # 천천히 주행
        self.line_detected_speed = 6.0

        # 점선이 잠깐 끊겨서 예측으로 갈 때 속도
        self.prediction_speed = 5.0

        # 노란선도 예측도 안 될 때 직진 속도
        self.no_line_speed = 4.0

        self.max_angle = 80.0

        # =====================================================
        # Control parameters
        # =====================================================
        # 차량 중심 목표. 화면 중앙.
        self.target_center_x_ratio = 0.50

        # Bird's Eye View 기준으로 어느 y 지점의 노란선을 따라갈지.
        # 0.78이면 아래쪽보다 조금 앞을 봄.
        self.lookahead_y_ratio = 0.78

        # 가까운 지점. 선의 진행 방향 예측용.
        self.near_y_ratio = 0.95

        # 중심 오차 조향 게인.
        self.center_gain = 0.22

        # 기울기 예측 보정 게인.
        self.heading_gain = 0.30

        # 조향 방향 보정.
        # 차가 반대로 꺾이면 -1.0을 1.0으로 바꾸면 됨.
        self.steering_sign = -1.0

        # 조향 smoothing.
        # 값이 클수록 부드럽지만 반응이 느림.
        self.smoothing_alpha = 0.70
        self.prev_angle = 0.0

        # =====================================================
        # ROI parameters
        # =====================================================
        # 화면 아래 절반만 사용
        self.roi_y_start_ratio = 0.50

        # =====================================================
        # Yellow color threshold
        # =====================================================
        # Unity 시뮬 노란색 점선 기준
        self.yellow_lower = np.array([15, 70, 90])
        self.yellow_upper = np.array([40, 255, 255])

        # 노란색 픽셀 최소 개수
        self.min_yellow_pixels = 80

        # =====================================================
        # Prediction parameters
        # =====================================================
        # 점선이 끊겼을 때 이전 직선 모델을 유지할 최대 프레임 수
        self.max_prediction_frames = 10
        self.prediction_count = 0

        # 직선 모델: x = a*y + b
        self.last_fit = None

        # 새 직선 모델 smoothing
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

        self.last_warped = None
        self.last_mask = None
        self.last_roi_y1 = None

    # =====================================================
    # Lifecycle
    # =====================================================

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info(
            "AutoDrive started: yellow dashed line follow "
            f"| line_speed:{self.line_detected_speed:.1f} "
            f"| prediction_speed:{self.prediction_speed:.1f} "
            f"| no_line_speed:{self.no_line_speed:.1f}"
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
            # 노란선도 없고 예측도 불가능하면 느린 직진
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

            # 차량 중심이 노란선 중심으로 가게 만드는 오차
            center_error = target_x - line_x
            self.last_error = center_error

            # 기울기 기반 예측 오차
            # near_x와 lookahead line_x 차이를 이용해서 선의 방향성을 반영
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

            if detected_now:
                self.speed = self.line_detected_speed
            else:
                self.speed = self.prediction_speed

            self.prev_angle = self.angle

        if self.show_debug:
            self.show_debug_view(image)

        self.print_log()

        return self.angle, self.speed

    # =====================================================
    # Image warp
    # =====================================================

    def warp_image(self, img):
        """
        전방 카메라 이미지를 Bird's Eye View로 변환.
        기존 네 코드에서 쓰던 방식과 유사한 사다리꼴 변환.
        """
        h, w = img.shape[:2]

        src = np.float32([
            [int(w * 0.02), h],
            [int(w * 0.35), int(h * 0.55)],
            [int(w * 0.65), int(h * 0.55)],
            [int(w * 0.98), h]
        ])

        dst = np.float32([
            [int(w * 0.15), h],
            [int(w * 0.15), 0],
            [int(w * 0.85), 0],
            [int(w * 0.85), h]
        ])

        matrix = cv2.getPerspectiveTransform(src, dst)

        warped = cv2.warpPerspective(
            img,
            matrix,
            (w, h),
            flags=cv2.INTER_LINEAR
        )

        return warped

    # =====================================================
    # Yellow line detection / prediction
    # =====================================================

    def detect_yellow_line(self, image):
        warped = self.warp_image(image)

        height, width = warped.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        roi = warped[roi_y1:height, 0:width].copy()

        if roi.size == 0:
            return None

        yellow_mask = self.create_yellow_mask(roi)

        # 점선 노이즈 정리
        open_kernel = np.ones((3, 3), np.uint8)
        close_kernel = np.ones((9, 9), np.uint8)

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

        self.last_warped = warped
        self.last_mask = yellow_mask
        self.last_roi_y1 = roi_y1

        ys, xs = yellow_mask.nonzero()

        pixel_count = len(xs)
        detected_now = pixel_count >= self.min_yellow_pixels

        fit = None

        if detected_now:
            # ROI 좌표를 전체 warped 이미지 좌표로 변환
            full_ys = ys + roi_y1
            full_xs = xs

            try:
                # 노란선 직선 피팅: x = a*y + b
                new_fit = np.polyfit(full_ys, full_xs, 1)

                if self.last_fit is None:
                    fit = new_fit
                else:
                    fit = (
                        self.fit_alpha * self.last_fit
                        + (1.0 - self.fit_alpha) * new_fit
                    )

                self.last_fit = fit
                self.prediction_count = 0

            except Exception:
                fit = None

        else:
            # 점선이 끊겨서 현재 프레임에서 노란색이 부족하면 이전 모델로 예측
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

        a = float(fit[0])
        b = float(fit[1])

        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        line_x = a * lookahead_y + b
        near_x = a * near_y + b

        # 예측값이 화면을 너무 벗어나면 폐기
        if line_x < -width * 0.25 or line_x > width * 1.25:
            return None

        if near_x < -width * 0.25 or near_x > width * 1.25:
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
        if self.last_warped is None:
            return

        original = image.copy()
        warped_debug = self.last_warped.copy()

        height, width = warped_debug.shape[:2]

        roi_y1 = int(height * self.roi_y_start_ratio)
        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        # ROI 표시
        cv2.rectangle(
            warped_debug,
            (0, roi_y1),
            (width - 1, height - 1),
            (255, 255, 0),
            2
        )

        # lookahead / near 라인 표시
        cv2.line(
            warped_debug,
            (0, lookahead_y),
            (width, lookahead_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            warped_debug,
            (0, near_y),
            (width, near_y),
            (120, 120, 120),
            2
        )

        # 목표 차량 중심선
        if self.last_target_x is not None:
            target_x = int(self.last_target_x)

            cv2.line(
                warped_debug,
                (target_x, roi_y1),
                (target_x, height),
                (255, 255, 255),
                2
            )

            cv2.circle(
                warped_debug,
                (target_x, lookahead_y),
                8,
                (255, 255, 255),
                -1
            )

        # 예측된 노란선 위치
        if self.last_line_x is not None:
            line_x = int(self.last_line_x)

            cv2.line(
                warped_debug,
                (line_x, roi_y1),
                (line_x, height),
                (0, 255, 255),
                2
            )

            cv2.circle(
                warped_debug,
                (line_x, lookahead_y),
                10,
                (0, 255, 255),
                -1
            )

        if self.last_near_x is not None:
            near_x = int(self.last_near_x)

            cv2.circle(
                warped_debug,
                (near_x, near_y),
                8,
                (0, 180, 255),
                -1
            )

        # 피팅된 직선 표시
        if self.last_fit is not None:
            a = float(self.last_fit[0])
            b = float(self.last_fit[1])

            y1 = roi_y1
            y2 = height - 1

            x1 = int(a * y1 + b)
            x2 = int(a * y2 + b)

            x1 = int(self.clamp(x1, 0, width - 1))
            x2 = int(self.clamp(x2, 0, width - 1))

            cv2.line(
                warped_debug,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                4
            )

        # 상태 표시
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
            warped_debug,
            status,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            status_color,
            2
        )

        cv2.putText(
            warped_debug,
            f"ANGLE:{self.angle:.2f} SPEED:{self.speed:.2f}",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.70,
            (255, 255, 255),
            2
        )

        if self.last_error is not None:
            cv2.putText(
                warped_debug,
                f"LINE_X:{self.last_line_x:.1f} "
                f"TARGET:{self.last_target_x:.1f} "
                f"ERR:{self.last_error:.1f}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                warped_debug,
                f"HEAD:{self.last_heading_error:.1f} "
                f"PIXELS:{self.last_pixel_count} "
                f"PRED:{self.prediction_count}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

        # 마스크 표시용 전체 이미지 생성
        mask_view = np.zeros((height, width), dtype=np.uint8)

        if self.last_mask is not None and self.last_roi_y1 is not None:
            mask_h, mask_w = self.last_mask.shape[:2]
            mask_view[self.last_roi_y1:self.last_roi_y1 + mask_h, 0:mask_w] = self.last_mask

        mask_color = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)

        # 3분할 디버그 화면
        view_w = 320
        view_h = 240

        original_small = cv2.resize(original, (view_w, view_h))
        warped_small = cv2.resize(warped_debug, (view_w, view_h))
        mask_small = cv2.resize(mask_color, (view_w, view_h))

        top = np.hstack((original_small, warped_small))
        bottom = np.hstack((mask_small, mask_small))
        combined = np.vstack((top, bottom))

        cv2.imshow(self.window_name, combined)
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