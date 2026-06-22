#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2
import numpy as np


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 주행 로직.

    동작:
        1. AUTO_DRIVE 시작 후 0.5초 동안 angle=10, speed=8
        2. 그 다음 1.0초 동안 angle=0, speed=8
        3. 이후 Bird's Eye View 기반 노란색 점선 추종
        4. 노란색 점선이 끊기면 이전 직선 모델로 잠깐 예측
        5. 예측도 끝났는데 노란선이 안 보이면 마지막 조향각 유지
        6. 코너에서는 더 일찍 반응하고 속도를 낮춤
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Yellow Dashed Line Follow"

        # =====================================================
        # Initial hard-coded driving logic
        # =====================================================
        self.first_angle = 10.0
        self.first_speed = 8.0
        self.first_duration = 0.5

        self.forward_angle = 0.0
        self.forward_speed = 8.0
        self.forward_duration = 1.0

        # =====================================================
        # Drive parameters
        # =====================================================
        self.angle = 0.0
        self.speed = 0.0

        self.line_detected_speed = 6.0
        self.prediction_speed = 5.0
        self.no_line_speed = 4.0

        self.corner_medium_speed = 5.0
        self.corner_strong_speed = 4.5

        self.max_angle = 100.0

        # =====================================================
        # Control parameters
        # =====================================================
        self.target_center_x_ratio = 0.50

        # 기존 0.78 -> 0.62
        # 더 먼 지점을 봐서 코너에 일찍 반응
        self.lookahead_y_ratio = 0.62

        # 기존 0.95 -> 0.88
        self.near_y_ratio = 0.88

        # 기존 0.30 -> 0.50
        # 선의 기울기 반영 강화
        self.heading_gain = 0.50

        # 차가 반대로 꺾이면 1.0으로 변경
        self.steering_sign = -1.0

        # =====================================================
        # Dynamic response parameters
        # =====================================================
        # 기존 20 -> 10
        self.near_error_px = 10.0

        # 기존 120 -> 80
        self.far_error_px = 80.0

        # 기존 0.14 -> 0.18
        self.min_center_gain = 0.18

        # 기존 0.55 -> 0.70
        self.max_center_gain = 0.70

        # 기존 5 -> 3
        self.dead_zone_px = 3.0

        # 기존 0.75 -> 0.60
        self.near_smoothing_alpha = 0.60

        # 기존 0.30 -> 0.18
        self.far_smoothing_alpha = 0.18

        self.last_dynamic_gain = 0.0
        self.last_dynamic_smoothing = 0.0

        self.prev_angle = 0.0

        # =====================================================
        # ROI parameters
        # =====================================================
        # 기존 0.50 -> 0.42
        # 도로의 더 먼 부분까지 사용해서 코너를 일찍 봄
        self.roi_y_start_ratio = 0.42

        # =====================================================
        # Yellow color threshold
        # =====================================================
        self.yellow_lower = np.array([15, 70, 90])
        self.yellow_upper = np.array([40, 255, 255])

        self.min_yellow_pixels = 80

        # =====================================================
        # Prediction parameters
        # =====================================================
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
        self.start_time = None
        self.warned_no_image = False
        self.log_count = 0

        self.initial_mode = True
        self.line_detected = False
        self.using_prediction = False
        self.holding_angle = False

        self.last_line_x = None
        self.last_near_x = None
        self.last_target_x = None
        self.last_error = None
        self.last_heading_error = None
        self.last_control_error = None
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
        self.start_time = time.time()

        self.log_info(
            "AutoDrive started: "
            "0.5s angle=10 speed=8 -> "
            "1.0s angle=0 speed=8 -> "
            "early corner yellow dashed line follow"
        )

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.prev_angle = 0.0

        self.started = False
        self.start_time = None

        self.initial_mode = True
        self.line_detected = False
        self.using_prediction = False
        self.holding_angle = False

        self.last_fit = None
        self.prediction_count = 0

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
        if not self.started:
            self.start()

        elapsed_time = time.time() - self.start_time

        # =====================================================
        # 1. Initial hard-coded driving logic
        # =====================================================
        if elapsed_time < self.first_duration:
            self.angle = self.first_angle
            self.speed = self.first_speed
            self.prev_angle = self.angle

            self.initial_mode = True
            self.line_detected = False
            self.using_prediction = False
            self.holding_angle = False

            self.print_initial_log(elapsed_time, "FIRST TURN")
            return self.angle, self.speed

        elif elapsed_time < self.first_duration + self.forward_duration:
            self.angle = self.forward_angle
            self.speed = self.forward_speed
            self.prev_angle = self.angle

            self.initial_mode = True
            self.line_detected = False
            self.using_prediction = False
            self.holding_angle = False

            self.print_initial_log(elapsed_time, "GO STRAIGHT")
            return self.angle, self.speed

        self.initial_mode = False

        # =====================================================
        # 2. Yellow dashed line following logic
        # =====================================================
        if image is None:
            if not self.warned_no_image:
                self.log_warn("AUTO_DRIVE: waiting for camera image...")
                self.warned_no_image = True

            self.angle = self.prev_angle
            self.speed = self.no_line_speed
            return self.angle, self.speed

        self.warned_no_image = False

        result = self.detect_yellow_line(image)

        if result is None:
            self.line_detected = False
            self.using_prediction = False
            self.holding_angle = True

            self.last_line_x = None
            self.last_near_x = None
            self.last_target_x = None
            self.last_error = None
            self.last_heading_error = None
            self.last_control_error = None
            self.last_pixel_count = 0
            self.last_dynamic_gain = 0.0
            self.last_dynamic_smoothing = 0.0

            # 노란선도 예측도 없으면 마지막 각도 유지
            self.angle = self.prev_angle
            self.speed = self.no_line_speed

        else:
            line_x, near_x, target_x, detected_now, pixel_count = result

            self.line_detected = detected_now
            self.using_prediction = not detected_now
            self.holding_angle = False

            self.last_line_x = line_x
            self.last_near_x = near_x
            self.last_target_x = target_x
            self.last_pixel_count = pixel_count

            # 차량 중심이 노란선 중심으로 가게 만드는 오차
            center_error = target_x - line_x
            self.last_error = center_error

            # 기울기 기반 예측 오차
            heading_error = near_x - line_x
            self.last_heading_error = heading_error

            control_error = center_error + self.heading_gain * heading_error
            self.last_control_error = control_error

            dynamic_gain, dynamic_smoothing = self.get_dynamic_response(
                control_error
            )

            self.last_dynamic_gain = dynamic_gain
            self.last_dynamic_smoothing = dynamic_smoothing

            raw_angle = (
                self.steering_sign
                * dynamic_gain
                * control_error
            )

            raw_angle = self.clamp(
                raw_angle,
                -self.max_angle,
                self.max_angle
            )

            smoothed_angle = (
                dynamic_smoothing * self.prev_angle
                + (1.0 - dynamic_smoothing) * raw_angle
            )

            self.angle = self.clamp(
                smoothed_angle,
                -self.max_angle,
                self.max_angle
            )

            # 코너에서 속도 낮추기
            abs_control_error = abs(control_error)

            if detected_now:
                if abs_control_error > 70.0:
                    self.speed = self.corner_strong_speed
                elif abs_control_error > 40.0:
                    self.speed = self.corner_medium_speed
                else:
                    self.speed = self.line_detected_speed
            else:
                self.speed = self.prediction_speed

            self.prev_angle = self.angle

        if self.show_debug:
            self.show_debug_view(image)

        self.print_log()

        return self.angle, self.speed

    # =====================================================
    # Dynamic response
    # =====================================================

    def get_dynamic_response(self, control_error):
        abs_error = abs(control_error)

        if abs_error < self.dead_zone_px:
            return 0.0, self.near_smoothing_alpha

        error_ratio = (
            (abs_error - self.near_error_px)
            / (self.far_error_px - self.near_error_px)
        )

        error_ratio = self.clamp(error_ratio, 0.0, 1.0)

        dynamic_gain = (
            self.min_center_gain
            + error_ratio * (self.max_center_gain - self.min_center_gain)
        )

        dynamic_smoothing = (
            self.near_smoothing_alpha
            - error_ratio * (
                self.near_smoothing_alpha - self.far_smoothing_alpha
            )
        )

        return dynamic_gain, dynamic_smoothing

    # =====================================================
    # Image warp
    # =====================================================

    def warp_image(self, img):
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
            # 점선이 끊기면 이전 직선 모델로 예측
            if (
                self.last_fit is not None
                and self.prediction_count < self.max_prediction_frames
            ):
                fit = self.last_fit
                self.prediction_count += 1
                detected_now = False
            else:
                self.prediction_count = self.max_prediction_frames
                return None

        if fit is None:
            return None

        a = float(fit[0])
        b = float(fit[1])

        lookahead_y = int(height * self.lookahead_y_ratio)
        near_y = int(height * self.near_y_ratio)

        line_x = a * lookahead_y + b
        near_x = a * near_y + b

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

        cv2.rectangle(
            warped_debug,
            (0, roi_y1),
            (width - 1, height - 1),
            (255, 255, 0),
            2
        )

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

        if self.last_target_x is not None:
            target_x = int(self.last_target_x)
        else:
            target_x = int(width * self.target_center_x_ratio)

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

        if self.holding_angle:
            status = "NO LINE - HOLD ANGLE"
            status_color = (0, 0, 255)
        elif self.using_prediction:
            status = "YELLOW PREDICTION"
            status_color = (0, 165, 255)
        elif self.line_detected:
            status = "YELLOW DETECTED"
            status_color = (0, 255, 0)
        elif self.initial_mode:
            status = "INITIAL HARD CODED DRIVE"
            status_color = (255, 255, 255)
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
                f"CTRL:{self.last_control_error:.1f} "
                f"PIX:{self.last_pixel_count}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )

            cv2.putText(
                warped_debug,
                f"GAIN:{self.last_dynamic_gain:.3f} "
                f"SMOOTH:{self.last_dynamic_smoothing:.2f} "
                f"PRED:{self.prediction_count}",
                (20, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2
            )
        else:
            cv2.putText(
                warped_debug,
                f"HOLD_ANGLE:{self.prev_angle:.2f} "
                f"PRED:{self.prediction_count}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (255, 255, 255),
                2
            )

        mask_view = np.zeros((height, width), dtype=np.uint8)

        if self.last_mask is not None and self.last_roi_y1 is not None:
            mask_h, mask_w = self.last_mask.shape[:2]
            mask_view[
                self.last_roi_y1:self.last_roi_y1 + mask_h,
                0:mask_w
            ] = self.last_mask

        mask_color = cv2.cvtColor(mask_view, cv2.COLOR_GRAY2BGR)

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

    def print_initial_log(self, elapsed_time, mode_text):
        self.log_count += 1

        if self.log_count % 20 != 0:
            return

        self.log_info(
            f"AUTO_DRIVE INITIAL | "
            f"elapsed:{elapsed_time:.2f} | "
            f"mode:{mode_text} | "
            f"angle:{self.angle:.2f} | "
            f"speed:{self.speed:.2f}"
        )

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

        ctrl_text = (
            "None"
            if self.last_control_error is None
            else f"{self.last_control_error:.1f}"
        )

        self.log_info(
            f"AUTO_DRIVE | "
            f"detected:{self.line_detected} | "
            f"prediction:{self.using_prediction} | "
            f"hold:{self.holding_angle} | "
            f"line_x:{line_x_text} | "
            f"err:{err_text} | "
            f"ctrl:{ctrl_text} | "
            f"gain:{self.last_dynamic_gain:.3f} | "
            f"smooth:{self.last_dynamic_smoothing:.2f} | "
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