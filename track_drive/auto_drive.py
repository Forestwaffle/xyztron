#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import cv2


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 단순 주행 로직.

    동작:
        1. AUTO_DRIVE 시작 후 0.5초 동안 angle=10, speed=8
        2. 그 다음 6.0초 동안 angle=0, speed=8
        3. 총 6.5초 이후 angle=0, speed=0
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Simple Mode"

        self.angle = 0.0
        self.speed = 0.0

        self.first_angle = 10.0
        self.first_speed = 8.0
        self.first_duration = 0.5

        self.forward_angle = 0.0
        self.forward_speed = 8.0
        self.forward_duration = 4.0

        self.stop_angle = 0.0
        self.stop_speed = 0.0

        self.started = False
        self.start_time = None
        self.log_count = 0

    def start(self):
        if self.started:
            return

        self.started = True
        self.start_time = time.time()

        self.log_info(
            "AutoDrive started: 0.5s angle=10 speed=8, "
            "then 6.0s angle=0 speed=8, then stop"
        )

    def process(self, image):
        """
        AUTO_DRIVE 메인 로직.

        Returns:
            0.0 ~ 0.5초: angle=10.0, speed=8.0
            0.5 ~ 6.5초: angle=0.0, speed=8.0
            6.5초 이후: angle=0.0, speed=0.0
        """
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
            self.angle = self.stop_angle
            self.speed = self.stop_speed
            mode_text = "STOP"

        if self.show_debug and image is not None:
            self.show_camera_debug(image, elapsed_time, mode_text)

        self.log_count += 1
        if self.log_count % 25 == 1:
            self.log_info(
                f"AUTO_DRIVE simple | "
                f"elapsed:{elapsed_time:.2f}s | "
                f"mode:{mode_text} | "
                f"angle:{self.angle:.2f} | "
                f"speed:{self.speed:.2f}"
            )

        return self.angle, self.speed

    def show_camera_debug(self, image, elapsed_time, mode_text):
        debug_frame = image.copy()
        height, width = debug_frame.shape[:2]

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
            f"TIME: {elapsed_time:.2f}s",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"ANGLE: {self.angle:.1f}  SPEED: {self.speed:.1f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"SIZE: {width}x{height}",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(self.window_name, debug_frame)
        cv2.waitKey(1)

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.started = False
        self.start_time = None

        try:
            cv2.destroyWindow(self.window_name)
            cv2.waitKey(1)
        except cv2.error:
            pass

        self.log_info("AutoDrive stopped")

    def log_info(self, msg):
        if self.logger is not None:
            self.logger.info(msg)

    def log_warn(self, msg):
        if self.logger is not None:
            self.logger.warn(msg)