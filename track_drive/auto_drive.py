#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2


class AutoDrive:
    """
    AUTO_DRIVE 상태에서 실행되는 별도 주행 로직.

    현재 기능:
        1. 차량 정지 명령 반환
        2. 전방 카메라 이미지 디버그 창 표시
    """

    def __init__(self, logger=None, show_debug=True):
        self.logger = logger
        self.show_debug = show_debug
        self.window_name = "AUTO_DRIVE Camera Debug"

        self.angle = 0.0
        self.speed = 0.0

        self.log_count = 0
        self.warned_no_image = False
        self.started = False

    def start(self):
        if self.started:
            return

        self.started = True
        self.log_info("AutoDrive started: stop + camera debug")

    def process(self, image):
        """
        AUTO_DRIVE 메인 로직.

        현재는 정지 상태를 유지하고 카메라 화면만 띄운다.

        Returns:
            angle = 0.0
            speed = 0.0
        """
        self.angle = 0.0
        self.speed = 0.0

        if image is None:
            if not self.warned_no_image:
                self.log_warn("AUTO_DRIVE: waiting for camera image...")
                self.warned_no_image = True

            return self.angle, self.speed

        self.warned_no_image = False

        if self.show_debug:
            self.show_camera_debug(image)

        return self.angle, self.speed

    def show_camera_debug(self, image):
        debug_frame = image.copy()
        height, width = debug_frame.shape[:2]

        self.log_count += 1

        if self.log_count % 20 == 1:
            self.log_info(
                f"AUTO_DRIVE camera debug | "
                f"shape:{debug_frame.shape} | "
                f"width:{width} | height:{height}"
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
            "STOP + CAMERA DEBUG",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            debug_frame,
            f"SIZE: {width}x{height}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(self.window_name, debug_frame)
        cv2.waitKey(1)

    def stop(self):
        self.angle = 0.0
        self.speed = 0.0
        self.started = False

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