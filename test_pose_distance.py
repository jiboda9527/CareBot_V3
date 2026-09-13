#!/usr/bin/env python3
"""Standalone MediaPipe Pose pixel-distance test (no robot control)."""

import argparse
import math
import sys
from collections import deque

import cv2
import mediapipe as mp


LANDMARK_VISIBILITY_MIN = 0.50


class MovingAverage:
    """Small rolling average to make the displayed pixel values readable."""

    def __init__(self, window_size):
        self.values = deque(maxlen=window_size)

    def update(self, value):
        self.values.append(value)
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()


def open_camera(camera_indexes):
    """Use the project's existing V4L2 probing convention without importing motor code."""
    for camera_index in camera_indexes:
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                print(f"Camera found: /dev/video{camera_index}")
                return capture
        capture.release()
    return None


def landmark_to_pixel(landmark, frame_width, frame_height):
    """Return a visible, in-frame landmark as an integer pixel coordinate."""
    if (
        landmark.visibility < LANDMARK_VISIBILITY_MIN
        or not 0.0 <= landmark.x <= 1.0
        or not 0.0 <= landmark.y <= 1.0
    ):
        return None
    return int(landmark.x * frame_width), int(landmark.y * frame_height)


def distance(point_a, point_b):
    return math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def draw_measurement(frame, point_a, point_b, text, text_y):
    cv2.line(frame, point_a, point_b, (0, 255, 255), 3)
    cv2.circle(frame, point_a, 5, (0, 255, 255), -1)
    cv2.circle(frame, point_b, 5, (0, 255, 255), -1)
    cv2.putText(
        frame, text, (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure shoulder width and torso length in pixels with MediaPipe Pose."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        help="Use one V4L2 camera index directly (default: probe 0, 1, 2, 3).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=8,
        help="Number of valid frames in the moving average (default: 8).",
    )
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be at least 1")
    return args


def main():
    args = parse_args()
    camera_indexes = [args.camera] if args.camera is not None else [0, 1, 2, 3]
    capture = open_camera(camera_indexes)
    if capture is None:
        print("No camera found. Check the USB camera and /dev/video devices.", file=sys.stderr)
        return 1

    shoulder_smoother = MovingAverage(args.smooth_window)
    torso_smoother = MovingAverage(args.smooth_window)
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_style = mp.solutions.drawing_styles

    try:
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as pose:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    print("Camera frame read failed.", file=sys.stderr)
                    break

                result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                display = frame.copy()
                status = "Pose Lost"

                if result.pose_landmarks:
                    mp_draw.draw_landmarks(
                        display,
                        result.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_style.get_default_pose_landmarks_style(),
                    )
                    landmarks = result.pose_landmarks.landmark
                    height, width = display.shape[:2]
                    left_shoulder = landmark_to_pixel(
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER], width, height
                    )
                    right_shoulder = landmark_to_pixel(
                        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER], width, height
                    )
                    left_hip = landmark_to_pixel(
                        landmarks[mp_pose.PoseLandmark.LEFT_HIP], width, height
                    )
                    right_hip = landmark_to_pixel(
                        landmarks[mp_pose.PoseLandmark.RIGHT_HIP], width, height
                    )

                    if all((left_shoulder, right_shoulder, left_hip, right_hip)):
                        shoulder_center = (
                            (left_shoulder[0] + right_shoulder[0]) // 2,
                            (left_shoulder[1] + right_shoulder[1]) // 2,
                        )
                        hip_center = (
                            (left_hip[0] + right_hip[0]) // 2,
                            (left_hip[1] + right_hip[1]) // 2,
                        )
                        shoulder_width = shoulder_smoother.update(
                            distance(left_shoulder, right_shoulder)
                        )
                        torso_length = torso_smoother.update(distance(shoulder_center, hip_center))
                        shoulder_text = f"Shoulder: {shoulder_width:.1f} px"
                        torso_text = f"Torso: {torso_length:.1f} px"
                        draw_measurement(
                            display, left_shoulder, right_shoulder, shoulder_text, 30
                        )
                        draw_measurement(
                            display, shoulder_center, hip_center, torso_text, 60
                        )
                        status = "Pose OK"
                        print(f"{shoulder_text} | {torso_text}", flush=True)
                    else:
                        shoulder_smoother.clear()
                        torso_smoother.clear()
                        print("Pose Lost (required shoulder/hip landmark missing)", flush=True)
                else:
                    shoulder_smoother.clear()
                    torso_smoother.clear()
                    print("Pose Lost", flush=True)

                color = (0, 255, 0) if status == "Pose OK" else (0, 0, 255)
                cv2.putText(display, status, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.imshow("Pose Pixel Distance Test", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
