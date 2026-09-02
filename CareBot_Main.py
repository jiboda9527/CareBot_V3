#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CareBot V3 main integration entry.

Phase 1 goal:
- keep existing demos intact
- run one SSD person detection per frame
- coordinate follow, fall check, voice confirmation, and LINE alert
"""

import argparse
import os
import sys
import time
from collections import deque
from enum import Enum

import cv2


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

import Track_SSD_Person_Follow as person_follow
from fall_detection import FallDetector, PersonDetection
from line_alert import send_fall_alert
from proximity_guard import PersonProximityGuard


class CareBotState(Enum):
    IDLE = "IDLE"
    SEARCH_PERSON = "SEARCH_PERSON"
    FOLLOW_PERSON = "FOLLOW_PERSON"
    HEALTH_CHECK = "HEALTH_CHECK"
    EMERGENCY = "EMERGENCY"


# More responsive tracking.  The previous conservative values accumulated
# noticeable delay before the chassis was allowed to react.
SMOOTH_ALPHA = 0.55
CHASSIS_UPDATE_INTERVAL = 0.12
MOVE_CONFIRM_FRAMES = 1
STOP_CONFIRM_FRAMES = 1

TURN_CENTER_DEAD_ZONE = 110
TURN_RELEASE_DEAD_ZONE = 60
TURN_FORCE_DEAD_ZONE = 190
TURN_LIMIT_HOLD_SECONDS = 0.25
TURN_ESCALATE_SECONDS = 2.5
TURN_SLOW_SPEED = 2
TURN_NORMAL_SPEED = 3
TARGET_CONFIRM_FRAMES = 1
# The chassis must wait for the slower pan servo to settle before driving.
FORWARD_CENTER_DEAD_ZONE = 35
FORWARD_RESUME_PAN_MARGIN = 20
FORWARD_CENTER_CONFIRM_SECONDS = 0.45

AREA_TARGET = person_follow.area_center
# Hysteresis limits for distance control.  The robot stops inside the broad
# middle band instead of reversing as soon as one delayed frame crosses the
# target area.
AREA_APPROACH_START = 110000
AREA_APPROACH_STOP = 145000
AREA_RETREAT_START = 245000
AREA_RETREAT_STOP = 215000
FORWARD_SLOW_SPEED = 3
FORWARD_NORMAL_SPEED = 4
BACKWARD_SLOW_SPEED = -3
BACKWARD_NORMAL_SPEED = -4

LOST_WAIT_SECONDS = 1.5
# A detector can miss a person for one frame even while the person is still in
# view.  Do not cancel a newly issued chassis command for such a brief miss.
PERSON_LOSS_STOP_GRACE_SECONDS = 0.70
LOST_SCAN_STEP = 2.0
EMERGENCY_COOLDOWN_SECONDS = 6.0
MAX_CONSECUTIVE_CAMERA_FAILURES = 5


class CareBotMain:
    def __init__(
        self,
        no_motor=False,
        no_window=False,
        detector_type="yolo",
        yolo_model=None,
        debug_inference=False,
        no_inference=False,
    ):
        self.no_motor = no_motor
        self.no_window = no_window
        self.state = CareBotState.IDLE
        if detector_type == "yolo":
            model_path = yolo_model or os.path.join(CURRENT_DIR, "models", "yolo26n.onnx")
            self.detector = person_follow.YoloPersonDetector(
                model_path,
                confidence_threshold=0.30,
                debug_inference=debug_inference,
            )
        else:
            self.detector = person_follow.PersonDetector()
        self.detector_type = detector_type
        self.no_inference = no_inference
        self.frame_times_ms = deque(maxlen=30)
        self.fall_helper = FallDetector(
            voice_enabled=True,
            line_alert_enabled=False,
            save_event_image=True,
            load_model=False,
        )
        self.pan_angle = float(person_follow.PAN_CENTER)
        self.tilt_angle = float(person_follow.TILT_CENTER)
        self.last_state_print_at = 0.0
        self.smooth_center_x = None
        self.smooth_area = None
        self.distance_motion = 0
        self.pending_motion = (0, 0)
        self.pending_motion_count = 0
        self.active_motion = (0, 0)
        self.last_chassis_update_at = 0.0
        self.lost_person_since = None
        self.lost_scan_direction = 1
        self.emergency_cooldown_until = 0.0
        self.pan_limit_since = None
        self.pan_limit_direction = 0
        self.chassis_turning = False
        self.chassis_turn_started_at = 0.0
        self.target_seen_frames = 0
        self.waiting_for_pan_center_after_turn = False
        self.forward_centered_since = None
        self.proximity_guard = PersonProximityGuard()

    def set_state(self, state):
        if self.state != state:
            print(f"[STATE] {self.state.value} -> {state.value}")
            self.state = state

    def reset_robot_pose(self):
        if not self.no_motor:
            person_follow.stop_robot()
            person_follow.Facebot.Ctrl_Servo(1, int(self.pan_angle))
            person_follow.Facebot.Ctrl_Servo(2, int(self.tilt_angle))
        person_follow.reset_tracking_pid()

    def stop_robot(self):
        if not self.no_motor:
            person_follow.stop_robot()
        person_follow.reset_tracking_pid()
        self.active_motion = (0, 0)
        self.pending_motion = (0, 0)
        self.pending_motion_count = 0
        self.reset_chassis_turn_state()
        self.waiting_for_pan_center_after_turn = False
        self.forward_centered_since = None

    def reset_fall_detection(self):
        person_follow.fall_frame_count = 0
        person_follow.is_fall = False

    def reset_tracking_smoothing(self):
        self.smooth_center_x = None
        self.smooth_area = None
        self.distance_motion = 0

    def reset_chassis_turn_state(self):
        self.pan_limit_since = None
        self.pan_limit_direction = 0
        self.chassis_turning = False
        self.chassis_turn_started_at = 0.0

    def smooth_target(self, center_x, area):
        if self.smooth_center_x is None:
            self.smooth_center_x = float(center_x)
            self.smooth_area = float(area)
        else:
            self.smooth_center_x = (
                (1.0 - SMOOTH_ALPHA) * self.smooth_center_x
                + SMOOTH_ALPHA * float(center_x)
            )
            self.smooth_area = (
                (1.0 - SMOOTH_ALPHA) * self.smooth_area
                + SMOOTH_ALPHA * float(area)
            )

        return int(self.smooth_center_x), int(self.smooth_area)

    def choose_forward_speed(self, bbox_area):
        """Return a stable distance command using area hysteresis."""
        if self.distance_motion > 0:
            if bbox_area >= AREA_APPROACH_STOP:
                self.distance_motion = 0
        elif self.distance_motion < 0:
            if bbox_area <= AREA_RETREAT_STOP:
                self.distance_motion = 0
        elif bbox_area <= AREA_APPROACH_START:
            self.distance_motion = FORWARD_NORMAL_SPEED
        elif bbox_area >= AREA_RETREAT_START:
            self.distance_motion = BACKWARD_SLOW_SPEED

        # Slow down before the stop threshold, while still approaching.
        if self.distance_motion > 0 and bbox_area >= AREA_APPROACH_START - 30000:
            return FORWARD_SLOW_SPEED
        return self.distance_motion

    def update_chassis_turn_gate(
        self,
        pan_error,
        pan_near_left_edge,
        pan_near_right_edge,
    ):
        if self.target_seen_frames < TARGET_CONFIRM_FRAMES:
            self.reset_chassis_turn_state()
            return False

        if abs(pan_error) < TURN_RELEASE_DEAD_ZONE:
            self.reset_chassis_turn_state()
            return False

        if pan_error > TURN_CENTER_DEAD_ZONE and pan_near_left_edge:
            needed_direction = 1
        elif pan_error < -TURN_CENTER_DEAD_ZONE and pan_near_right_edge:
            needed_direction = -1
        else:
            self.reset_chassis_turn_state()
            return False

        now = time.monotonic()
        if needed_direction != self.pan_limit_direction:
            self.pan_limit_direction = needed_direction
            self.pan_limit_since = now
            self.chassis_turning = False
            self.chassis_turn_started_at = 0.0
            return False

        if self.pan_limit_since is None:
            self.pan_limit_since = now
            return False

        if not self.chassis_turning:
            if now - self.pan_limit_since < TURN_LIMIT_HOLD_SECONDS:
                return False
            self.chassis_turning = True
            self.chassis_turn_started_at = now

        return True

    def choose_turn_value(self, pan_error):
        if abs(pan_error) < TURN_CENTER_DEAD_ZONE:
            return 0

        turn_elapsed = time.monotonic() - self.chassis_turn_started_at
        turn_speed = (
            TURN_NORMAL_SPEED
            if (
                turn_elapsed >= TURN_ESCALATE_SECONDS
                and abs(pan_error) >= TURN_FORCE_DEAD_ZONE
            )
            else TURN_SLOW_SPEED
        )
        return turn_speed if pan_error > 0 else -turn_speed

    def allow_forward_motion_after_turn(self):
        if not self.waiting_for_pan_center_after_turn:
            return True

        pan_center_error = abs(self.pan_angle - person_follow.PAN_CENTER)
        if pan_center_error <= FORWARD_RESUME_PAN_MARGIN:
            self.waiting_for_pan_center_after_turn = False
            return True

        return False

    def allow_forward_motion(self, pan_error):
        centered = (
            self.target_seen_frames >= TARGET_CONFIRM_FRAMES
            and abs(pan_error) <= FORWARD_CENTER_DEAD_ZONE
            and self.allow_forward_motion_after_turn()
        )
        if not centered:
            self.forward_centered_since = None
            return False

        now = time.monotonic()
        if self.forward_centered_since is None:
            self.forward_centered_since = now
            return False
        return now - self.forward_centered_since >= FORWARD_CENTER_CONFIRM_SECONDS

    def command_chassis(self, speed_value, turn_value):
        if turn_value != 0:
            speed_value = 0
            if self.active_motion[0] != 0:
                self.active_motion = (0, 0)
                self.pending_motion = (0, turn_value)
                self.pending_motion_count = 1
                self.last_chassis_update_at = time.monotonic()
                if not self.no_motor:
                    person_follow.stop_robot()
                return self.active_motion

        desired_motion = (speed_value, turn_value)
        if desired_motion == self.pending_motion:
            self.pending_motion_count += 1
        else:
            self.pending_motion = desired_motion
            self.pending_motion_count = 1

        required_frames = STOP_CONFIRM_FRAMES if desired_motion == (0, 0) else MOVE_CONFIRM_FRAMES
        if self.pending_motion_count < required_frames:
            speed_value, turn_value = self.active_motion
        else:
            speed_value, turn_value = desired_motion

        now = time.monotonic()
        if (
            desired_motion != (0, 0)
            and now - self.last_chassis_update_at < CHASSIS_UPDATE_INTERVAL
        ):
            speed_value, turn_value = self.active_motion

        if (speed_value, turn_value) != self.active_motion:
            self.last_chassis_update_at = now
            self.active_motion = (speed_value, turn_value)

        if not self.no_motor:
            if self.active_motion == (0, 0):
                person_follow.stop_robot()
            else:
                # turn_value already uses the motor driver's left/right sign.
                person_follow.control_motor_speed(
                    self.active_motion[0],
                    self.active_motion[1],
                )

        return self.active_motion

    def handle_lost_person(self):
        now = time.monotonic()
        if self.lost_person_since is None:
            self.lost_person_since = now
            return False

        if now - self.lost_person_since < PERSON_LOSS_STOP_GRACE_SECONDS:
            return False

        self.stop_robot()
        self.reset_fall_detection()
        self.reset_tracking_smoothing()
        self.target_seen_frames = 0

        if now - self.lost_person_since < LOST_WAIT_SECONDS:
            return True

        self.pan_angle += self.lost_scan_direction * LOST_SCAN_STEP
        if self.pan_angle >= person_follow.PAN_MAX:
            self.pan_angle = person_follow.PAN_MAX
            self.lost_scan_direction = -1
        elif self.pan_angle <= person_follow.PAN_MIN:
            self.pan_angle = person_follow.PAN_MIN
            self.lost_scan_direction = 1

        if not self.no_motor:
            person_follow.Facebot.Ctrl_Servo(1, int(round(self.pan_angle)))
        return True

    def follow_target(self, frame, bbox, center_x, image_width, image_height):
        x, y, w, h = bbox
        proximity = self.proximity_guard.update(frame, bbox)
        raw_area = person_follow.limit_max_vlaue(w * h, 2000, 350000)
        smooth_center_x, smooth_area = self.smooth_target(center_x, raw_area)
        person_center_y = y + h // 2
        target_x = image_width // 2
        pan_error = smooth_center_x - target_x
        tilt_error = person_center_y - image_height // 2

        if abs(pan_error) > person_follow.PAN_DEAD_ZONE:
            pan_step = person_follow.clamp(
                pan_error * person_follow.PAN_GAIN,
                -person_follow.PAN_MAX_STEP,
                person_follow.PAN_MAX_STEP,
            )
            self.pan_angle = person_follow.clamp(
                self.pan_angle - pan_step,
                person_follow.PAN_MIN,
                person_follow.PAN_MAX,
            )
            if not self.no_motor:
                person_follow.Facebot.Ctrl_Servo(1, int(round(self.pan_angle)))

        if abs(tilt_error) > person_follow.TILT_DEAD_ZONE:
            tilt_step = person_follow.clamp(
                tilt_error * person_follow.TILT_GAIN,
                -person_follow.TILT_MAX_STEP,
                person_follow.TILT_MAX_STEP,
            )
            self.tilt_angle = person_follow.clamp(
                self.tilt_angle - tilt_step,
                person_follow.TILT_MIN,
                person_follow.TILT_MAX,
            )
            if not self.no_motor:
                person_follow.Facebot.Ctrl_Servo(2, int(round(self.tilt_angle)))

        pan_near_left_edge = (
            self.pan_angle <= person_follow.PAN_MIN + person_follow.PAN_CHASSIS_MARGIN
        )
        pan_near_right_edge = (
            self.pan_angle >= person_follow.PAN_MAX - person_follow.PAN_CHASSIS_MARGIN
        )
        chassis_turn_allowed = self.update_chassis_turn_gate(
            pan_error,
            pan_near_left_edge,
            pan_near_right_edge,
        )

        if chassis_turn_allowed:
            turn_value = self.choose_turn_value(pan_error)
            if turn_value != 0:
                self.waiting_for_pan_center_after_turn = True
        else:
            turn_value = 0
            person_follow.reset_pid(person_follow.direction_pid)

        if proximity.too_close:
            # A partial person or a box extending beyond the view must never
            # be treated as a normal target merely because its area is small.
            turn_value = 0
            speed_value = BACKWARD_SLOW_SPEED
            self.distance_motion = 0
        else:
            forward_allowed = turn_value == 0 and self.allow_forward_motion(pan_error)
            speed_value = (
                self.choose_forward_speed(smooth_area)
                if forward_allowed
                else 0
            )
        active_speed, active_turn = self.command_chassis(speed_value, turn_value)

        cv2.circle(frame, (target_x, image_height // 2), 10, (0, 0, 255), -1)
        cv2.line(
            frame,
            (target_x, image_height // 2),
            (smooth_center_x, person_center_y),
            (0, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"state={self.state.value}",
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        proximity_text = (
            f"TOO CLOSE: {proximity.reason}"
            if proximity.too_close
            else f"body h{proximity.head_points} t{proximity.torso_points} e{proximity.extremity_points}"
        )
        cv2.putText(
            frame,
            proximity_text,
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255) if proximity.too_close else (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            f"motor v={active_speed} turn={active_turn}",
            (20, 132),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        # Per-frame console I/O is costly on a Raspberry Pi and adds lag.

    def build_person_detection(self, bboxs, bbox):
        confidence = 0.0
        for _, item_bbox, score in bboxs:
            if item_bbox == bbox:
                confidence = float(score)
                break
        return PersonDetection("person", confidence, bbox)

    def run_health_check(self):
        self.stop_robot()
        self.fall_helper.ask_safety_confirmation()
        response_status, response_text = self.fall_helper.listen_for_safety_response()

        if response_status == "safe":
            print(f"Safety confirmed: {response_text}")
            return "safe", response_text

        print(f"Emergency response status: {response_status}, text={response_text}")
        return response_status, response_text

    def run_emergency(self, frame, person, response_status, response_text):
        self.stop_robot()
        self.fall_helper.save_fall_event_image(frame, person, fallen_duration=0.0)
        send_fall_alert(response_status=response_status, response_text=response_text)

    def run(self):
        capture = person_follow.open_camera()
        if capture is None:
            print("No camera found")
            return

        # Keep the camera's negotiated V4L2 format.  Forcing 640x480 after the
        # device has started streaming makes some UVC cameras stop delivering
        # frames.  YOLO performs its own resize to the model input size.
        image_width = 0
        image_height = 0

        self.reset_robot_pose()
        self.set_state(CareBotState.SEARCH_PERSON)
        last_person = None
        camera_failure_count = 0
        reported_camera_size = None

        try:
            while True:
                frame_started_at = time.perf_counter()
                ok, frame = capture.read()
                if not ok or frame is None:
                    camera_failure_count += 1
                    print(
                        "Camera frame read failed "
                        f"({camera_failure_count}/{MAX_CONSECUTIVE_CAMERA_FAILURES})"
                    )
                    self.stop_robot()
                    if camera_failure_count >= MAX_CONSECUTIVE_CAMERA_FAILURES:
                        print("Camera unavailable; stopping safely. Check the USB camera and restart.")
                        break
                    time.sleep(0.1)
                    continue

                camera_failure_count = 0
                image_height, image_width = frame.shape[:2]
                if reported_camera_size != (image_width, image_height):
                    reported_camera_size = (image_width, image_height)
                    print(f"Camera stream: {image_width}x{image_height}")

                if self.no_inference:
                    bboxs, bbox, center_x = [], (0, 0, 0, 0), 0
                else:
                    frame, bboxs, bbox, center_x = self.detector.findPersons(frame)
                frame_ms = (time.perf_counter() - frame_started_at) * 1000.0
                self.frame_times_ms.append(frame_ms)
                avg_frame_ms = sum(self.frame_times_ms) / len(self.frame_times_ms)
                fps = 1000.0 / avg_frame_ms if avg_frame_ms else 0.0
                inference_ms = getattr(self.detector, "last_inference_ms", 0.0)

                if not bboxs:
                    if self.handle_lost_person():
                        self.set_state(CareBotState.SEARCH_PERSON)
                else:
                    self.lost_person_since = None
                    self.target_seen_frames += 1

                    if time.monotonic() < self.emergency_cooldown_until:
                        self.stop_robot()
                        self.reset_tracking_smoothing()
                        self.reset_fall_detection()
                        self.set_state(CareBotState.SEARCH_PERSON)
                        if not self.no_window:
                            cv2.imshow("CareBot V3", frame)
                            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                                break
                        continue

                    last_person = self.build_person_detection(bboxs, bbox)
                    self.set_state(CareBotState.FOLLOW_PERSON)
                    self.follow_target(frame, bbox, center_x, image_width, image_height)

                    x, y, w, h = bbox
                    if person_follow.fall_detect(w, h):
                        cv2.putText(
                            frame,
                            "FALL DETECTED",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3,
                        )
                        self.set_state(CareBotState.HEALTH_CHECK)
                        response_status, response_text = self.run_health_check()

                        if response_status == "safe":
                            self.reset_fall_detection()
                            self.set_state(CareBotState.FOLLOW_PERSON)
                        else:
                            self.set_state(CareBotState.EMERGENCY)
                            self.run_emergency(
                                frame,
                                last_person,
                                response_status,
                                response_text,
                            )
                            self.emergency_cooldown_until = (
                                time.monotonic() + EMERGENCY_COOLDOWN_SECONDS
                            )
                            self.reset_fall_detection()
                            self.reset_tracking_smoothing()
                            self.set_state(CareBotState.IDLE)
                            time.sleep(1.0)
                            self.set_state(CareBotState.SEARCH_PERSON)

                if not self.no_window:
                    cv2.putText(
                        frame,
                        f"{self.detector_type.upper()}  infer {inference_ms:.1f} ms  {fps:.1f} FPS",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    )
                    cv2.putText(
                        frame,
                        f"loop {frame_ms:.1f} ms",
                        (20, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2,
                    )
                    cv2.imshow("CareBot V3", frame)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                        break

        except KeyboardInterrupt:
            print("\nCareBot stopped by user.")
        finally:
            self.stop_robot()
            self.proximity_guard.close()
            if not self.no_motor:
                person_follow.Facebot.Ctrl_Servo(1, 90)
                person_follow.Facebot.Ctrl_Servo(2, 25)
            capture.release()
            cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="CareBot V3 integrated main.")
    parser.add_argument(
        "--no-motor",
        action="store_true",
        help="Run detection/state machine without moving the chassis.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without cv2.imshow, useful over SSH.",
    )
    parser.add_argument(
        "--detector",
        choices=("yolo", "ssd"),
        default="yolo",
        help="Person detector to use. YOLO is default; SSD remains as fallback.",
    )
    parser.add_argument(
        "--yolo-model",
        default=None,
        help="Path to YOLO26n ONNX (default: models/yolo26n.onnx).",
    )
    parser.add_argument(
        "--debug-inference",
        action="store_true",
        help="Print markers immediately before and after every YOLO inference.",
    )
    parser.add_argument(
        "--no-inference",
        action="store_true",
        help="Skip detector inference for camera-only diagnostics.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    CareBotMain(
        no_motor=args.no_motor,
        no_window=args.no_window,
        detector_type=args.detector,
        yolo_model=args.yolo_model,
        debug_inference=args.debug_inference,
        no_inference=args.no_inference,
    ).run()
