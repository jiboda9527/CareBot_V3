#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone fall detection demo for the elderly care robot.

The first version keeps the logic intentionally simple:
1. Detect people with the existing COCO SSD MobileNet model.
2. Treat a wide person bounding box as a possible fallen posture.
3. Trigger an alert if that state continues for 5 seconds.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import argparse
import subprocess
import time

import cv2

from line_alert import send_fall_alert


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
DETECTION_DIR = PROJECT_ROOT / "07.AI_Visual_Recognition" / "detection"

LABELS_PATH = DETECTION_DIR / "object_detection_coco.txt"
MODEL_PATH = DETECTION_DIR / "frozen_inference_graph.pb"
CONFIG_PATH = DETECTION_DIR / "ssd_mobilenet_v2_coco.txt"

PERSON_LABEL = "person"
DEFAULT_CONFIDENCE = 0.6
DEFAULT_FALL_ASPECT_RATIO = 1.35
DEFAULT_FALL_SECONDS = 5.0
DEFAULT_ALERT_COOLDOWN_SECONDS = 10.0
DEFAULT_CONFIRMATION_TEXT = "大丈夫ですか。転倒の可能性を検出しました。"
DEFAULT_CONFIRMATION_VOICE = "ja-JP-NanamiNeural"
VOICE_FILE = CURRENT_DIR / "fall_alert.mp3"
FALL_EVENT_DIR = CURRENT_DIR / "fall_events"
SAFE_RESPONSE_WORDS = (
    "ok",
    "okay",
    "fine",
    "yes",
    "はい",
    "はい大丈夫",
    "問題ない",
    "問題ありません",
    "i am ok",
    "i'm ok",
    "im ok",
    "大丈夫",
    "没事",
    "沒事",
    "没关系",
    "無事",
)
HELP_RESPONSE_WORDS = (
    "help",
    "call",
    "ambulance",
    "emergency",
    "not ok",
    "not okay",
    "cannot move",
    "救命",
    "助け",
    "助けて",
    "痛",
    "痛い",
    "疼",
    "不行",
    "動けない",
    "動けません",
    "だめ",
)


@dataclass
class PersonDetection:
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]

    @property
    def area(self):
        _, _, width, height = self.bbox
        return width * height


class FallDetector:
    def __init__(
        self,
        confidence_threshold=DEFAULT_CONFIDENCE,
        fall_aspect_ratio=DEFAULT_FALL_ASPECT_RATIO,
        fall_seconds=DEFAULT_FALL_SECONDS,
        voice_enabled=True,
        line_alert_enabled=True,
        save_event_image=True,
        confirm_before_line=False,
        confirmation_text=DEFAULT_CONFIRMATION_TEXT,
        confirmation_voice=DEFAULT_CONFIRMATION_VOICE,
        load_model=True,
    ):
        self.confidence_threshold = confidence_threshold
        self.fall_aspect_ratio = fall_aspect_ratio
        self.fall_seconds = fall_seconds
        self.voice_enabled = voice_enabled
        self.line_alert_enabled = line_alert_enabled
        self.save_event_image_enabled = save_event_image
        self.confirm_before_line = confirm_before_line
        self.confirmation_text = confirmation_text
        self.confirmation_voice = confirmation_voice
        self.class_names = []
        self.model = None
        if load_model:
            self.class_names = self._load_class_names()
            self.model = cv2.dnn.readNet(
                model=str(MODEL_PATH),
                config=str(CONFIG_PATH),
                framework="TensorFlow",
            )
        self.fall_started_at = None
        self.alerted = False
        self.last_alert_at = 0.0

    def _load_class_names(self):
        if not LABELS_PATH.exists():
            raise FileNotFoundError(f"Missing COCO labels: {LABELS_PATH}")

        with LABELS_PATH.open("r", encoding="utf-8") as labels_file:
            return [line.strip() for line in labels_file if line.strip()]

    def detect_people(self, frame):
        if self.model is None:
            raise RuntimeError("FallDetector model is not loaded.")

        image_height, image_width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image=frame,
            size=(300, 300),
            mean=(104, 117, 123),
            swapRB=True,
        )

        self.model.setInput(blob)
        output = self.model.forward()
        people = []

        for detection in output[0, 0, :, :]:
            confidence = float(detection[2])
            if confidence < self.confidence_threshold:
                continue

            class_id = int(detection[1])
            class_index = class_id - 1
            if class_index < 0 or class_index >= len(self.class_names):
                continue

            label = self.class_names[class_index]
            if label != PERSON_LABEL:
                continue

            x_min = int(detection[3] * image_width)
            y_min = int(detection[4] * image_height)
            x_max = int(detection[5] * image_width)
            y_max = int(detection[6] * image_height)

            x_min = max(0, min(x_min, image_width - 1))
            y_min = max(0, min(y_min, image_height - 1))
            x_max = max(0, min(x_max, image_width - 1))
            y_max = max(0, min(y_max, image_height - 1))

            width = max(0, x_max - x_min)
            height = max(0, y_max - y_min)
            if width == 0 or height == 0:
                continue

            people.append(
                PersonDetection(
                    label=label,
                    confidence=confidence,
                    bbox=(x_min, y_min, width, height),
                )
            )

        return people

    def is_fallen(self, person):
        _, _, width, height = person.bbox
        aspect_ratio = width / max(height, 1)
        return aspect_ratio >= self.fall_aspect_ratio

    def update_fall_state(self, fallen_now, frame=None, person=None):
        now = time.monotonic()

        if fallen_now:
            if self.fall_started_at is None:
                self.fall_started_at = now

            fallen_duration = now - self.fall_started_at
            if fallen_duration >= self.fall_seconds and not self.alerted:
                self.trigger_alert(frame, person, fallen_duration)
                self.alerted = True

            return fallen_duration

        self.fall_started_at = None
        self.alerted = False
        return 0.0

    def trigger_alert(self, frame=None, person=None, fallen_duration=0.0):
        now = time.monotonic()
        if now - self.last_alert_at < DEFAULT_ALERT_COOLDOWN_SECONDS:
            return

        self.last_alert_at = now
        print("FALL DETECTED")
        self.save_fall_event_image(frame, person, fallen_duration)
        self.ask_safety_confirmation()

        response_status = None
        response_text = None
        if self.confirm_before_line:
            response_status, response_text = self.listen_for_safety_response()
            if response_status == "safe":
                print(f"Safety response received: {response_text}")
                print("LINE alert skipped.")
                return
            if response_status == "help":
                print(f"Help response received: {response_text}")
                print("Sending LINE alert.")
            else:
                print("No clear safety response. Sending LINE alert.")

        self.send_emergency_notification(response_status, response_text)

    def save_fall_event_image(self, frame, person, fallen_duration):
        if not self.save_event_image_enabled or frame is None:
            return None

        FALL_EVENT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_path = FALL_EVENT_DIR / f"fall_event_{timestamp}.jpg"
        event_frame = frame.copy()

        if person is not None:
            event_frame = self.draw_detection(
                event_frame,
                person,
                True,
                fallen_duration,
            )

        cv2.putText(
            event_frame,
            f"FALL DETECTED {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            (20, event_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        if cv2.imwrite(str(image_path), event_frame):
            print(f"Fall event image saved: {image_path}")
            return image_path

        print(f"Failed to save fall event image: {image_path}")
        return None

    def send_emergency_notification(self, response_status=None, response_text=None):
        if not self.line_alert_enabled:
            return

        send_fall_alert(response_status=response_status, response_text=response_text)

    def ask_safety_confirmation(self):
        if not self.voice_enabled:
            return

        try:
            subprocess.run(
                [
                    "edge-tts",
                    "--voice",
                    self.confirmation_voice,
                    "--text",
                    self.confirmation_text,
                    "--write-media",
                    str(VOICE_FILE),
                ],
                check=True,
            )
            subprocess.run(["mpg123", "-q", str(VOICE_FILE)], check=False)
        except FileNotFoundError as exc:
            print(f"Voice confirmation skipped, command not found: {exc.filename}")
        except subprocess.CalledProcessError as exc:
            print(f"Voice confirmation failed: {exc}")

    def listen_for_safety_response(self):
        try:
            from speech_control import listen
        except Exception as exc:
            print(f"Safety response listening unavailable: {exc}")
            return "unknown", None

        print("Listening for safety response...")
        result = listen()
        if isinstance(result, tuple):
            text = result[0]
        else:
            text = result

        text = (text or "").strip()
        if not text:
            return "unknown", None

        print(f"Safety response: {text}")
        return self.classify_safety_response(text), text

    def classify_safety_response(self, text):
        normalized = text.lower().replace(" ", "")

        for word in HELP_RESPONSE_WORDS:
            if word.lower().replace(" ", "") in normalized:
                return "help"

        for word in SAFE_RESPONSE_WORDS:
            if word.lower().replace(" ", "") in normalized:
                return "safe"

        return "unknown"

    def draw_detection(self, frame, person, fallen_now, fallen_duration):
        x, y, width, height = person.bbox
        color = (0, 0, 255) if fallen_now else (0, 255, 0)
        status = "FALLING?" if fallen_now else "PERSON"
        aspect_ratio = width / max(height, 1)

        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
        cv2.putText(
            frame,
            f"{status} {person.confidence:.2f} ar={aspect_ratio:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        if fallen_now:
            cv2.putText(
                frame,
                f"fallen for {fallen_duration:.1f}s",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )

        return frame

    def process_frame(self, frame):
        people = self.detect_people(frame)
        main_person = max(people, key=lambda item: item.area, default=None)

        if main_person is None:
            fallen_duration = self.update_fall_state(False)
            return frame, None, False, fallen_duration

        fallen_now = self.is_fallen(main_person)
        fallen_duration = self.update_fall_state(fallen_now, frame, main_person)
        frame = self.draw_detection(frame, main_person, fallen_now, fallen_duration)
        return frame, main_person, fallen_now, fallen_duration


def open_camera(camera_index, width, height):
    capture = cv2.VideoCapture(camera_index)
    cv_edition = cv2.__version__

    if cv_edition[0] == "3":
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"XVID"))
    else:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def run_demo(args):
    detector = FallDetector(
        confidence_threshold=args.confidence,
        fall_aspect_ratio=args.aspect_ratio,
        fall_seconds=args.seconds,
        voice_enabled=not args.no_voice,
        line_alert_enabled=not args.no_line,
        save_event_image=not args.no_save_image,
        confirm_before_line=args.confirm_before_line,
        confirmation_text=args.confirmation_text,
        confirmation_voice=args.voice,
    )
    capture = open_camera(args.camera, args.width, args.height)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera}")

    print("Fall detection started. Press q to exit.")
    if args.no_window:
        print("No-window mode: press Ctrl+C to exit.")
    print(
        "Alert rule: person bbox width / height >= "
        f"{args.aspect_ratio} for {args.seconds} seconds."
    )

    last_status_at = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Camera frame read failed")
                time.sleep(0.1)
                continue

            frame, person, fallen_now, fallen_duration = detector.process_frame(frame)

            if args.no_window:
                now = time.monotonic()
                if now - last_status_at >= 1.0:
                    if person is None:
                        print("status: no person detected")
                    else:
                        _, _, box_width, box_height = person.bbox
                        aspect_ratio = box_width / max(box_height, 1)
                        print(
                            "status:",
                            f"person={person.confidence:.2f}",
                            f"aspect_ratio={aspect_ratio:.2f}",
                            f"fallen={fallen_now}",
                            f"duration={fallen_duration:.1f}s",
                        )
                    last_status_at = now

            if not args.no_window:
                cv2.imshow("fall_detection", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    break

    except KeyboardInterrupt:
        print("\nFall detection stopped.")

    finally:
        capture.release()
        cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect a possible elderly fall with camera bounding boxes."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--confidence", type=float, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--aspect-ratio", type=float, default=DEFAULT_FALL_ASPECT_RATIO)
    parser.add_argument("--seconds", type=float, default=DEFAULT_FALL_SECONDS)
    parser.add_argument("--voice", default=DEFAULT_CONFIRMATION_VOICE)
    parser.add_argument("--confirmation-text", default=DEFAULT_CONFIRMATION_TEXT)
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Do not play the safety confirmation voice after a fall alert.",
    )
    parser.add_argument(
        "--no-line",
        action="store_true",
        help="Do not send LINE emergency notifications after a fall alert.",
    )
    parser.add_argument(
        "--no-save-image",
        action="store_true",
        help="Do not save a fall event image when a fall alert is triggered.",
    )
    parser.add_argument(
        "--confirm-before-line",
        action="store_true",
        help="Ask and listen once before sending LINE; skip LINE if the user says they are okay.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without cv2.imshow, useful for SSH-only testing.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_demo(parse_args())
