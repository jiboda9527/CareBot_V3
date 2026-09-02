"""Near-distance safety checks for CareBot person following.

YOLO's ``person`` box says that a person was detected, but it cannot say
whether the whole person fits in the camera view.  This module combines box
clipping with MediaPipe Pose landmark visibility to identify a person who is
too close to the camera.
"""

from dataclasses import dataclass

import cv2

try:
    import mediapipe as mp
except ImportError:  # Keep normal YOLO following available if Pose is absent.
    mp = None


POSE_VISIBILITY_THRESHOLD = 0.50
EDGE_MARGIN_RATIO = 0.035
OVERCLOSE_CONFIRM_FRAMES = 2


@dataclass
class ProximityResult:
    too_close: bool = False
    reason: str = "normal"
    edge_contacts: int = 0
    head_points: int = 0
    torso_points: int = 0
    extremity_points: int = 0


class PersonProximityGuard:
    """Classify whether the tracked person is too close to follow safely."""

    # MediaPipe Pose landmark indexes.  Groups deliberately include both sides
    # so a person partly hidden by furniture can still be classified safely.
    HEAD = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
    TORSO = (11, 12, 23, 24)
    EXTREMITIES = (13, 14, 15, 16, 25, 26, 27, 28, 29, 30, 31, 32)

    def __init__(self, enabled=True):
        self.pose = None
        self.overclose_frames = 0
        self.available = bool(enabled and mp is not None)
        if self.available:
            try:
                self.pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=0,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
            except Exception as error:
                # Some MediaPipe packages fetch the lightweight TFLite model
                # on first use.  A robot must retain the bbox safety rule when
                # that optional model is unavailable (for example, offline).
                self.available = False
                print(f"Pose proximity check unavailable: {error}")

    @staticmethod
    def _edge_contacts(bbox, image_width, image_height):
        x, y, width, height = bbox
        margin_x = max(1, int(image_width * EDGE_MARGIN_RATIO))
        margin_y = max(1, int(image_height * EDGE_MARGIN_RATIO))
        return sum(
            (
                x <= margin_x,
                y <= margin_y,
                x + width >= image_width - margin_x,
                y + height >= image_height - margin_y,
            )
        )

    def _visible_pose_groups(self, frame):
        if self.pose is None:
            return 0, 0, 0

        result = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.pose_landmarks:
            return 0, 0, 0

        landmarks = result.pose_landmarks.landmark

        def count_visible(indexes):
            return sum(
                landmark.visibility >= POSE_VISIBILITY_THRESHOLD
                and 0.0 <= landmark.x <= 1.0
                and 0.0 <= landmark.y <= 1.0
                for index, landmark in enumerate(landmarks)
                if index in indexes
            )

        return (
            count_visible(self.HEAD),
            count_visible(self.TORSO),
            count_visible(self.EXTREMITIES),
        )

    def evaluate(self, frame, bbox):
        """Return the current close-range evidence without temporal filtering."""
        image_height, image_width = frame.shape[:2]
        x, y, width, height = bbox
        edge_contacts = self._edge_contacts(bbox, image_width, image_height)
        area_ratio = (width * height) / float(image_width * image_height)
        width_ratio = width / float(image_width)
        height_ratio = height / float(image_height)

        # A large YOLO box clipped by two sides of the frame is reliable
        # evidence that the person extends beyond the camera view.  One-side
        # clipping needs a much larger box so normal framing near an edge is
        # not treated as an emergency retreat.
        bbox_too_close = (
            edge_contacts >= 2
            and (area_ratio >= 0.22 or width_ratio >= 0.62 or height_ratio >= 0.72)
        ) or (
            edge_contacts >= 1
            and (area_ratio >= 0.48 or width_ratio >= 0.82 or height_ratio >= 0.90)
        )

        head_points, torso_points, extremity_points = self._visible_pose_groups(frame)
        pose_too_close = (
            extremity_points >= 1 and head_points == 0 and torso_points < 2
        )

        if bbox_too_close:
            reason = "frame-edge"
        elif pose_too_close:
            reason = "partial-body"
        else:
            reason = "normal"
        return ProximityResult(
            too_close=bbox_too_close or pose_too_close,
            reason=reason,
            edge_contacts=edge_contacts,
            head_points=head_points,
            torso_points=torso_points,
            extremity_points=extremity_points,
        )

    def update(self, frame, bbox):
        """Apply a short confirmation window before commanding a retreat."""
        result = self.evaluate(frame, bbox)
        if result.too_close:
            self.overclose_frames += 1
        else:
            self.overclose_frames = 0
        result.too_close = self.overclose_frames >= OVERCLOSE_CONFIRM_FRAMES
        return result

    def close(self):
        if self.pose is not None:
            self.pose.close()
