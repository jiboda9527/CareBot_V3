"""Lightweight YOLO26 ONNX person detector for CareBot.

Uses ONNX Runtime directly, so the Raspberry Pi does not need the large
PyTorch/ultralytics runtime. It supports the default end-to-end YOLO26 ONNX
export and the traditional Ultralytics detection output as a compatibility
fallback.
"""

from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort


COCO_PERSON_CLASS_ID = 0


class YoloPersonDetector:
    """Detect COCO ``person`` objects from a YOLO26 ONNX model."""

    def __init__(
        self,
        model_path,
        confidence_threshold=0.45,
        nms_threshold=0.45,
        debug_inference=False,
    ):
        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO26 ONNX model was not found: {model_path}\n"
                "Place yolo26n.onnx there, or pass --yolo-model /path/to/model."
            )
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.debug_inference = debug_inference
        # Keep ONNX Runtime predictable on Raspberry Pi.  The default thread
        # pool can contend with camera/GUI work and produce unstable latency.
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = 1
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        self.input_height = int(input_shape[2]) if isinstance(input_shape[2], int) else 640
        self.input_width = int(input_shape[3]) if isinstance(input_shape[3], int) else 640
        self.last_inference_ms = 0.0
        print(f"YOLO26 person detector loaded: {model_path.name} ({self.input_width}x{self.input_height})")

    def _letterbox(self, frame):
        frame_height, frame_width = frame.shape[:2]
        scale = min(self.input_width / frame_width, self.input_height / frame_height)
        resized_width, resized_height = int(round(frame_width * scale)), int(round(frame_height * scale))
        resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        pad_x, pad_y = (self.input_width - resized_width) / 2, (self.input_height - resized_height) / 2
        padded = cv2.copyMakeBorder(
            resized, int(np.floor(pad_y)), int(np.ceil(pad_y)),
            int(np.floor(pad_x)), int(np.ceil(pad_x)), cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        return np.ascontiguousarray(blob, dtype=np.float32)[None] / 255.0, scale, pad_x, pad_y

    def _person_boxes(self, output, scale, pad_x, pad_y, frame_width, frame_height):
        prediction = np.squeeze(output)
        if prediction.ndim != 2:
            return []

        # Default YOLO26 ONNX export is end-to-end: [x1, y1, x2, y2,
        # confidence, class_id]. NMS is already part of that graph, so doing
        # NMS here again would only waste CPU and can suppress valid results.
        if prediction.shape[1] == 6:
            boxes, scores = [], []
            for x1, y1, x2, y2, score, class_id in prediction:
                if int(class_id) != COCO_PERSON_CLASS_ID or score < self.confidence_threshold:
                    continue
                x1 = (float(x1) - pad_x) / scale
                y1 = (float(y1) - pad_y) / scale
                x2 = (float(x2) - pad_x) / scale
                y2 = (float(y2) - pad_y) / scale
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(frame_width), x2), min(float(frame_height), y2)
                width, height = x2 - x1, y2 - y1
                if width >= 2.0 and height >= 2.0:
                    boxes.append([int(x1), int(y1), int(width), int(height)])
                    scores.append(float(score))
            return list(zip(boxes, scores))

        if prediction.shape[0] in (84, 85) and prediction.shape[0] < prediction.shape[1]:
            prediction = prediction.T
        if prediction.shape[1] < 5:
            return []

        boxes, scores = [], []
        for row in prediction:
            # Traditional output: x, y, w, h, class_0 ... class_79; person is class 0.
            score = float(row[4 + COCO_PERSON_CLASS_ID])
            if score < self.confidence_threshold:
                continue
            cx, cy, width, height = map(float, row[:4])
            x, y = (cx - width / 2 - pad_x) / scale, (cy - height / 2 - pad_y) / scale
            width, height = width / scale, height / scale
            x, y = max(0.0, min(x, frame_width - 1.0)), max(0.0, min(y, frame_height - 1.0))
            width, height = max(0.0, min(width, frame_width - x)), max(0.0, min(height, frame_height - y))
            if width >= 2.0 and height >= 2.0:
                boxes.append([int(x), int(y), int(width), int(height)])
                scores.append(score)
        if not boxes:
            return []
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        return [(boxes[int(index)], scores[int(index)]) for index in np.array(indices).reshape(-1)]

    def findPersons(self, frame):
        """Return the same tuple format as the existing TensorFlow SSD detector."""
        frame_height, frame_width = frame.shape[:2]
        blob, scale, pad_x, pad_y = self._letterbox(frame)
        inference_started = time.perf_counter()
        if self.debug_inference:
            print("[YOLO] before inference", flush=True)
        output = self.session.run(None, {self.input_name: blob})[0]
        if self.debug_inference:
            print("[YOLO] after inference", flush=True)
        self.last_inference_ms = (time.perf_counter() - inference_started) * 1000.0
        detections = self._person_boxes(output, scale, pad_x, pad_y, frame_width, frame_height)

        bboxs, bbox, center_x, largest_area = [], (0, 0, 0, 0), 0, 0
        for index, (item_bbox, score) in enumerate(detections):
            x, y, width, height = item_bbox
            bboxs.append([index, item_bbox, score])
            if width * height > largest_area:
                largest_area, bbox, center_x = width * height, item_bbox, x + width // 2
        if largest_area:
            self.fancyDraw(frame, bbox)
        return frame, bboxs, bbox, center_x

    @staticmethod
    def fancyDraw(frame, bbox):
        x, y, width, height = bbox
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
        return frame
