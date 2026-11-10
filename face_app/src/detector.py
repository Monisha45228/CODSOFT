"""
detector.py
-----------
Face detection module. Supports two interchangeable backends:

1. "haar"  - Classic Viola-Jones Haar cascade classifier (fast, CPU-only,
             ships with OpenCV, weaker on angled/occluded faces).
2. "dnn"   - Deep-learning SSD face detector (ResNet-10 backbone,
             res10_300x300_ssd_iter_140000.caffemodel). Much more robust
             to pose, lighting, and partial occlusion; still fast enough
             for real-time video on CPU.

Usage:
    detector = FaceDetector(method="dnn")
    faces = detector.detect(frame)   # -> list of (x, y, w, h)
"""

from pathlib import Path
import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class FaceDetector:
    def __init__(self, method: str = "dnn", confidence_threshold: float = 0.5):
        """
        Args:
            method: "haar" or "dnn"
            confidence_threshold: minimum confidence to keep a DNN detection
        """
        self.method = method.lower()
        self.confidence_threshold = confidence_threshold

        if self.method == "haar":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.model = cv2.CascadeClassifier(cascade_path)
            if self.model.empty():
                raise IOError(f"Could not load Haar cascade from {cascade_path}")

        elif self.method == "dnn":
            prototxt = MODELS_DIR / "deploy.prototxt"
            weights = MODELS_DIR / "res10_300x300_ssd_iter_140000.caffemodel"
            if not prototxt.exists() or not weights.exists():
                raise FileNotFoundError(
                    "DNN model files not found. Run `python src/download_models.py` first, "
                    f"expected files at:\n  {prototxt}\n  {weights}"
                )
            self.model = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))

        else:
            raise ValueError("method must be 'haar' or 'dnn'")

    def detect(self, frame: np.ndarray):
        """
        Detect faces in a BGR image (as read by cv2.imread / video capture).

        Returns:
            List of bounding boxes as (x, y, w, h) tuples.
        """
        if self.method == "haar":
            return self._detect_haar(frame)
        return self._detect_dnn(frame)

    def _detect_haar(self, frame: np.ndarray):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        boxes = self.model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        return [tuple(map(int, b)) for b in boxes]

    def _detect_dnn(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self.model.setInput(blob)
        detections = self.model.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confidence_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            if x2 > x1 and y2 > y1:
                boxes.append((x1, y1, x2 - x1, y2 - y1))
        return boxes


def draw_boxes(frame: np.ndarray, boxes, labels=None, color=(0, 255, 0)):
    """Draw bounding boxes (and optional labels) on a copy of the frame."""
    out = frame.copy()
    labels = labels or [None] * len(boxes)
    for (x, y, w, h), label in zip(boxes, labels):
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        if label:
            cv2.putText(
                out, label, (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA
            )
    return out
