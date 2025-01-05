from .base_detector import ExpressionDetector
import cv2
from app.src.detectors.cascade_loader import CascadeLoader

class FaceDetector(ExpressionDetector):
    def __init__(self, config):
        self.config = config
        _, self.face_cascade = CascadeLoader.load_cascades()

    def detect(self, frame, region=None):
        return self.face_cascade.detectMultiScale(
            frame,
            scaleFactor=float(self.config.FACE_SCALE_FACTOR),
            minNeighbors=int(self.config.FACE_MIN_NEIGHBOURS)
        )
        
    def draw_detection(self, frame, detections, color=(0, 0, 255)):
        if self.config.DEBUG_MODE:
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)