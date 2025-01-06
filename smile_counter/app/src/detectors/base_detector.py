from abc import ABC, abstractmethod
import cv2

class ExpressionDetector(ABC):
    @abstractmethod
    def detect(self, frame, region=None):
        """Detect expression in frame/region."""
        pass
    
    @abstractmethod
    def draw_detection(self, frame, detections, color=(0, 255, 0)):
        """Draw detection rectangles."""
        pass

    @staticmethod
    def draw_rectangles(frame, coordinates, color=(0, 255, 0), thickness=2):
        """Draw rectangles on frame using coordinates."""
        for (x, y, w, h) in coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)