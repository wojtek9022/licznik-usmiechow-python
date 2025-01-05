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