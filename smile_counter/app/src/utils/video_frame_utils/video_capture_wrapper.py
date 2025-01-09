import cv2
from typing import Tuple, Any

class VideoCaptureWrapper:
    """
    Wrapper for OpenCV video capture functionality.

    This class provides a simplified interface for video capture operations,
    handling camera initialization, frame capture, and resource cleanup.

    Attributes:
        capture (cv2.VideoCaptureWrapper): OpenCV video capture object
    """

    def __init__(self, source: int = 0, api_preference: int = None) -> None:
        """
        Initialize video capture from specified source.

        Args:
            source (int, optional): Camera index or video file path. 
                                  Defaults to 0 (first available camera).
            api_preference (int, optional): Preferred capture API backend.
                                          Defaults to None.
        """
        if api_preference is not None:
            self.capture = cv2.VideoCapture(source, api_preference)
        else:
            self.capture = cv2.VideoCapture(source)

    def read(self) -> Tuple[bool, Any]:
        """
        Read a frame from the video capture.

        Returns:
            Tuple[bool, Any]: Tuple containing:
                - bool: True if frame was successfully captured
                - Any: Captured frame as numpy array, or None if capture failed
        """
        return self.capture.read()

    def release(self) -> None:
        """
        Release video capture resources.

        Should be called when video capture is no longer needed
        to free system resources.
        """
        self.capture.release()
