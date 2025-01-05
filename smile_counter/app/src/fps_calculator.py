import time
from typing import Union

class FPSCalculator:
    """
    Calculates frames per second (FPS) for video processing.

    This class handles FPS calculation by tracking frame counts and
    time intervals between measurements. Provides methods for
    incrementing frame count and calculating current FPS.

    Attributes:
        prev_time (float): Previous time measurement
        frames (int): Number of frames processed since last calculation
    """

    def __init__(self):
        self.prev_time = 0.0
        self.frames = 0

    def calculate(self, current_time: float) -> float:
        """
        Calculate current frames per second.

        Args:
            current_time (float): Current timestamp for calculation

        Returns:
            float: Calculated FPS value. Returns 0 on first call or
                  if no frames were processed
        """
        if self.prev_time == 0:
            self.prev_time = current_time  
            return 0

        fps = self.frames / (current_time - self.prev_time) if self.frames > 0 else 0
        self.prev_time = current_time
        self.frames = 0
        return fps

    def increment_frames(self) -> None:
        """
        Increment the processed frames counter.

        Increases internal frame counter by one. This method should
        be called for each processed video frame.
        """
        self.frames += 1
