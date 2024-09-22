import time

class FPSCalculator:
    def __init__(self):
        self.prev_time = 0
        self.frames = 0

    def calculate(self, current_time):
        if self.prev_time == 0:
            self.prev_time = current_time  
            return 0

        fps = self.frames / (current_time - self.prev_time) if self.frames > 0 else 0
        self.prev_time = current_time
        self.frames = 0
        return fps

    def increment_frames(self):
        self.frames += 1
