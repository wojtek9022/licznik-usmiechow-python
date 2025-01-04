import cv2

class VideoCapture:
    def __init__(self, source=0):
        self.capture = cv2.VideoCapture(source)

    def read(self):
        return self.capture.read()

    def release(self):
        self.capture.release()
