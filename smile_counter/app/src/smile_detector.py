import time
import cv2
from .cascade_loader import CascadeLoader
from .config import FACE_SCALE_FACTOR, FACE_MIN_NEIGHBOURS, SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBOURS, TIME_TO_START_COUNTING

class SmileDetector:
    def __init__(self):
        # NOTE: Pay attention to the return in load_cascades, 
        # if the order of assigned variables is wrong the program will not work correctly.
        self.smile_cascade, self.face_cascade = CascadeLoader.load_cascades()
        self.smiles_detected = 0
        self.smile_active = False
        self.last_smile_time = 0

    def detect_faces(self, gray_frame):
        return self.face_cascade.detectMultiScale(gray_frame, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBOURS)

    def detect_smiles(self, gray_face_region):
        return self.smile_cascade.detectMultiScale(gray_face_region, scaleFactor=SMILE_SCALE_FACTOR, minNeighbors=SMILE_MIN_NEIGHBOURS)

    def handle_smile_and_draw(self, smile_detected, frame, smiles, face_x, face_y):
        current_time = time.time()
        if smile_detected:
            if not self.smile_active and (current_time - self.last_smile_time) > TIME_TO_START_COUNTING:
                self.smiles_detected += 1
                self.smile_active = True
        else:
            self.smile_active = False
            self.last_smile_time = current_time

        if smiles and self.smile_active:
            largest_smile = max(smiles, key=lambda s: s[2] * s[3])
            sx, sy, sw, sh = largest_smile
            smile_x, smile_y = face_x + sx, face_y + sy
            self.draw_rectangles(frame, [(smile_x, smile_y, sw, sh)], (255, 0, 0))  # Blue rectangle

    @staticmethod
    def draw_rectangles(frame, coordinates, color, thickness=3):
        for (x, y, w, h) in coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
