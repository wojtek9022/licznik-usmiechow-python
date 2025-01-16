import cv2
from typing import List, Tuple
import time
from app.handlers.config_handler import ConfigHandler

class ExpressionHandler:
    def __init__(self, config, face_detector, smile_detector):
        self.config_handler = ConfigHandler()
        self.config_handler.add_observer(self)
        self.config = config
        self.face_detector = face_detector
        self.smile_detector = smile_detector
        # FIXME: Magic string
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.continuous_smile_start = 0  # Track when smile started

    def on_config_changed(self, new_config):
        """Handle config changes"""
        self.config = new_config
        
    def convert_to_gray(self, frame) -> cv2.Mat:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    def detect_faces(self, gray_frame) -> List[Tuple[int, int, int, int]]:
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=float(self.config.FACE_SCALE_FACTOR),
            minNeighbors=int(self.config.FACE_MIN_NEIGHBOURS)
        )
        
    def draw_detection(self, frame, detections: List[Tuple], color: Tuple = (0, 0, 255)):
        if self.config.DEBUG_MODE:
            for (x, y, w, h) in detections:
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
    def handle_expression(self, frame, detector, region=None):
        detections = detector.detect(frame, region)
        return len(detections) > 0, detections

    def process_frame(self, frame) -> cv2.Mat:
        # FIXME: Refactor this method for better genericity
        # It is only being used for smile detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detect(gray_frame)
        self.face_detector.draw_detection(frame, faces)
        
        smile_detected = False
        for (face_x, face_y, face_w, face_h) in faces:
            face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]
            smiles = self.smile_detector.detect(face_region)
            
            if len(smiles) > 0:
                # Keep all 4 coordinates (x,y,w,h) when adjusting smiles
                adjusted_smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]
                self.smile_detector.draw_detection(
                    frame, 
                    adjusted_smiles, 
                    face_coords=(face_x, face_y)
                )
                self.smile_detector.smile_just_counted = True
                smile_detected = True
                
                # Start counting time when smile first detected
                if self.continuous_smile_start == 0:
                    self.continuous_smile_start = time.time()
                    
                # Check if smile has been held long enough
                elif time.time() - self.continuous_smile_start >= float(self.config.TIME_TO_START_COUNTING):
                    self.smile_detector.handle_smile(True)
                    self.continuous_smile_start = 0  # Reset timer after counting
            
        if not smile_detected:
            self.continuous_smile_start = 0  # Reset timer when smile breaks
            self.smile_detector.smile_just_counted = False
            self.smile_detector.handle_smile(False)
                
        return frame