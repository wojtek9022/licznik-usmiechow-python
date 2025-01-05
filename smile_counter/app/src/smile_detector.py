import time
import cv2
from .cascade_loader import CascadeLoader
from app.config_handler import ConfigHandler

class SmileDetector:
    """
    Handles face and smile detection using OpenCV cascades.

    Attributes:
        smile_cascade: OpenCV cascade classifier for smile detection
        face_cascade: OpenCV cascade classifier for face detection
        smiles_detected (int): Counter for detected smiles
        smile_active (bool): Flag indicating if smile is currently detected
        last_smile_time (float): Timestamp of last detected smile
        config (Any): Configuration object with detection parameters
    """

    def __init__(self):
        self.smile_cascade, self.face_cascade = CascadeLoader.load_cascades()
        self.smiles_detected = 0
        self.smile_active = False
        self.last_smile_time = 0
        self.config = ConfigHandler().get_config()

    def detect_faces_and_smiles(self, frame):
        """Process frame for face and smile detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.config.FACE_SCALE_FACTOR,
            minNeighbors=self.config.FACE_MIN_NEIGHBOURS
        )

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            
            smiles = self.smile_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=self.config.SMILE_SCALE_FACTOR,
                minNeighbors=self.config.SMILE_MIN_NEIGHBOURS
            )

            if len(smiles) > 0:
                if not self.smile_active:
                    current_time = time.time()
                    if current_time - self.last_smile_time > self.config.TIME_TO_START_COUNTING:
                        self.smiles_detected += 1
                        self.smile_active = True
                        self.last_smile_time = current_time
            else:
                self.smile_active = False

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        return frame, self.smiles_detected
