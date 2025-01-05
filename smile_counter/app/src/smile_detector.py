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

    def detect_faces(self, gray_frame) -> list:
        """Detect faces in grayscale frame."""
        scale_factor = float(self.config.FACE_SCALE_FACTOR)
        min_neighbors = int(self.config.FACE_MIN_NEIGHBOURS)
        
        return self.face_cascade.detectMultiScale(
            gray_frame,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

    def detect_smiles(self, roi_gray) -> list:
        """Detect smiles in face region."""
        scale_factor = float(self.config.SMILE_SCALE_FACTOR)
        min_neighbors = int(self.config.SMILE_MIN_NEIGHBOURS)
        
        return self.smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

