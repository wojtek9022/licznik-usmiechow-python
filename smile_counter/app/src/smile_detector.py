import time
import cv2
from app.src.cascade_loader import CascadeLoader
from app.config_handler import ConfigHandler

class SmileDetector:
    """
    Handles face and smile detection using OpenCV cascades.

    This class manages face detection, smile detection, and visualization
    of detection results using OpenCV cascade classifiers.

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

    def detect_smiles(self, gray_face_region) -> list:
        """Detect smiles in face region."""
        scale_factor = float(self.config.SMILE_SCALE_FACTOR)
        min_neighbors = int(self.config.SMILE_MIN_NEIGHBOURS)
        
        return self.smile_cascade.detectMultiScale(
            gray_face_region,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

    def handle_smile_and_draw(self, smile_detected: bool, frame, smiles: list, face_x: int, face_y: int) -> None:
        """
        Handle smile detection and draw visualization.

        Args:
            smile_detected (bool): Whether a smile was detected
            frame: Video frame to draw on
            smiles (list): List of detected smile coordinates
            face_x (int): X coordinate of face region
            face_y (int): Y coordinate of face region
        """
        current_time = time.time()
        cooldown_time = float(self.config.COUNTED_SMILE_COOLDOWN_TIME)
        
        if smile_detected:
            if not self.smile_active and (current_time - self.last_smile_time) > cooldown_time:
                self.smiles_detected += 1
                self.smile_active = True
                self.last_smile_time = current_time
        else:
            self.smile_active = False

        # Draw blue rectangle for detected smiles
        if smiles and self.smile_active:
            for (sx, sy, sw, sh) in smiles:
                smile_x, smile_y = face_x + sx, face_y + sy
                self.draw_rectangles(frame, [(smile_x, smile_y, sw, sh)], (255, 0, 0))

    @staticmethod
    def draw_rectangles(frame, coordinates: list, color: tuple, thickness: int = 3) -> None:
        """
        Draw rectangles on frame.

        Args:
            frame: Video frame to draw on
            coordinates (list): List of rectangle coordinates (x, y, w, h)
            color (tuple): RGB color tuple
            thickness (int, optional): Line thickness. Defaults to 3.
        """
        for (x, y, w, h) in coordinates:
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
