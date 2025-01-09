import cv2
import os
from typing import Tuple

class CascadeLoader:
    """
    Handles loading of OpenCV cascade classifiers for face and smile detection.

    This class provides static methods for loading haar cascade classifier files
    from predefined locations. It validates classifier files existence and
    ensures proper loading before returning classifier objects.
    """

    @staticmethod
    def load_cascades() -> Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]:
        """
        Load face and smile cascade classifiers.

        Loads haar cascade classifier files from the data directory and
        validates their existence and proper loading.

        Returns:
            Tuple[cv2.CascadeClassifier, cv2.CascadeClassifier]: 
                Tuple containing (smile_cascade, face_cascade) classifiers

        Raises:
            Exception: If either cascade file is missing or fails to load
        """
        current_dir = os.path.dirname(__file__)
        face_cascade_path = os.path.join(current_dir, 'data/haar_classifiers/haarcascade_frontalface_default.xml')
        smile_cascade_path = os.path.join(current_dir, 'data/haar_classifiers/haarcascade_smile.xml')
        
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        smile_cascade = cv2.CascadeClassifier(smile_cascade_path)

        if face_cascade.empty():
            raise Exception(
                f"Error: Failed to load haar frontal face classifier file: {face_cascade_path}."
                "Please check if the file exists and is accessible."
                "Closing App."
            )
        if smile_cascade.empty():
            raise Exception(
                f"Error: Failed to load haar smile classifier file: {smile_cascade_path}."
                "Please check if the file exists and is accessible."
                "Closing App."
            )

        return smile_cascade, face_cascade
