import cv2
import os

class CascadeLoader:
    @staticmethod
    def load_cascades():
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
