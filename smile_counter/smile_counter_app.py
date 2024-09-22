import cv2
import time
from src.video_capture import VideoCapture
from src.smile_detector import SmileDetector
from src.fps_calculator import FPSCalculator
from src.config import FONT

class SmileCounter:
    def __init__(self):
        self.video_capture = VideoCapture()
        self.smile_detector = SmileDetector()
        self.fps_calculator = FPSCalculator()

    def display_text(self, frame, text, position):
        cv2.putText(frame, text, position, getattr(cv2, f'FONT_{FONT["font"]}'), FONT['scale'], FONT['color'], FONT['thickness'], FONT['line_type'])

    def process_video(self):
        while True:
            check, frame = self.video_capture.read()
            if not check:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.smile_detector.detect_faces(gray_frame)

            for (face_x, face_y, face_w, face_h) in faces:
                self.smile_detector.draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))  # Draw red rectangle around face
                face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]  # Lower half of the face
                smiles = self.smile_detector.detect_smiles(face_region)
                smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]

                smile_detected = len(smiles) > 0
                self.smile_detector.handle_smile_and_draw(smile_detected, frame, smiles, face_x, face_y)

                current_time = time.time()
                #FIXME
                #self.fps_calculator.increment_frames()  # Increment frames before calculating
                #fps = self.fps_calculator.calculate(current_time)
                #text_to_show = f"Detected smiles: {self.smile_detector.smiles_detected} | FPS: {round(fps)}"
                text_to_show = f"Detected smiles: {self.smile_detector.smiles_detected}"
                h, w, _ = frame.shape
                self.display_text(frame, text_to_show, (round(w / 4), round(h / 8)))

            cv2.imshow('Smile Detection', frame)

            if cv2.waitKey(30) & 0xFF == 27:  # Exit on escape key
                break

        self.video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SmileCounter()
    app.process_video()
