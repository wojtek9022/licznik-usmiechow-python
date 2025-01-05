import cv2
from app.src.video_capture import VideoCapture
from app.src.smile_detector import SmileDetector
from app.src.fps_calculator import FPSCalculator
from PIL import Image, ImageTk
import tkinter as tk
from typing import Optional

class VideoHandler:
    """
    Handles video capture and smile detection processing.

    This class manages the video capture stream, smile detection logic,
    and UI updates related to video display. It coordinates between
    the camera input, smile detection algorithm, and the UI components.

    Attributes:
        master (tk.Tk): Main application window
        language (object): Current language module with text strings
        header (tk.Label): Application header label
        subtitle (tk.Label): Application subtitle label
        logo_label (tk.Label): Logo display label
        button_frame (tk.Frame): Frame containing control buttons
        language_frame (tk.Frame): Frame containing language selection
        video_frame (Optional[tk.Frame]): Frame for video display
        canvas (Optional[tk.Canvas]): Canvas for rendering video
        video_capture (Optional[VideoCapture]): Video capture handler
        smile_detector (SmileDetector): Smile detection processor
        fps_calculator (FPSCalculator): FPS calculation utility
        smiles_detected (int): Counter for detected smiles
        running (bool): Video processing state flag
    """

    def __init__(self, master: tk.Tk, language: object, header: tk.Label, subtitle: tk.Label, logo_label: tk.Label, button_frame: tk.Frame, language_frame: tk.Frame) -> None:
        self.master = master
        self.language = language
        self.header = header
        self.subtitle = subtitle
        self.logo_label = logo_label
        self.button_frame = button_frame
        self.language_frame = language_frame
        self.video_frame: Optional[tk.Frame] = None
        self.canvas: Optional[tk.Canvas] = None
        self.video_capture: Optional[VideoCapture] = None
        self.smile_detector: SmileDetector = SmileDetector()
        self.fps_calculator: FPSCalculator = FPSCalculator()
        self.smiles_detected: int = 0
        self.running: bool = False

    def start_video(self) -> None:
        """
        Start video capture and smile detection.

        Initializes video capture, hides main menu elements,
        sets up video display frame and starts frame processing loop.
        """
        self._hide_main_menu()
        self._setup_video_frame()
        self.video_capture = VideoCapture()
        self.running = True
        self.update_frame()

    def stop_video(self) -> None:
        """
        Stop video capture and processing.

        Releases video capture resources, stops frame processing,
        and returns to main menu view.
        """

    def _hide_main_menu(self) -> None:
        self.header.pack_forget()
        self.subtitle.pack_forget()
        self.logo_label.pack_forget()
        self.button_frame.pack_forget()
        self.language_frame.pack_forget()

    def _setup_video_frame(self) -> None:
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def update_frame(self) -> None:
        """
        Process and display the next video frame.

        Captures frame from camera, processes it for smile detection,
        updates smile counter and FPS display, and schedules next frame update.
        """
        if self.running:
            ret, frame = self.video_capture.read()
            if ret:
                self._process_frame(frame)
                self.canvas.after(10, self.update_frame)

    def _process_frame(self, frame: cv2.Mat) -> None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.smile_detector.detect_faces(gray_frame)

        for (face_x, face_y, face_w, face_h) in faces:
            self.smile_detector.draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))
            face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]
            smiles = self.smile_detector.detect_smiles(face_region)
            smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]

            smile_detected = len(smiles) > 0
            self.smile_detector.handle_smile_and_draw(smile_detected, frame, smiles, face_x, face_y)
            self.smiles_detected = self.smile_detector.smiles_detected

        self._display_frame(frame)

    def _display_frame(self, frame: cv2.Mat) -> None:
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

        text_to_show = self.language.DETECTED_SMILES_TEXT.format(count=self.smiles_detected)
        self.canvas.create_text(10, 10, anchor=tk.NW, text=text_to_show, fill="red", font=("Helvetica", 16))