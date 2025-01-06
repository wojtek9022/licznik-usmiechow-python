import cv2
import numpy as np
import logging
from PIL import Image, ImageTk
import tkinter as tk
import time
from app.config_handler import ConfigHandler
from app.src.expression_handler import ExpressionHandler
from app.src.detectors.smile_detector import SmileDetector
from app.src.detectors.face_detector import FaceDetector
from app.src.video_capture_wrapper import VideoCaptureWrapper

# Suppress OpenCV warnings
logging.getLogger("cv2").setLevel(logging.ERROR)

class VideoHandler:
    """
    Handles video capture and UI display.
    Coordinates between camera input and expression detection.
    """
    
    def __init__(self, master: tk.Tk, language: object, ui_handler: object) -> None:
        self.master = master
        self.language = language
        self.ui_handler = ui_handler
        self.config = ConfigHandler().get_config()
        
        # Initialize video capture with DirectShow backend
        camera_source = int(self.config.CAMERA_SOURCE)
        self.video_capture_wrapper = VideoCaptureWrapper(
            source=camera_source,
            api_preference=cv2.CAP_DSHOW
        )
        
        # Initialize detectors
        self.face_detector = FaceDetector(self.config)
        self.smile_detector = SmileDetector(self.config)
        self.expression_handler = ExpressionHandler(
            self.config,
            self.face_detector,
            self.smile_detector
        )
        
        self.running = False
        self.video_frame = None
        self.canvas = None
        self.master.bind('<Escape>', self._handle_escape)

    def start_video(self) -> None:
        self.ui_handler.hide_main_menu()
        self._setup_video_frame()
        self.running = True
        self.update_frame()

    def stop_video(self) -> None:
        self.running = False
        if self.video_frame:
            self.video_frame.destroy()
        self.video_capture_wrapper.release()
        cv2.destroyAllWindows()

    def update_frame(self) -> None:
        if not self.running:
            return
        
        check, frame = self.video_capture_wrapper.read()
        if check:
            self._process_frame(frame)
            self.master.after(10, self.update_frame)

    def _process_frame(self, frame: cv2.Mat) -> None:
        """Process frame for expression detection."""
        processed_frame = self.expression_handler.process_frame(frame)
        self._display_frame(processed_frame)

    def _display_frame(self, frame: cv2.Mat) -> None:
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk
        self.smile_detector.draw_counted_text(self.canvas, self.language)

    def _handle_escape(self, event) -> None:
        """Handle ESC key press to stop video and restore menu."""
        if self.running:
            self.stop_video()
            self.ui_handler.show_main_menu()

    def _setup_video_frame(self) -> None:
        """Initialize video frame and canvas for display."""
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(
            self.video_frame, 
            width=self.master.winfo_width(),
            height=self.master.winfo_height(),
            bg='black'
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)