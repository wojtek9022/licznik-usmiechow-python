import cv2
import numpy as np
import logging
from PIL import Image, ImageTk
import tkinter as tk
import time
from .config_handler import ConfigHandler
from .auto_config_adjusting_handler import AutoConfigAdjustingHandler
from ..src.utils.video_frame_utils.expression_handler import ExpressionHandler
from ..src.detectors.smile_detector import SmileDetector
from ..src.detectors.face_detector import FaceDetector
from ..src.utils.video_frame_utils.video_capture_wrapper import VideoCaptureWrapper
from ..src.utils.video_frame_utils.effects_handler import EffectsHandler
from app.src.utils.video_frame_utils.frame_export_handler import FrameExportHandler

# Suppress OpenCV warnings
logging.getLogger("cv2").setLevel(logging.ERROR)

class VideoHandler:
    """
    Handles video capture and UI display.
    Coordinates between camera input and expression detection.
    """
    
    def __init__(self, master: tk.Tk, language: object, ui_handler: object) -> None:
        self.config_handler = ConfigHandler()
        self.config_handler.add_observer(self)
        self.master = master
        self.language = language
        self.ui_handler = ui_handler
        self.config = self.config_handler.get_config()
        self.camera_source = int(self.config.CAMERA_SOURCE)
        
        self.video_frame = None
        self.canvas = None
        
        # Initialize handlers
        self.face_detector = FaceDetector(self.config)
        self.smile_detector = SmileDetector(self.config)
        self.expression_handler = ExpressionHandler(
            self.config,
            self.face_detector,
            self.smile_detector
        )
        
        self.effects_handler = EffectsHandler()
        self.video_capture_wrapper = None
        self.running = False
        self.frame_export_handler = FrameExportHandler()
        
        # Initialize auto config handler
        self.auto_config_handler = None  # Will be initialized after canvas setup
        
        # Bind calibration keys
        self.master.bind('c', self._handle_calibration)
        self.master.bind('C', self._handle_calibration)
        
        # Add ESC key binding
        self.master.bind('<Escape>', self._handle_escape)

    def on_config_changed(self, new_config):
        """Handle config changes"""
        self.config = new_config
        self.camera_source = int(self.config.CAMERA_SOURCE)
        self.expression_handler.on_config_changed(new_config)
        self.face_detector.on_config_changed(new_config) 
        self.smile_detector.on_config_changed(new_config)

    def _initialize_camera(self) -> None:
        """Initialize or reinitialize camera capture."""
        if self.video_capture_wrapper:
            self.video_capture_wrapper.release()
            
        self.video_capture_wrapper = VideoCaptureWrapper(
            source=self.camera_source,
            api_preference=cv2.CAP_DSHOW
        )

    def start_video(self) -> None:
        """Start or restart video capture."""
        # Reload config to get latest camera source
        self.config = ConfigHandler().get_config()
        new_camera_source = int(self.config.CAMERA_SOURCE)
        
        # Check if camera source changed
        if new_camera_source != self.camera_source:
            self.camera_source = new_camera_source
            if self.video_capture_wrapper:
                self.video_capture_wrapper.release()
                self.video_capture_wrapper = None
        
        self._initialize_camera()
        self.ui_handler.hide_main_menu()
        self._setup_video_frame()
        self.running = True
        self.update_frame()

    def stop_video(self) -> None:
        """Stop video capture and cleanup resources"""
        self.running = False
        if self.video_capture_wrapper:
            self.video_capture_wrapper.release()
            self.video_capture_wrapper = None
        if self.auto_config_handler:
            self.auto_config_handler.reset_calibration()

    def update_frame(self) -> None:
        if not self.running:
            return
        
        check, frame = self.video_capture_wrapper.read()
        if check:
            self._process_frame(frame)
            self.master.after(10, self.update_frame)

    def _process_frame(self, frame: cv2.Mat) -> None:
        """Process frame for expression detection and effects."""
        processed_frame = self.expression_handler.process_frame(frame)

        # If smile was just counted, export the frame
        if hasattr(self.smile_detector, 'smile_counted') and self.smile_detector.smile_counted:
            self.frame_export_handler.export_frame(processed_frame)
        
        # Apply effects if faces were detected
        if hasattr(self.expression_handler, 'face_detector'):
            faces = self.expression_handler.face_detector.detect(
                cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            )
            for face_coords in faces:
                processed_frame = self.effects_handler.apply_random_effect(
                    processed_frame, face_coords
                )
        
        if self.config.AUTO_CONFIG_ADJUSTING:
            # Check for smile detection
            smile_detected = self.smile_detector.smile_counted
            self.auto_config_handler.handle_smile_detection(smile_detected)
        
        self._display_frame(processed_frame)

    def _display_frame(self, frame: cv2.Mat) -> None:
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk
        if self.config.AUTO_CONFIG_ADJUSTING:
            # Draw calibration text
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()
            self.auto_config_handler.draw_calibration_text(canvas_width, canvas_height)
        self.smile_detector.draw_counted_text(self.canvas, self.language)

    def _handle_escape(self, event) -> None:
        """Handle ESC key press - stop video and return to menu"""
        if self.running:
            self.stop_video()
            if self.video_frame:
                self.video_frame.destroy()
            self.ui_handler.show_main_menu()  # Changed from show_menu to show_main_menu

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
        
        # Initialize auto config handler with VideoHandler instance
        self.auto_config_handler = AutoConfigAdjustingHandler(self)

    def _handle_calibration(self, event):
        if self.config.AUTO_CONFIG_ADJUSTING and self.auto_config_handler:
            self.auto_config_handler.start_calibration()