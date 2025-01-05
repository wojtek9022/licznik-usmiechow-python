import cv2
from app.src.video_capture_wrapper import VideoCaptureWrapper
from app.src.smile_detector import SmileDetector
from app.src.fps_calculator import FPSCalculator
from PIL import Image, ImageTk
import tkinter as tk
from typing import Optional
import time

class VideoHandler:
    """
    Handles video capture and smile detection processing.

    This class manages the video capture stream, smile detection logic,
    and UI updates related to video display. It coordinates between
    the camera input, smile detection algorithm, and the UI components.

    Attributes:
        master (tk.Tk): Main application window
        language (object): Current language module with text strings
        ui_handler (object): Handler for UI updates and menu management
        video_capture_wrapper (VideoCaptureWrapper): Video capture handler
        smile_detector (SmileDetector): Smile detection processor
        running (bool): Video processing state flag
        video_frame (Optional[tk.Frame]): Frame for video display
        canvas (Optional[tk.Canvas]): Canvas for rendering video
        smiles_detected (int): Counter for detected smiles
    """

    def __init__(self, master: tk.Tk, language: object, ui_handler: object) -> None:
        """
        Initialize video handler.

        Args:
            master (tk.Tk): Main application window
            language (object): Language module for text strings
            ui_handler (object): Handler for UI updates
        """
        self.master = master
        self.language = language
        self.ui_handler = ui_handler
        self.video_capture_wrapper = VideoCaptureWrapper()
        self.smile_detector = SmileDetector()
        self.running = False
        self.video_frame = None
        self.canvas = None
        self.smiles_detected = 0
        self.master.bind('<Escape>', self._handle_escape)
        self.show_counted_text = False
        self.counted_text_timestamp = 0
        self.SMILE_TEXT_DURATION = 2.0  # seconds

    def setup_video_canvas(self) -> None:
        """Set up video display canvas."""
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def start_video(self) -> None:
        """
        Start video capture and processing.
        Initializes video canvas and starts frame processing loop.
        """
        self.ui_handler.hide_main_menu()  # Use UIHandler instead of direct manipulation
        self._setup_video_frame()  # Renamed from setup_video_canvas
        self.running = True
        self.update_frame()

    def stop_video(self) -> None:
        """
        Stop video capture and cleanup resources.
        Releases video capture, destroys windows and restores main menu.
        """
        self.running = False
        if self.video_capture_wrapper:
            self.video_capture_wrapper.release()
        if self.video_frame:
            self.video_frame.destroy()
        cv2.destroyAllWindows()
        self.ui_handler.show_main_menu()

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
        Process and display video frames.
        Handles frame capture, smile detection and UI updates.
        """
        if not self.running:
            return
        
        check, frame = self.video_capture_wrapper.read()
        if check:
            self._process_frame(frame)
            self.master.after(10, self.update_frame)

    def _process_frame(self, frame: cv2.Mat) -> None:
        """Process frame and check for ESC key."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.smile_detector.detect_faces(gray_frame)

        for (face_x, face_y, face_w, face_h) in faces:
            self.smile_detector.draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))
            face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]
            smiles = self.smile_detector.detect_smiles(face_region)
            smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]

            smile_detected = len(smiles) > 0
            self.smile_detector.handle_smile_and_draw(smile_detected, frame, smiles, face_x, face_y)
            if smile_detected and self.smiles_detected < self.smile_detector.smiles_detected:
                self.show_counted_text = True
                self.counted_text_timestamp = time.time()
            self.smiles_detected = self.smile_detector.smiles_detected
        self._display_frame(frame)

    def _handle_escape(self, event) -> None:
        """Handle ESC key press."""
        if self.running:
            self.stop_video()
            self.ui_handler.show_main_menu()

    def _display_frame(self, frame: cv2.Mat) -> None:
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

        text_to_show = self.language.DETECTED_SMILES_TEXT.format(count=self.smiles_detected)
        self.canvas.create_text(10, 10, anchor=tk.NW, text=text_to_show, fill="red", font=("Helvetica", 16))

        # Display smile counted text
        if self.show_counted_text:
            if time.time() - self.counted_text_timestamp > self.SMILE_TEXT_DURATION:
                self.show_counted_text = False
            else:
                self.canvas.create_text(
                    self.canvas.winfo_width() // 2,
                    self.canvas.winfo_height() // 4,
                    text=self.language.SMILE_COUNTED_TEXT,
                    fill="green",
                    font=("Helvetica", 28, "bold")
                )