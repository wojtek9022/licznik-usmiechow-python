from .abstract_detectors.base_detector import ExpressionDetector
import cv2
import time
from .cascade_loader import CascadeLoader
from app.handlers.config_handler import ConfigHandler
import tkinter as tk

class SmileDetector(ExpressionDetector):
    # FIXME: Magic number anti-pattern
    SMILE_TEXT_DURATION = 2.0  # Duration to show counted text

    def __init__(self, config):
        super().__init__()  # 
        self.config = config
        self.config_handler = ConfigHandler()
        self.config_handler.add_observer(self)
        self.smile_cascade, self.face_cascade = CascadeLoader.load_cascades()
        self.smiles_detected = int(config.TOTAL_SMILES_DETECTED)
        self.smile_active = False
        self.last_smile_time = 0
        self.show_counted_text = False
        self.counted_text_timestamp = 0
        self.smile_counted = False
        self.observers = []
        self.current_smile_status = False

    def add_observer(self, observer):
        self.observers.append(observer)

    def remove_observer(self, observer):
        self.observers.remove(observer)

    def notify_observers(self):
        for observer in self.observers:
            observer.on_smile_status_changed(self.current_smile_status)

    def update_smile_status(self, smile_detected: bool):
        if self.current_smile_status != smile_detected:
            self.current_smile_status = smile_detected
            self.notify_observers()

    def on_config_changed(self, new_config):
        """Handle config changes"""
        self.config = new_config

    def detect(self, frame, scaleFactor=None, minNeighbors=None):
        """Detect smiles in the given frame using config parameters"""
        if scaleFactor is None:
            scaleFactor = float(self.config.SMILE_SCALE_FACTOR)
        if minNeighbors is None:
            minNeighbors = int(self.config.SMILE_MIN_NEIGHBOURS)
        
        return self.smile_cascade.detectMultiScale(
            frame,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors
        )
        
    def draw_detection(self, frame, detections, face_coords=None, color=(0, 255, 0)):
        """
        Draw rectangles around detected smiles.
        
        Args:
            frame: Video frame to draw on
            detections: List of smile coordinates (x,y,w,h)
            face_coords: Tuple of face coordinates (x,y) or None
            color: RGB color tuple for rectangles
        """
        if self.config.DEBUG_MODE:
            if face_coords:
                face_x, face_y = face_coords
                # Adjust smile coordinates relative to face
                adjusted_detections = [(x + face_x, y + face_y, w, h) 
                                     for (x, y, w, h) in detections]
                ExpressionDetector.draw_rectangles(frame, adjusted_detections, color)
            else:
                ExpressionDetector.draw_rectangles(frame=frame, coordinates=detections, color=color)
                
    def handle_smile(self, smile_detected: bool) -> bool:
        current_time = time.time()
        
        if (smile_detected and 
            not self.smile_counted and 
            current_time - self.last_smile_time >= float(self.config.COUNTED_SMILE_COOLDOWN_TIME)):
            
            self.smile_counted = True
            self.last_smile_time = current_time
            self.smile_counted = True
            self.show_counted_text = True 
            self.counted_text_timestamp = current_time  
            self.smiles_detected += 1  
            self.config_handler.log_smile()
            return True
            
        if not smile_detected:
            self.smile_counted = False
            self.smile_counted = False
            
        return False

    def draw_counted_text(self, canvas, language) -> None:
        """Draw smile counter text overlay."""
        # Display total smiles count
        text_to_show = language.DETECTED_SMILES_TEXT.format(count=int(self.config.TOTAL_SMILES_DETECTED))
        canvas.create_text(10, 10, anchor=tk.NW, text=text_to_show, 
                         fill="red", font=("Helvetica", 16))

        # Display "Smile Counted!" text
        if self.show_counted_text:
            current_time = time.time()
            if current_time - self.counted_text_timestamp > self.SMILE_TEXT_DURATION:
                self.show_counted_text = False
            else:
                canvas_width = canvas.winfo_width()
                canvas_height = canvas.winfo_height()
                canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    text=language.SMILE_COUNTED_TEXT,
                    fill="green",
                    font=("Helvetica", 32, "bold")
                )