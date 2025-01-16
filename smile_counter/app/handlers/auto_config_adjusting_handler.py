# auto_config_adjusting_handler.py

import time
import tkinter as tk
from app.handlers.config_handler import ConfigHandler

class AutoConfigAdjustingHandler:
    CALIBRATION_TEXT_DURATION = 5.0  # Duration to show calibration prompt
    CALIBRATION_DURATION = 10.0  # Duration of calibration process
    NO_SMILE_THRESHOLD = 1.5  # Time without smile before adjusting
    CONTINUOUS_SMILE_THRESHOLD = 1.0  # Time of continuous smile before adjusting
    ADJUSTMENT_PERCENTAGE = 0.10  # 10% adjustment factor

    def __init__(self, canvas):
        self.config_handler = ConfigHandler()
        self.config = ConfigHandler().get_config()
        self.canvas = canvas  # Store canvas reference
        self.calibration_start_time = None
        self.show_calibration_text = True
        self.calibration_text_start = time.time()
        self.calibration_active = False
        self.last_smile_state = False
        self.continuous_smile_start = 0
        self.no_smile_start = time.time()
        self.original_config_values = {}
        self.calibration_end_time = None

    def update_canvas(self, canvas):
        """Update canvas reference if needed"""
        self.canvas = canvas

    def draw_calibration_text(self, canvas_width, canvas_height) -> None:
        if not self.canvas:
            return
            
        current_time = time.time()
        
        if self.calibration_active:
            # Display calibration in progress text
            self.canvas.create_text(
                canvas_width // 2,
                canvas_height // 2,
                anchor=tk.CENTER,
                text="Configuration in progress\nPlease alternate between smiling and not smiling",
                fill="yellow",
                font=("Helvetica", 18, "bold")
            )
        elif current_time - self.calibration_text_start <= self.CALIBRATION_TEXT_DURATION or self.calibration_active == False:
            # Display calibration prompt
            self.canvas.create_text(
                10,  # X position
                canvas_height - 30,  # Y position
                anchor=tk.W,  # Left alignment
                text="Press 'C' to begin parameters calibration",
                fill="yellow",
                font=("Helvetica", 16)
            )
        else:
            self.show_calibration_text = False

    def start_calibration(self):
        if not self.calibration_active:
            # Save current state
            self._save_original_config()
            
            # Initialize calibration
            self.calibration_active = True
            self.calibration_start_time = time.time()
            self.calibration_end_time = self.calibration_start_time + self.CALIBRATION_DURATION
            self.last_smile_state = False
            self.continuous_smile_start = time.time()
            self.no_smile_start = time.time()
            
            # Disable features during calibration
            self._disable_features_during_calibration()

    def _save_original_config(self):
        """Save original configuration values"""
        self.original_config_values = {
            'APPLY_FACE_EFFECTS': self.config.APPLY_FACE_EFFECTS,
            'EXPORT_SMILE_FRAMES': self.config.EXPORT_SMILE_FRAMES,
            'SMILE_MIN_NEIGHBOURS': self.config.SMILE_MIN_NEIGHBOURS
        }

    def _disable_features_during_calibration(self):
        """Disable features during calibration"""
        self.config_handler.update_config({
            'APPLY_FACE_EFFECTS': 'False',
            'EXPORT_SMILE_FRAMES': 'False'
        })

    def _restore_original_config(self):
        """Restore original configuration after calibration"""
        self.config_handler.update_config({
            'APPLY_FACE_EFFECTS': str(self.original_config_values['APPLY_FACE_EFFECTS']),
            'EXPORT_SMILE_FRAMES': str(self.original_config_values['EXPORT_SMILE_FRAMES'])
        })

    def handle_smile_detection(self, smile_detected: bool):
        if not self.calibration_active:
            return

        current_time = time.time()
        
        # Check if calibration time is over
        if current_time >= self.calibration_end_time:
            self.calibration_active = False
            self._restore_original_config()
            return

        if smile_detected:
            if not self.last_smile_state:
                self.continuous_smile_start = current_time
            elif current_time - self.continuous_smile_start > self.CONTINUOUS_SMILE_THRESHOLD:
                # Increase minimum neighbors if smiling too long
                self._adjust_smile_min_neighbours(increase=True)
                self.continuous_smile_start = current_time
        else:
            if self.last_smile_state:
                self.no_smile_start = current_time
            elif current_time - self.no_smile_start > self.NO_SMILE_THRESHOLD:
                # Decrease minimum neighbors if not smiling for too long
                self._adjust_smile_min_neighbours(increase=False)
                self.no_smile_start = current_time

        self.last_smile_state = smile_detected

    def _adjust_smile_min_neighbours(self, increase: bool):
        current_value = int(self.config.SMILE_MIN_NEIGHBOURS)
        adjustment = max(1, int(current_value * self.ADJUSTMENT_PERCENTAGE))
        
        if increase:
            new_value = current_value + adjustment
        else:
            new_value = max(1, current_value - adjustment)
            
        print(f"Adjusting SMILE_MIN_NEIGHBOURS from {current_value} to {new_value}")
        self.config_handler.update_config({'SMILE_MIN_NEIGHBOURS': str(new_value)})

    def reset_calibration(self):
        """Reset calibration state"""
        self.calibration_active = False
        self.show_calibration_text = True
        self.calibration_text_start = time.time()
        if self.original_config_values:
            self._restore_original_config()