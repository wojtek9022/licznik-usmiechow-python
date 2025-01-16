# auto_config_adjusting_handler.py

import time
import tkinter as tk
from app.handlers.config_handler import ConfigHandler
from app.src.data.lang import lang_pl, lang_en

class AutoConfigAdjustingHandler:
    CALIBRATION_TEXT_DURATION = 5.0  # Duration to show calibration prompt
    CALIBRATION_DURATION = 20.0  # Duration of calibration process
    PHASE_ONE_PERCENTAGE = 0.65  # First phase takes 65% of total time
    NO_SMILE_THRESHOLD = 0.65  # Time without smile before adjusting
    CONTINUOUS_SMILE_THRESHOLD = 0.65  # Time of continuous smile before adjusting
    ADJUSTMENT_PERCENTAGE = 0.20  # adjustment factor

    def __init__(self, canvas):
        self.config_handler = ConfigHandler()
        self.config_handler.add_observer(self)  # Register as observer
        self.config = self.config_handler.get_config()
        self.canvas = canvas  # Store canvas reference
        self.language = lang_pl if self.config.LANGUAGE == 'pl' else lang_en
        self.calibration_start_time = None
        self.show_calibration_text = True
        self.calibration_text_start = time.time()
        self.calibration_active = False
        self.last_smile_state = False
        self.continuous_smile_start = 0
        self.no_smile_start = time.time()
        self.original_config_values = {}
        self.calibration_end_time = None
        self.calibration_phase = 1  # Track current phase
        self.phase_one_end_time = None

    def on_config_changed(self, new_config):
        """Handle configuration changes"""
        self.config = new_config
        self.language = lang_pl if self.config.LANGUAGE == 'pl' else lang_en

    def update_canvas(self, canvas):
        """Update canvas reference if needed"""
        self.canvas = canvas

    def draw_calibration_text(self, canvas_width, canvas_height) -> None:
        if not self.canvas:
            return
            
        current_time = time.time()
        
        if self.calibration_active:
            phase_text = (self.language.CALIBRATION_SMILE_TEXT 
                         if self.calibration_phase == 1 
                         else self.language.CALIBRATION_NO_SMILE_TEXT)
            
            if self.calibration_phase == 1:
                self.canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    text=phase_text,
                    fill="yellow",
                    font=("Helvetica", 18, "bold")
                    )
            else:
                    self.canvas.create_text(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    text=phase_text,
                    fill="red",
                    font=("Helvetica", 18, "bold")
                    )
        elif current_time - self.calibration_text_start <= self.CALIBRATION_TEXT_DURATION or self.calibration_active == False:
            # Display calibration prompt
            self.canvas.create_text(
                10,  # X position
                canvas_height - 30,  # Y position
                anchor=tk.W,  # Left alignment
                text=self.language.CALIBRATION_PROMPT_TEXT,
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
            self.phase_one_end_time = self.calibration_start_time + (self.CALIBRATION_DURATION * self.PHASE_ONE_PERCENTAGE)
            self.calibration_phase = 1
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
        elapsed_time = current_time - self.calibration_start_time
        phase_one_duration = self.CALIBRATION_DURATION * self.PHASE_ONE_PERCENTAGE
        
        # Phase transition check using elapsed time
        if self.calibration_phase == 1 and elapsed_time >= phase_one_duration:
            self.calibration_phase = 2
            self.no_smile_start = current_time
            
        # Calibration end check
        if elapsed_time >= self.CALIBRATION_DURATION:
            self.calibration_active = False
            self._restore_original_config()
            return

        # Phase-specific smile detection
        if self.calibration_phase == 1:
            # Phase 1: Looking for smile
            if not smile_detected and (current_time - self.no_smile_start > self.NO_SMILE_THRESHOLD):
                self._adjust_smile_min_neighbours(increase=False)
                self.no_smile_start = current_time
        else:
            # Phase 2: Looking for no smile
            if smile_detected and (current_time - self.no_smile_start > self.NO_SMILE_THRESHOLD):
                self._adjust_smile_min_neighbours(increase=True)
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