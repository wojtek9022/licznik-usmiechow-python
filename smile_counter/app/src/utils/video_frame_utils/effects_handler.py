import cv2
import os
import random
import time
from typing import List, Tuple
import numpy as np
from PIL import Image
from app.handlers.config_handler import ConfigHandler

class EffectsHandler:
    """Handles applying visual effects to detected faces in frames."""
    
    def __init__(self):
        self.config = ConfigHandler().get_config()
        
        # Get current file's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get app directory
        app_dir = os.path.dirname(os.path.dirname(current_dir))
        # Build path to effects directory - remove redundant 'src'
        self.effects_base_path = os.path.join(app_dir, 'data', 'img', 'effects')
        
        self._load_effects()

        # Add these new instance variables
        self.last_effect_time = time.time()
        self.last_effect_function = None
        #FIXME: magic numbers
        self.effect_duration = 5.0  # Duration in seconds

    def _load_effects(self) -> None:
        """Load effects from respective directories."""
        self.effects = {
            'hair': self._load_directory(os.path.join('head', 'hair')),
            'mustache': self._load_directory(os.path.join('head', 'mustache')),
            'beard': self._load_directory(os.path.join('head', 'beard')),
            'lips': self._load_directory(os.path.join('head', 'lips'))
        }

    def _load_directory(self, subpath: str) -> List[str]:
        """
        Recursively load all PNG files from specified directory and its subdirectories.
        """
        full_path = os.path.normpath(os.path.join(self.effects_base_path, subpath))
        print(f"Checking path: {full_path}")  # Debug line
        if not os.path.exists(full_path):
            print(f"Path does not exist: {full_path}")  # Debug line
            return []
            
        effect_files = []
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.endswith('.png'):
                    effect_files.append(os.path.join(root, file))
        return effect_files

    def _get_random_effect(self, effect_type: str) -> str:
        """Get random effect from specified type including subdirectories."""
        if not self.effects[effect_type]:
            return None
        return random.choice(self.effects[effect_type])

    def _apply_hair_effect(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply hair effect above the face area."""
        x, y, w, h = face_coords
        effect_path = self._get_random_effect('hair')
        if not effect_path:
            return frame
        
        effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
        effect_img = cv2.resize(effect_img, (w, int(h*0.6)))
        
        return self._overlay_effect(frame, effect_img, (x, y-int(h*0.3)))

    def _apply_mustache_effect(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply mustache effect in the middle of face area."""
        x, y, w, h = face_coords
        effect_path = self._get_random_effect('mustache')
        if not effect_path:
            return frame
            
        effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
        effect_img = cv2.resize(effect_img, (int(w*0.6), int(h*0.15)))
        
        return self._overlay_effect(frame, effect_img, 
                                  (x + int(w*0.2), y + int(h*0.6)))

    def _overlay_effect(self, frame: np.ndarray, effect: np.ndarray, 
                       position: Tuple[int, int]) -> np.ndarray:
        """Overlay effect on frame with transparency."""
        x, y = position
        if effect.shape[2] < 4:  # No alpha channel
            return frame
            
        alpha = effect[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+effect.shape[0], x:x+effect.shape[1], c] = \
                frame[y:y+effect.shape[0], x:x+effect.shape[1], c] * (1 - alpha) + \
                effect[:, :, c] * alpha
                
        return frame

    def apply_random_effect(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply random effect to the frame if DEBUG_MODE is enabled. Effect changes every x seconds."""
        if not self.config.DEBUG_MODE:
            return frame
            
        current_time = time.time()
        
        # Simplified face coordinates validation
        has_valid_face = (
            face_coords is not None and 
            len(face_coords) == 4
        )
        
        if (self.last_effect_function is None or 
            current_time - self.last_effect_time >= self.effect_duration or 
            not has_valid_face):
            
            effect_functions = [
                self._apply_hair_effect,
                self._apply_mustache_effect
            ]
            self.last_effect_function = random.choice(effect_functions)
            self.last_effect_time = current_time
        
        if has_valid_face:
            return self.last_effect_function(frame, face_coords)
        return frame