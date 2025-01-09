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
        self.effects_base_path = os.path.join(app_dir, 'data', 'img', 'effects')
        
        self._load_effects()

        # Add these new instance variables
        self.last_effect_time = time.time()
        self.current_effect_func = None
        self.current_effect_path = None 
        #FIXME: magic numbers
        self.effect_duration = 5.0
        self.effect_functions = {
            'hair': self._apply_hair_effect,
            'mustache': self._apply_mustache_effect,
            'beard': self._apply_beard_effect
        }

    def _load_effects(self) -> None:
        """Load effects from respective directories with verification."""
        self.effects = {
            'hair': self._load_directory(os.path.join('head', 'hair')),
            'mustache': self._load_directory(os.path.join('head', 'mustache')),
            'beard': self._load_directory(os.path.join('head', 'beard'))
        }
        
        # Verify loaded effects
        for effect_type, paths in self.effects.items():
            print(f"Loaded {len(paths)} {effect_type} effects")

    def _load_directory(self, subpath: str) -> List[str]:
        """Load effects from directory with debug logging."""
        full_path = os.path.join(self.effects_base_path, subpath)
        effects = []
        
        if not os.path.exists(full_path):
            print(f"Warning: Effects directory not found: {full_path}")
            return effects
            
        for root, _, files in os.walk(full_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    effect_path = os.path.join(root, file)
                    effects.append(effect_path)
                    print(f"Loaded effect: {effect_path}")  # Debug logging
                    
        if not effects:
            print(f"Warning: No effects found in {full_path}")
            
        return effects

    def _clean_image_profile(self, image_path: str) -> str:
        """Remove problematic sRGB profile from PNG images."""
        try:
            with Image.open(image_path) as img:
                # Create clean copy without profile
                cleaned = Image.new(img.mode, img.size)
                cleaned.putdata(list(img.getdata()))
                
                # Save to temporary file
                temp_path = image_path + '.tmp'
                cleaned.save(temp_path, 'PNG')
                
                # Replace original with cleaned version
                os.replace(temp_path, image_path)
                
        except Exception as e:
            print(f"Warning: Could not clean image profile for {image_path}: {e}")
        
        return image_path

    def _get_random_effect(self, effect_type: str) -> str:
        """Get random effect from specified type including subdirectories."""
        if not self.effects[effect_type]:
            return None
        effect_path = random.choice(self.effects[effect_type])
        return self._clean_image_profile(effect_path)

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

    def _apply_beard_effect(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply beard effect at bottom of face area."""
        x, y, w, h = face_coords
        effect_path = self._get_random_effect('beard')
        if not effect_path:
            return frame
        
        effect_img = cv2.imread(effect_path, cv2.IMREAD_UNCHANGED)
        effect_img = cv2.resize(effect_img, (int(w*0.7), int(h*0.4)))  # Larger than mustache
        
        return self._overlay_effect(frame, effect_img, 
                                  (x + int(w*0.15), y + int(h*0.65)))  # Position below mustache

    def _overlay_effect(self, frame: np.ndarray, effect: np.ndarray, position: Tuple[int, int]) -> np.ndarray:
        """Overlay effect on frame with transparency."""
        x, y = position
        frame_h, frame_w = frame.shape[:2]
        effect_h, effect_w = effect.shape[:2]
        
        # Ensure coordinates are within frame bounds
        x = max(0, min(x, frame_w))
        y = max(0, min(y, frame_h))
        
        # Calculate valid overlay region
        x_end = min(x + effect_w, frame_w)
        y_end = min(y + effect_h, frame_h)
        
        # Skip if no valid overlay region
        if x_end <= x or y_end <= y:
            return frame
            
        # Calculate effect region to use
        effect_x = 0
        effect_y = 0
        effect_w = x_end - x
        effect_h = y_end - y
        
        alpha = effect[effect_y:effect_y+effect_h, effect_x:effect_x+effect_w, 3] / 255.0
        for c in range(3):
            frame[y:y+effect_h, x:x+effect_w, c] = \
                frame[y:y+effect_h, x:x+effect_w, c] * (1 - alpha) + \
                effect[effect_y:effect_y+effect_h, effect_x:effect_x+effect_w, c] * alpha
                
        return frame

    def apply_random_effect(self, frame: np.ndarray, face_coords: Tuple[int, int, int, int]) -> np.ndarray:
        if not self.config.APPLY_FACE_EFFECTS:
            return frame
            
        current_time = time.time()
        has_valid_face = (face_coords is not None and len(face_coords) == 4)
        
        if (self.current_effect_func is None or 
            current_time - self.last_effect_time >= self.effect_duration or 
            not has_valid_face):
            
            effect_category = random.choice(list(self.effect_functions.keys()))
            self.current_effect_func = self.effect_functions[effect_category]
            self.current_effect_path = self._get_random_effect(effect_category)
            self.last_effect_time = current_time
        
        if has_valid_face and self.current_effect_path:
            x, y, w, h = face_coords
            effect_img = cv2.imread(self.current_effect_path, cv2.IMREAD_UNCHANGED)
            
            if self.current_effect_func == self._apply_hair_effect:
                effect_img = cv2.resize(effect_img, (w, int(h*0.6)))
                return self._overlay_effect(frame, effect_img, (x, y-int(h*0.3)))
            elif self.current_effect_func == self._apply_mustache_effect:
                effect_img = cv2.resize(effect_img, (int(w*0.6), int(h*0.15)))
                return self._overlay_effect(frame, effect_img, (x + int(w*0.2), y + int(h*0.6)))
            elif self.current_effect_func == self._apply_beard_effect:
                effect_img = cv2.resize(effect_img, (int(w*1.1), int(h*0.9)))
                return self._overlay_effect(frame, effect_img, (x - int(w*0.1), y + int(h*0.55)))
                
        return frame