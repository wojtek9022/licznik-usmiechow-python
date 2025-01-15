import tkinter as tk
from typing import Dict, Any
from tkinter import messagebox
from .config_handler import ConfigHandler
from ..src.utils.options_utils.options_types import OptionsConfig
from ..src.utils.options_utils.options_validator import OptionsValidator
from ..src.utils.options_utils.options_ui import OptionsUI
from ..src.data.lang import lang_en, lang_pl

class OptionsHandler:
    """
    Handles application configuration and options management.

    This class manages loading, saving, and validating configuration options,
    as well as handling language changes and UI updates. It serves as a bridge
    between the configuration storage and the options UI.

    Attributes:
        master (tk.Tk): Main application window
        config_handler (ConfigHandler): Configuration file handler
        language (object): Current language module with text strings
        validator (OptionsValidator): Options validation handler
        ui (OptionsUI): Options window UI manager
        values (Dict[str, Any]): Current configuration values
    """
    DEFAULT_CONFIG = {
        'FACE_SCALE_FACTOR': {'type': float, 'row': 1},
        'FACE_MIN_NEIGHBOURS': {'type': int, 'row': 2},
        'SMILE_SCALE_FACTOR': {'type': float, 'row': 3},
        'SMILE_MIN_NEIGHBOURS': {'type': int, 'row': 4},
        'TIME_TO_START_COUNTING': {'type': float, 'row': 5},
        'COUNTED_SMILE_COOLDOWN_TIME': {'type': float, 'row': 6},
        'CAMERA_SOURCE': {'type': int, 'row': 7},
        'DEBUG_MODE': {'type': bool, 'row': 8},
        'APPLY_FACE_EFFECTS': {'type': bool, 'row':9},
        'EXPORT_SMILE_FRAMES': {'type': bool, 'row': 10},
        'AUTO_CONFIG_ADJUSTING': {'type': bool, 'row': 11},
    }

    def __init__(self, master: tk.Tk, language: object) -> None:
        self.master = master
        self.config_handler = ConfigHandler()
        self.language = language
        self.validator = OptionsValidator()
        self.ui = OptionsUI(master, self.language, self.validator)
        self.values: Dict[str, Any] = {}
        self.load_config()

    def update_language(self, language: object) -> None:
        """
        Update OptionsHandler language from UIHandler.
        
        Args:
            language (object): New language module
        """
        self.language = language
        self.ui.on_language_change(self.language)

    def _parse_bool_value(self, value: str) -> bool:
        """Convert string value to boolean."""
        return str(value).lower() in ('true', '1', 'yes', 'on')

    def _parse_value(self, value: Any, value_type: type) -> Any:
        """
        Parse value based on type.
        
        Args:
            value: Value to parse
            value_type: Type to convert to
        
        Returns:
            Parsed value of specified type
        """
        if value_type == bool:
            return self._parse_bool_value(value)
        return value

    def load_config(self) -> None:
        """Load configuration values from storage."""
        config = self.config_handler.get_config()
        for option_name, options in self.DEFAULT_CONFIG.items():
            value = getattr(config, option_name)
            self.values[option_name] = self._parse_value(value, options['type'])

    def save_options(self) -> None:
        """Save updated options to config."""
        try:
            all_valid = True
            updated_values = {}
            
            for option_name, options in self.DEFAULT_CONFIG.items():
                value = self.ui.get_value(option_name)
                
                if options['type'] == bool:
                    updated_values[option_name] = bool(value)
                else:
                    if not self.validator.validate_value(value, options['type']):
                        all_valid = False
                        break
                    updated_values[option_name] = options['type'](value)
            
            if all_valid:
                self.config_handler.update_config(updated_values)
                print("Reloading config in options")
                self.load_config() # Reload config to update values in options window
                messagebox.showinfo("Success", self.language.OPTIONS_SAVED_TEXT)
                self.ui.window.destroy()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def show_options(self) -> None:
        """
        Display the options configuration window.

        Creates new window if none exists or brings existing one to front.
        Initializes all input fields with current values.
        """
        self.ui.create_window(self.DEFAULT_CONFIG, self.values, self.save_options)