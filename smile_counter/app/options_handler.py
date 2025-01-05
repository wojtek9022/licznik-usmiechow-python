import tkinter as tk
from typing import Dict, Any
from tkinter import messagebox
from .config_handler import ConfigHandler
from .options_types import OptionsConfig
from .options_validator import OptionsValidator
from .options_ui import OptionsUI
from app.src.lang import lang_en, lang_pl

class OptionsHandler:
    DEFAULT_CONFIG: OptionsConfig = {
        'FACE_SCALE_FACTOR': {'type': float, 'row': 0},
        'FACE_MIN_NEIGHBOURS': {'type': int, 'row': 1},
        'SMILE_SCALE_FACTOR': {'type': float, 'row': 2},
        'SMILE_MIN_NEIGHBOURS': {'type': int, 'row': 3},
        'TIME_TO_START_COUNTING': {'type': float, 'row': 4}
    }

    def __init__(self, master: tk.Tk):
        self.master = master
        self.config_handler = ConfigHandler()
        self.language = self._load_language()
        self.validator = OptionsValidator()
        self.ui = OptionsUI(master, self.language, self.validator)
        self.values: Dict[str, Any] = {}
        self.load_config()

    def _load_language(self) -> object:
        config = self.config_handler.get_config()
        try:
            language = config.get('LANGUAGE', fallback='en')
            return lang_pl if language == 'pl' else lang_en
        except Exception:
            return lang_en

    def change_language(self, lang_code: str) -> None:
        self.language = lang_pl if lang_code == 'pl' else lang_en
        self.config_handler.update_config({'LANGUAGE': lang_code})
        self.ui.update_language(self.language)  # Update UI directly

    def load_config(self) -> None:
        config = self.config_handler.get_config()
        for option_name, options in self.DEFAULT_CONFIG.items():
            self.values[option_name] = options['type'](getattr(config, option_name))

    def save_options(self) -> None:
        try:
            # Validate all entries
            all_valid = True
            for option_name in self.ui.entries:
                if not self.ui._validate_entry(option_name):
                    all_valid = False
            
            if not all_valid:
                return
                
            updates = {
                option_name: self.DEFAULT_CONFIG[option_name]['type'](entry.get())
                for option_name, entry in self.ui.entries.items()
            }
            
            self.config_handler.update_config(updates)
            self.load_config()
            messagebox.showinfo("Success", self.language.SUCCESS_MESSAGE_TEXT)
        except Exception as e:
            messagebox.showerror("Error", self.language.ERROR_MESSAGE_TEXT.format(error=e))

    def show_options(self) -> None:
        self.ui.create_window(self.DEFAULT_CONFIG, self.values, self.save_options)

    def update_language(self, language: object) -> None:
        self.language = language
        self.ui.update_language(language)