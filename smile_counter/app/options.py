import tkinter as tk
from tkinter import messagebox
from typing import Dict, Any, Tuple
from configparser import ConfigParser
import os
from pathlib import Path
from app.config_handler import ConfigHandler


class Options:
    def __init__(self, master: tk.Tk, language: object, language_manager: object) -> None:
        self.master = master
        self.language = language
        self.language_manager = language_manager
        self.config_handler = ConfigHandler()
        
        self.config_options = {
            'FACE_SCALE_FACTOR': {'type': float, 'row': 0},
            'FACE_MIN_NEIGHBOURS': {'type': int, 'row': 1},
            'SMILE_SCALE_FACTOR': {'type': float, 'row': 2},
            'SMILE_MIN_NEIGHBOURS': {'type': int, 'row': 3},
            'TIME_TO_START_COUNTING': {'type': float, 'row': 4}
        }
        
        self.values: Dict[str, Any] = {}
        self.entries: Dict[str, tk.Entry] = {}
        self.labels: Dict[str, tk.Label] = {}
        self.options_window: tk.Toplevel | None = None
        
        self._init_ui_elements()
        self.load_config()

    def load_config(self) -> None:
        """Load configuration values from config file"""
        config = self.config_handler.get_config()
        for option_name in self.config_options:
            self.values[option_name] = self.config_options[option_name]['type'](
                getattr(config, option_name)
            )
        self._update_entries()

    def save_options(self) -> None:
        """Save configuration values to config file"""
        try:
            updates = {
                option_name: self.config_options[option_name]['type'](entry.get())
                for option_name, entry in self.entries.items()
            }
            self.config_handler.update_config(updates)
            self.load_config()
            messagebox.showinfo("Success", self.language.SUCCESS_MESSAGE_TEXT)
        except Exception as e:
            messagebox.showerror("Error", self.language.ERROR_MESSAGE_TEXT.format(error=e))

    def _update_entries(self) -> None:
        """Update UI entries with current values if they exist"""
        if not self.entries:
            return
            
        for option_name, value in self.values.items():
            if option_name in self.entries and self.entries[option_name]:
                entry = self.entries[option_name]
                entry.delete(0, tk.END)
                entry.insert(0, str(value))

    def _init_ui_elements(self) -> None:
        """Initialize UI elements with None values"""
        for option_name in self.config_options:
            self.entries[option_name] = None
            self.labels[option_name] = None

    def show_options(self) -> None:
        if self.options_window is not None and tk.Toplevel.winfo_exists(self.options_window):
            self.options_window.destroy()
        
        self.options_window = tk.Toplevel(self.master)
        self.options_window.title(self.language.OPTIONS_TITLE_TEXT)
        self._create_options_entries()
        self._create_save_button()

    def _create_options_entries(self) -> None:
        for option_name, options in self.config_options.items():
            label_text = getattr(self.language, f'{option_name}_TEXT')
            label, entry = self._create_option_entry(
                label_text, 
                self.values[option_name],
                options['row']
            )
            # Store reference with exact name
            setattr(self, f'{option_name.lower()}_label', label)
            self.labels[option_name] = label
            self.entries[option_name] = entry

    def _create_option_entry(self, label_text: str, value: float, row: int) -> tuple:
        label = tk.Label(self.options_window, text=label_text)
        label.grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(self.options_window)
        entry.grid(row=row, column=1, padx=10, pady=5)
        entry.insert(0, str(value))
        return label, entry

    def _create_save_button(self) -> None:
        save_button = tk.Button(
            self.options_window, 
            text=self.language.SAVE_BUTTON_TEXT, 
            command=self.save_options
        )
        save_button.grid(row=5, columnspan=2, padx=10, pady=10)

    def update_options_text(self, language: object) -> None:
        if self.options_window is not None and tk.Toplevel.winfo_exists(self.options_window):
            self.language = language
            # Update labels using stored references in self.labels
            for option_name, label in self.labels.items():
                label.config(text=getattr(language, f'{option_name}_TEXT'))
            self.options_window.title(language.OPTIONS_TITLE_TEXT)