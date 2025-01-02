import tkinter as tk
from tkinter import messagebox
from typing import Callable
from app.config_manager import ConfigManager

class Options:
    def __init__(self, master: tk.Tk, language: object, language_manager: object) -> None:
        self.master = master
        self.language = language
        self.language_manager = language_manager
        self.config_manager = ConfigManager()
        config = self.config_manager.get_config()
        
        # Get configuration values
        self.FACE_SCALE_FACTOR = config.FACE_SCALE_FACTOR
        self.FACE_MIN_NEIGHBOURS = config.FACE_MIN_NEIGHBOURS
        self.SMILE_SCALE_FACTOR = config.SMILE_SCALE_FACTOR
        self.SMILE_MIN_NEIGHBOURS = config.SMILE_MIN_NEIGHBOURS
        self.TIME_TO_START_COUNTING = config.TIME_TO_START_COUNTING
        
        self._init_ui_elements()

    def _init_ui_elements(self) -> None:
        self.face_scale_label = None
        self.face_min_neighbours_label = None
        self.smile_scale_label = None
        self.smile_min_neighbours_label = None
        self.time_to_start_label = None
        self.options_window = None
        self.face_scale_entry = None
        self.face_min_neighbours_entry = None
        self.smile_scale_entry = None
        self.smile_min_neighbours_entry = None
        self.time_to_start_entry = None

    def show_options(self) -> None:
        if self.options_window is not None and tk.Toplevel.winfo_exists(self.options_window):
            self.options_window.destroy()
        
        self.options_window = tk.Toplevel(self.master)
        self.options_window.title(self.language.OPTIONS_TITLE_TEXT)
        self._create_options_entries()
        self._create_save_button()

    def _create_options_entries(self) -> None:
        self.face_scale_label, self.face_scale_entry = self._create_option_entry(
            self.language.FACE_SCALE_FACTOR_TEXT, self.FACE_SCALE_FACTOR, 0)
        self.face_min_neighbours_label, self.face_min_neighbours_entry = self._create_option_entry(
            self.language.FACE_MIN_NEIGHBOURS_TEXT, self.FACE_MIN_NEIGHBOURS, 1)
        self.smile_scale_label, self.smile_scale_entry = self._create_option_entry(
            self.language.SMILE_SCALE_FACTOR_TEXT, self.SMILE_SCALE_FACTOR, 2)
        self.smile_min_neighbours_label, self.smile_min_neighbours_entry = self._create_option_entry(
            self.language.SMILE_MIN_NEIGHBOURS_TEXT, self.SMILE_MIN_NEIGHBOURS, 3)
        self.time_to_start_label, self.time_to_start_entry = self._create_option_entry(
            self.language.TIME_TO_START_COUNTING_TEXT, self.TIME_TO_START_COUNTING, 4)

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

    def save_options(self) -> None:
        try:
            updates = {
                "FACE_SCALE_FACTOR": float(self.face_scale_entry.get()),
                "FACE_MIN_NEIGHBOURS": int(self.face_min_neighbours_entry.get()),
                "SMILE_SCALE_FACTOR": float(self.smile_scale_entry.get()),
                "SMILE_MIN_NEIGHBOURS": int(self.smile_min_neighbours_entry.get()),
                "TIME_TO_START_COUNTING": float(self.time_to_start_entry.get())
            }
            
            self.config_manager.update_config(updates)
            messagebox.showinfo("Success", self.language.SUCCESS_MESSAGE_TEXT)
            
        except Exception as e:
            messagebox.showerror("Error", self.language.ERROR_MESSAGE_TEXT.format(error=e))

    def update_options_text(self, language: object) -> None:
        if self.options_window is not None and tk.Toplevel.winfo_exists(self.options_window):
            self.language = language
            self.face_scale_label.config(text=language.FACE_SCALE_FACTOR_TEXT)
            self.face_min_neighbours_label.config(text=language.FACE_MIN_NEIGHBOURS_TEXT)
            self.smile_scale_label.config(text=language.SMILE_SCALE_FACTOR_TEXT)
            self.smile_min_neighbours_label.config(text=language.SMILE_MIN_NEIGHBOURS_TEXT)
            self.time_to_start_label.config(text=language.TIME_TO_START_COUNTING_TEXT)
            self.options_window.title(language.OPTIONS_TITLE_TEXT)