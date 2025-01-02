import tkinter as tk
from typing import Callable

class ButtonCreator:
    def __init__(self, master: tk.Tk, language: object) -> None:
        self.master = master
        self.language = language

    def create_buttons(self, start_command: Callable, options_command: Callable, exit_command: Callable) -> tk.Frame:
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=20)

        self.start_button = tk.Button(self.button_frame, text=self.language.START_BUTTON_TEXT, command=start_command, width=15, height=2)
        self.start_button.pack(side=tk.TOP, padx=10, pady=5)

        self.options_button = tk.Button(self.button_frame, text=self.language.OPTIONS_BUTTON_TEXT, command=options_command, width=15, height=2)
        self.options_button.pack(side=tk.TOP, padx=10, pady=5)

        self.exit_button = tk.Button(self.button_frame, text=self.language.EXIT_BUTTON_TEXT, command=exit_command, width=15, height=2)
        self.exit_button.pack(side=tk.TOP, padx=10, pady=5)

        return self.button_frame

    def update_buttons(self, language: object) -> None:
        self.start_button.config(text=language.START_BUTTON_TEXT)
        self.options_button.config(text=language.OPTIONS_BUTTON_TEXT)
        self.exit_button.config(text=language.EXIT_BUTTON_TEXT)