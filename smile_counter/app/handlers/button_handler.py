import tkinter as tk
from typing import Callable

class ButtonHandler:
    """
    Creates and manages application control buttons.

    This class handles the creation and updating of main menu buttons
    including start, options, and exit controls. It manages button layout
    and language-specific text updates.

    Attributes:
        master (tk.Tk): Main application window
        language (object): Current language module with text strings
        button_frame (tk.Frame): Container for all buttons
        start_button (tk.Button): Button to start video capture
        options_button (tk.Button): Button to open options menu
        statistics_button (tk.Button): Button to view statistics
        exit_button (tk.Button): Button to close application
    """
    def __init__(self, master: tk.Tk, language: object) -> None:
        self.master = master
        self.language = language

    def create_buttons(self, start_command: Callable, options_command: Callable, 
                      statistics_command: Callable, exit_command: Callable) -> tk.Frame:
        """
        Create and arrange main menu buttons.

        Creates a frame containing four buttons: start, options, statistics, and exit.
        Each button is configured with the provided command callback and
        current language text.

        Args:
            start_command (Callable): Function to execute on start button click
            options_command (Callable): Function to execute on options button click
            statistics_command (Callable): Function to execute on statistics button click
            exit_command (Callable): Function to execute on exit button click

        Returns:
            tk.Frame: Frame containing all created buttons
        """
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=20)

        self.start_button = tk.Button(self.button_frame, text=self.language.START_BUTTON_TEXT, 
                                     command=start_command, width=15, height=2)
        self.start_button.pack(side=tk.TOP, padx=10, pady=5)

        self.options_button = tk.Button(self.button_frame, text=self.language.OPTIONS_BUTTON_TEXT, 
                                      command=options_command, width=15, height=2)
        self.options_button.pack(side=tk.TOP, padx=10, pady=5)

        self.statistics_button = tk.Button(self.button_frame, text=self.language.STATISTICS_BUTTON_TEXT, 
                                         command=statistics_command, width=15, height=2)
        self.statistics_button.pack(side=tk.TOP, padx=10, pady=5)

        self.exit_button = tk.Button(self.button_frame, text=self.language.EXIT_BUTTON_TEXT, 
                                    command=exit_command, width=15, height=2)
        self.exit_button.pack(side=tk.TOP, padx=10, pady=5)

        return self.button_frame

    def update_buttons(self, language: object) -> None:
        """
        Update button text with new language.

        Updates the text of all buttons to match the newly selected language.

        Args:
            language (object): Language module containing button text strings
        """
        self.start_button.config(text=language.START_BUTTON_TEXT)
        self.options_button.config(text=language.OPTIONS_BUTTON_TEXT)
        self.statistics_button.config(text=language.STATISTICS_BUTTON_TEXT)
        self.exit_button.config(text=language.EXIT_BUTTON_TEXT)