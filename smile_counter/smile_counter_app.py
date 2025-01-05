import tkinter as tk
import tkinter.messagebox  # Required for building with pyinstaller
from app.initialize import ensure_app_initialized
ensure_app_initialized()  # Initialize before other imports

from app.button_handler import ButtonHandler
from app.options_handler import OptionsHandler as Options
from app.video_handler import VideoHandler
from app.ui_handler import UIHandler

class SmileCounterApp:
    """
    Main application class for the Smile Counter application.

    This class manages the main GUI window and coordinates all UI components,
    video handling, and language management. It follows the MVC pattern where
    this class acts as the controller.

    Attributes:
        master (tk.Tk): Main application window
        options (Options): Application settings and configuration manager
        language (object): Current language module with text strings
        button_handler (ButtonHandler): Handles button creation and management
        ui_handler (UIHandler): Manages UI components and their updates
        header (tk.Label): Main application header
        subtitle (tk.Label): Application version subtitle
        logo_label (tk.Label): Application logo display
        button_frame (tk.Frame): Container for main buttons
        language_frame (tk.Frame): Container for language selection buttons
        video_handler (VideoHandler): Manages video capture and processing
    """
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.button_handler = ButtonHandler(self.master, None)  # Initially no language
        self.options = Options(self.master, None)  # Initially no language
        self.ui_handler = UIHandler(self.master, self.button_handler, self.options)
        self.language = self.ui_handler.language  # Get language from UIHandler
        self.button_handler.language = self.language  # Update ButtonHandler with language
        self.options.update_language(self.language)  # Update Options with language
        self._init_ui()

    def change_language(self, lang_code: str) -> None:
        """
        Change application language through UIHandler.
        
        Args:
            lang_code (str): Language code to switch to ('en' or 'pl')
        """
        self.ui_handler.change_language(lang_code)
        self.language = self.ui_handler.language

    def _init_ui(self) -> None:
        self.header, self.subtitle, self.logo_label = self.ui_handler.create_header()
        self.button_frame = self.button_handler.create_buttons(
            self.start_video, 
            self.show_options,
            self.on_closing
        )
        self.language_frame = self.ui_handler.create_language_buttons(self.change_language)

    def on_closing(self) -> None:
        """
        Handle application closing.

        Releases video capture resources if active and closes application window.
        """
        if self.video_handler.video_capture_wrapper:
            self.video_handler.video_capture_wrapper.release()
        self.master.destroy()

    def start_video(self) -> None:
        """
        Start video capture and smile detection.
        
        Initializes video capture, hides main menu, and starts
        the smile detection process.
        """
        self.video_handler = VideoHandler(
            self.master, 
            self.language,
            self.header,
            self.subtitle,
            self.logo_label,
            self.button_frame,
            self.language_frame
        )
        self.video_handler.start_video()

    def show_options(self) -> None:
        """
        Display the options configuration window.

        Opens a new window allowing user to modify application settings
        like face detection parameters and smile detection sensitivity.
        """
        self.options.show_options()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()