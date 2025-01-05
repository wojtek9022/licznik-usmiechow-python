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
        self.options = Options(self.master)
        self.language = self.options.language
        self.button_handler = ButtonHandler(self.master, self.language)
        self.ui_handler = UIHandler(self.master, self.language, self.button_handler, self.options)
        self._init_ui()

    def change_language(self, lang_code: str) -> None:
        """
        Change application language and update all UI elements.

        Changes the application's language based on the provided language code
        and updates all UI components to display text in the new language.

        Args:
            lang_code (str): Language code to switch to ('en' or 'pl')
        """
        # FIXME: Too many dependecies
        self.options.change_language(lang_code)
        self.language = self.options.language
        self.ui_handler.language = self.language
        self.ui_handler._refresh_ui()

    def _init_ui(self) -> None:
        self.header, self.subtitle, self.logo_label = self.ui_handler.create_header()
        self.button_frame = self.button_handler.create_buttons(
            self.start_video, 
            self.show_options,
            self.on_closing
        )
        self.language_frame = self.ui_handler.create_language_buttons(self.change_language)
        self.video_handler = VideoHandler(
            self.master,
            self.language,
            self.header,
            self.subtitle,
            self.logo_label,
            self.button_frame,
            self.language_frame
        )

    def on_closing(self) -> None:
        """
        Handle application closing.

        Properly releases video capture resources if active and
        closes the application window.
        """
        if self.video_handler.video_capture:
            self.video_handler.video_capture.release()
        self.master.destroy()

    def start_video(self) -> None:
        """
        Start video capture and smile detection.
        
        Initializes video capture, hides main menu, and starts
        the smile detection process.
        """
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