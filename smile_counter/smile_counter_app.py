import tkinter as tk
import tkinter.messagebox  # Required for building an executable
from app.handlers.initialize_handler import get_initialized_config
get_initialized_config()  

from app.handlers.ui_handler import UIHandler

class SmileCounterApp:
    """
    Main application class for the Smile Counter application.
    Delegates all UI and functionality handling to UIHandler.
    """
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.ui_handler = UIHandler(self.master)
        self._init_ui()

    def _init_ui(self) -> None:
        """Initialize UI components through UIHandler."""
        self.ui_handler.create_main_menu(
            self.start_video,
            self.show_options,
            self.on_closing
        )

    def change_language(self, lang_code: str) -> None:
        """Change application language through UIHandler."""
        self.ui_handler.change_language(lang_code)

    def on_closing(self) -> None:
        """Handle application closing."""
        self.ui_handler.cleanup()
        self.master.destroy()

    def start_video(self) -> None:
        """Start video capture through UIHandler."""
        self.ui_handler.start_video()

    def show_options(self) -> None:
        """Show options window through UIHandler."""
        self.ui_handler.show_options()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()