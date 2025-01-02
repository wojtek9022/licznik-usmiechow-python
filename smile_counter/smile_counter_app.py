import tkinter as tk
from app.initialize import ensure_app_initialized
ensure_app_initialized()  # Initialize before other imports

from app.buttons import ButtonCreator
from app.options import Options
from app.language import LanguageManager
from app.video import VideoHandler
from app.ui import UIHandler
from app.config_manager import ConfigManager

class SmileCounterApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master = master
        self.config_manager = ConfigManager()
        
        self.language_manager = LanguageManager()
        self.language = self.language_manager.language
        
        self.button_creator = ButtonCreator(self.master, self.language)
        self.options = Options(self.master, self.language, self.language_manager)
        self.ui_handler = UIHandler(self.master, self.language, self.button_creator, self.options)
        
        self._init_ui()

    def change_language(self, lang_code: str) -> None:
        self.language_manager.change_language(lang_code)
        self.language = self.language_manager.language
        self.ui_handler.language = self.language
        self.options.language = self.language
        self.ui_handler._refresh_ui()

    def _init_ui(self) -> None:
        self.header, self.subtitle, self.logo_label = self.ui_handler.create_header()
        self.button_frame = self.button_creator.create_buttons(
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
        if self.video_handler.video_capture:
            self.video_handler.video_capture.release()
        self.master.destroy()

    def start_video(self) -> None:
        self.video_handler.start_video()

    def show_options(self) -> None:
        self.options.show_options()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()