import tkinter as tk
from PIL import Image, ImageTk
import os
from typing import Callable
from .src.lang import lang_pl, lang_en
from app.config_handler import ConfigHandler

class UIHandler:
    """
    Manages user interface elements and layout.

    Attributes:
        master (tk.Tk): Main application window
        language (object): Current language module
        button_handler (ButtonHandler): Button creation and management
        options (OptionsManager): Application configuration manager
        header (tk.Label): Main application header
        subtitle (tk.Label): Version information label
        logo_label (tk.Label): Application logo display
        button_frame (tk.Frame): Container for control buttons
        language_frame (tk.Frame): Container for language selection
        logo_path (str): Path to logo image file
        icon_path (str): Path to application icon
        flag_en_path (str): Path to English flag image
        flag_pl_path (str): Path to Polish flag image
    """

    def __init__(self, master: tk.Tk, button_handler: object, options: object) -> None:
        self.master = master
        self.config_handler = ConfigHandler()
        self.language = self._load_language()
        self.button_handler = button_handler
        self.options = options
        self.header = None
        self.subtitle = None
        self.logo_label = None
        self.button_frame = None
        self.language_frame = None
        self._setup_main_window()
        self._load_images()

    def _load_language(self) -> object:
        """Load language based on config or default to English."""
        config = self.config_handler.get_config()
        try:
            # Access LANGUAGE as attribute instead of using get()
            language = getattr(config, 'LANGUAGE', 'en')
            return lang_pl if language == 'pl' else lang_en
        except Exception as e:
            print(f"Error loading language: {e}. Defaulting to English.")
            return lang_en

    def change_language(self, lang_code: str) -> None:
        """
        Central method for changing application language.
        
        Args:
            lang_code (str): Language code ('en' or 'pl')
        """
        self.language = lang_pl if lang_code == "pl" else lang_en
        self.config_handler.update_config({"LANGUAGE": lang_code})
        self._update_all_ui()

    def _update_all_ui(self) -> None:
        """Update all UI components with new language."""
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT)
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self.options.update_language(self.language)

    def _setup_main_window(self) -> None:
        self.master.title(self.language.TITLE_TEXT)
        self.master.geometry("800x600")

    def _load_images(self) -> None:
        current_dir: str = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        self.logo_path: str = os.path.join(current_dir, 'img', 'main_menu_logo.png')
        self.icon_path: str = os.path.join(current_dir, 'img', 'icon.ico')
        self.flag_en_path: str = os.path.join(current_dir, 'img', 'flag_en.png')  # Placeholder for English flag
        self.flag_pl_path: str = os.path.join(current_dir, 'img', 'flag_pl.png')  # Placeholder for Polish flag
        icon_image: Image.Image = Image.open(self.icon_path)
        icon_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(icon_image)
        self.master.iconphoto(True, icon_photo)

    def create_header(self) -> tuple[tk.Label, tk.Label, tk.Label]:
        """
        Create and configure the application header section.

        Creates main title, subtitle, and logo elements with appropriate
        styling and positioning in the main window.

        Returns:
            tuple[tk.Label, tk.Label, tk.Label]: Header, subtitle, and logo labels
        """
        self.header: tk.Label = tk.Label(self.master, text=self.language.TITLE_TEXT, font=("Helvetica", 24))
        self.header.pack(pady=20)
        self.subtitle: tk.Label = tk.Label(self.master, text=self.language.VERSION_TEXT, font=("Helvetica", 12))
        self.subtitle.pack(pady=5)
        self.logo_label = self._load_logo()
        return self.header, self.subtitle, self.logo_label

    def _load_logo(self) -> tk.Label:
        try:
            if os.path.exists(self.logo_path):
                print(f"Loading logo from: {self.logo_path}")  # Debugging line
                logo_image: Image.Image = Image.open(self.logo_path)
                logo_image = logo_image.resize((200, 200), Image.LANCZOS)  # Resize logo if necessary
                logo_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(logo_image)
                logo_label: tk.Label = tk.Label(self.master, image=logo_photo)  # Use image in Label
                logo_label.image = logo_photo  # Keep a reference to avoid garbage collection
            else:
                raise FileNotFoundError(f"Logo file not found: {self.logo_path}")
        except Exception as e:
            print(f"Error loading logo: {e}")
            logo_label: tk.Label = tk.Label(self.master, text=self.language.LOGO_NOT_FOUND_TEXT, font=("Helvetica", 16))  # Fallback text
        logo_label.pack(pady=10)
        return logo_label

    def create_language_buttons(self, change_language_callback) -> tk.Frame:
        language_frame: tk.Frame = tk.Frame(self.master)
        language_frame.pack(pady=10)

        try:
            flag_en_image: Image.Image = Image.open(self.flag_en_path).resize((30, 20), Image.LANCZOS)
            self.flag_en_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(flag_en_image)
            self.flag_en_button: tk.Button = tk.Button(language_frame, image=self.flag_en_photo, command=lambda: change_language_callback('en'))
        except FileNotFoundError:
            self.flag_en_button: tk.Button = tk.Button(language_frame, text="English", command=lambda: change_language_callback('en'))
        self.flag_en_button.pack(side=tk.LEFT, padx=5)

        try:
            flag_pl_image: Image.Image = Image.open(self.flag_pl_path).resize((30, 20), Image.LANCZOS)
            self.flag_pl_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(flag_pl_image)
            self.flag_pl_button: tk.Button = tk.Button(language_frame, image=self.flag_pl_photo, command=lambda: change_language_callback('pl'))
        except FileNotFoundError:
            self.flag_pl_button: tk.Button = tk.Button(language_frame, text="Polski", command=lambda: change_language_callback('pl'))
        self.flag_pl_button.pack(side=tk.LEFT, padx=5)

        return language_frame

    def _refresh_ui(self) -> None:
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT)
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self.options.update_language(self.language)  # Use direct update method

    def create_main_menu(self, start_video_callback: Callable, show_options_callback: Callable, exit_callback: Callable) -> None:
        """Create and display the main menu interface."""
        self.header, self.subtitle, self.logo_label = self.create_header()
        self.button_frame = self.button_handler.create_buttons(
            start_video_callback,
            show_options_callback,
            exit_callback
        )
        self.language_frame = self.create_language_buttons(self.change_language)

    def hide_main_menu(self) -> None:
        """Hide all main menu components."""
        if self.header: self.header.pack_forget()
        if self.subtitle: self.subtitle.pack_forget()
        if self.logo_label: self.logo_label.pack_forget()
        if hasattr(self.button_handler, 'button_frame'):
            self.button_handler.button_frame.pack_forget()
        if self.language_frame: self.language_frame.pack_forget()

    def show_main_menu(self) -> None:
        """Restore main menu components."""
        if self.header: self.header.pack(pady=20)
        if self.subtitle: self.subtitle.pack(pady=5)
        if self.logo_label: self.logo_label.pack(pady=10)
        if hasattr(self.button_handler, 'button_frame'):
            self.button_handler.button_frame.pack(pady=20)
        if self.language_frame: self.language_frame.pack(pady=10)

    def update_language(self, language: object) -> None:
        """
        Update all UI text elements with new language.

        Updates main window and options window text elements
        to display content in the newly selected language.

        Args:
            language (object): Language module containing text strings
        """
        self.language = language
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT)
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        
        # Update options window if exists
        if hasattr(self.options.ui, 'window') and self.options.ui.window:
            self._update_options_language(language)

    def _update_options_language(self, language: object) -> None:
        """Update options window text elements."""
        window = self.options.ui.window
        if window and tk.Toplevel.winfo_exists(window):
            window.title(language.OPTIONS_TITLE_TEXT)
            # Update labels
            for option_name, label in self.options.ui.labels.items():
                label_text = getattr(language, f'{option_name}_TEXT')
                label.config(text=label_text)
            # Update save button
            if hasattr(self.options.ui, 'save_button'):
                self.options.ui.save_button.config(text=language.SAVE_BUTTON_TEXT)

    def on_language_change(self, language: object) -> None:
        """
        Handle language change event.

        Args:
            language (object): New language module
        """
        self.language = language
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT)
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self._update_options_if_exists(language)