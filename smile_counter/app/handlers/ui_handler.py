import tkinter as tk
from PIL import Image, ImageTk
import os
from typing import Callable
from ..src.data.lang import lang_pl, lang_en
from .config_handler import ConfigHandler
from .button_handler import ButtonHandler
from .video_handler import VideoHandler
from .options_handler import OptionsHandler
from .statistics_handler import StatisticsHandler
from app.src.utils.config_utils.version_handler import VersionHandler

class UIHandler:
    # FIXME: Refactor this class, Its too long
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

    def __init__(self, master: tk.Tk) -> None:
        """Initialize UI Handler"""
        self.master = master
        self.language = lang_pl
        self._setup_window_properties()
        self.config_handler = ConfigHandler()
        self.language = self._load_language()
        
        # Initialize handlers - fix naming
        self.button_handler = ButtonHandler(self.master, self.language)
        self.options_handler = OptionsHandler(self.master, self.language)  # Changed from options to options_handler
        self.video_handler = None
        self.statistics_handler = StatisticsHandler(self.master, self.language)
        
        # Add after other initializations
        self.version = VersionHandler.get_version()
        
        # Setup UI components
        self.header = None
        self.subtitle = None
        self.logo_label = None
        self.button_frame = None
        self.language_frame = None
        
        self._setup_main_window()
        self._load_images()

    def _setup_window_properties(self) -> None:
        """Setup main window size and scaling properties"""
        self.master.title(self.language.TITLE_TEXT)
        # Set initial window size
        self.master.geometry("800x800")
        # Set minimum window size
        self.master.minsize(800, 700)
        
        # Configure window scaling
        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=1)

    def start_video(self) -> None:
        """Start video capture mode."""
        ConfigHandler().get_config()  # Force config reload
        if not self.video_handler:
            self.video_handler = VideoHandler(
                master=self.master,
                language=self.language,  # Add language parameter
                ui_handler=self
            )
        self.video_handler.start_video()

    def cleanup(self) -> None:
        """Clean up resources before closing"""
        if self.video_handler:
            self.video_handler.stop_video()

    def show_options(self) -> None:
        """Show options configuration window with loading message"""
        # Create loading label
        loading_frame = tk.Frame(self.master)
        loading_frame.place(relx=0.02, rely=0.95, anchor="sw")  # Position at bottom left
        
        loading_text = self.language.OPTIONS_LOADING_TEXT if hasattr(self.language, 'OPTIONS_LOADING_TEXT') else "Loading options, please wait..."
        loading_label = tk.Label(
            loading_frame, 
            text=loading_text,
            font=("Helvetica", 14, "bold"),
            fg="blue"
        )
        loading_label.pack(pady=10, padx=2)

        # Update GUI to show loading message
        self.master.update()

        # Show options window
        self.options_handler.show_options()

        # Remove loading message
        loading_frame.destroy()

    def show_statistics(self) -> None:
        """Show statistics window"""
        self.statistics_handler.show_statistics()

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
        self.subtitle.config(text=self.language.VERSION_TEXT.format(version=self.version))
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self.options_handler.update_language(self.language)  # Changed from options to options_handler

    def _setup_main_window(self) -> None:
        self.master.title(self.language.TITLE_TEXT)
        self.master.geometry("800x600")

    def _load_images(self) -> None:
        current_dir: str = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        #FIXME: Refactor this image paths
        # Get handlers directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Get app directory
        app_dir = os.path.dirname(current_dir)
        # Build paths to images in src/img
        self.icon_path = os.path.join(app_dir, 'src', 'data', 'img', 'menu', 'icon.ico')
        self.flag_en_path = os.path.join(app_dir, 'src', 'data', 'img', 'menu', 'flag_en.png')
        self.flag_pl_path = os.path.join(app_dir, 'src', 'data', 'img', 'menu', 'flag_pl.png')
        self.logo_path: str = os.path.join(app_dir, 'src', 'data', 'img', 'menu', 'main_menu_logo.png')
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
        self.subtitle: tk.Label = tk.Label(self.master, text=self.language.VERSION_TEXT.format(version=self.version), font=("Helvetica", 12))
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
        # Use format to insert version
        self.subtitle.config(text=self.language.VERSION_TEXT.format(version=self.version))
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self.options.update_language(self.language)  # Use direct update method

    def create_main_menu(self, start_video_callback: Callable, show_options_callback: Callable, 
                        show_statistics_callback: Callable, exit_callback: Callable) -> None:
        """Create and display the main menu interface."""
        self.header, self.subtitle, self.logo_label = self.create_header()
        self.button_frame = self.button_handler.create_buttons(
            start_video_callback,
            show_options_callback,
            show_statistics_callback,
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
        self.subtitle.config(text=self.language.VERSION_TEXT.format(version=self.version))
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        
        # Update options window if exists
        if hasattr(self.options.ui, 'window') and self.options.ui.window:
            self._update_options_language(language)

    def _update_options_language(self, language: object) -> None:
        """Update options window text elements."""
        if self.options_handler.ui.window and tk.Toplevel.winfo_exists(self.options_handler.ui.window):
            self.options_handler.update_language(language)

    def on_language_change(self, language: object) -> None:
        """
        Handle language change event.

        Args:
            language (object): New language module
        """
        self.language = language
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT.format(version=self.version))
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)
        self.button_handler.update_buttons(self.language)
        self._update_options_if_exists(language)