import tkinter as tk
from PIL import Image, ImageTk
import os

class UIHandler:
    def __init__(self, master: tk.Tk, language: object, button_creator: object, options: object) -> None:
        self.master = master
        self.language = language
        self.button_creator = button_creator
        self.options = options
        self._setup_main_window()
        self._load_images()

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

    def create_header(self) -> tuple:
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
        self.button_creator.update_buttons(self.language)
        self.options.update_language(self.language)  # Use direct update method