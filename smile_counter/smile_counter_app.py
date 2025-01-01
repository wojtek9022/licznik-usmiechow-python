import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from src.video_capture import VideoCapture
from src.smile_detector import SmileDetector
from src.fps_calculator import FPSCalculator
import os
from typing import Optional

# Import configuration constants
from src.config import FONT, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBOURS, SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBOURS, TIME_TO_START_COUNTING, LANGUAGE

# Import language files
import src.lang.lang_en as lang_en
import src.lang.lang_pl as lang_pl

class SmileCounterApp:
    def __init__(self, master: tk.Tk) -> None:
        self.master: tk.Tk = master
        self.language = self._load_language()
        self._setup_main_window()
        self._load_images()
        self._create_header()
        self._create_buttons()
        self._create_language_buttons()

        self.video_frame: Optional[tk.Frame] = None
        self.canvas: Optional[tk.Canvas] = None
        self.video_capture: Optional[VideoCapture] = None
        self.smile_detector: SmileDetector = SmileDetector()
        self.fps_calculator: FPSCalculator = FPSCalculator()
        self.smiles_detected: int = 0
        self.running: bool = False

    def _setup_main_window(self) -> None:
        self.master.title(self.language.TITLE_TEXT)
        self.master.geometry("800x600")

    def _load_images(self) -> None:
        current_dir: str = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        self.logo_path: str = os.path.join(current_dir, 'src', 'data', 'img', 'main_menu_logo.png')
        self.icon_path: str = os.path.join(current_dir, 'src', 'data', 'img', 'icon.ico')
        self.flag_en_path: str = os.path.join(current_dir, 'src', 'data', 'img', 'flag_en.png')  # Placeholder for English flag
        self.flag_pl_path: str = os.path.join(current_dir, 'src', 'data', 'img', 'flag_pl.png')  # Placeholder for Polish flag
        icon_image: Image.Image = Image.open(self.icon_path)
        icon_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(icon_image)
        self.master.iconphoto(True, icon_photo)

    def _create_header(self) -> None:
        self.header: tk.Label = tk.Label(self.master, text=self.language.TITLE_TEXT, font=("Helvetica", 24))
        self.header.pack(pady=20)
        self.subtitle: tk.Label = tk.Label(self.master, text=self.language.VERSION_TEXT, font=("Helvetica", 12))
        self.subtitle.pack(pady=5)
        self._load_logo()

    def _load_logo(self) -> None:
        if os.path.exists(self.logo_path):
            self.logo_image: Image.Image = Image.open(self.logo_path)
            self.logo_image = self.logo_image.resize((200, 200), Image.LANCZOS)  # Resize logo if necessary
            self.logo_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(self.logo_image)
            self.logo_label: tk.Label = tk.Label(self.master, image=self.logo_photo)  # Use image in Label
        else:
            self.logo_label: tk.Label = tk.Label(self.master, text=self.language.LOGO_NOT_FOUND_TEXT, font=("Helvetica", 16))  # Fallback text
        self.logo_label.pack(pady=10)

    def _create_buttons(self) -> None:
        self.button_frame: tk.Frame = tk.Frame(self.master)
        self.button_frame.pack(pady=20)

        self.start_button: tk.Button = tk.Button(self.button_frame, text=self.language.START_BUTTON_TEXT, command=self.start_video, width=15, height=2)
        self.start_button.pack(side=tk.TOP, padx=10, pady=5)

        self.options_button: tk.Button = tk.Button(self.button_frame, text=self.language.OPTIONS_BUTTON_TEXT, command=self.show_options, width=15, height=2)
        self.options_button.pack(side=tk.TOP, padx=10, pady=5)

        self.exit_button: tk.Button = tk.Button(self.button_frame, text=self.language.EXIT_BUTTON_TEXT, command=self.on_closing, width=15, height=2)
        self.exit_button.pack(side=tk.TOP, padx=10, pady=5)

    def _create_language_buttons(self) -> None:
        self.language_frame: tk.Frame = tk.Frame(self.master)
        self.language_frame.pack(pady=10)

        try:
            flag_en_image: Image.Image = Image.open(self.flag_en_path).resize((30, 20), Image.LANCZOS)
            self.flag_en_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(flag_en_image)
            self.flag_en_button: tk.Button = tk.Button(self.language_frame, image=self.flag_en_photo, command=lambda: self.change_language('en'))
        except FileNotFoundError:
            self.flag_en_button: tk.Button = tk.Button(self.language_frame, text="English", command=lambda: self.change_language('en'))
        self.flag_en_button.pack(side=tk.LEFT, padx=5)

        try:
            flag_pl_image: Image.Image = Image.open(self.flag_pl_path).resize((30, 20), Image.LANCZOS)
            self.flag_pl_photo: ImageTk.PhotoImage = ImageTk.PhotoImage(flag_pl_image)
            self.flag_pl_button: tk.Button = tk.Button(self.language_frame, image=self.flag_pl_photo, command=lambda: self.change_language('pl'))
        except FileNotFoundError:
            self.flag_pl_button: tk.Button = tk.Button(self.language_frame, text="Polski", command=lambda: self.change_language('pl'))
        self.flag_pl_button.pack(side=tk.LEFT, padx=5)

    def start_video(self) -> None:
        self._hide_main_menu()
        self._setup_video_frame()
        self.video_capture = VideoCapture()
        self.running = True
        self.update_frame()

    def _hide_main_menu(self) -> None:
        self.header.pack_forget()
        self.subtitle.pack_forget()
        self.logo_label.pack_forget()
        self.button_frame.pack_forget()
        self.language_frame.pack_forget()

    def _setup_video_frame(self) -> None:
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def update_frame(self) -> None:
        if self.running:
            ret, frame = self.video_capture.read()
            if ret:
                self._process_frame(frame)
                self.canvas.after(10, self.update_frame)

    def _process_frame(self, frame: cv2.Mat) -> None:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.smile_detector.detect_faces(gray_frame)

        for (face_x, face_y, face_w, face_h) in faces:
            self.smile_detector.draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))
            face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]
            smiles = self.smile_detector.detect_smiles(face_region)
            smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]

            smile_detected = len(smiles) > 0
            self.smile_detector.handle_smile_and_draw(smile_detected, frame, smiles, face_x, face_y)
            self.smiles_detected = self.smile_detector.smiles_detected

        self._display_frame(frame)

    def _display_frame(self, frame: cv2.Mat) -> None:
        frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        self.canvas.imgtk = imgtk

        text_to_show = self.language.DETECTED_SMILES_TEXT.format(count=self.smiles_detected)
        self.canvas.create_text(10, 10, anchor=tk.NW, text=text_to_show, fill="red", font=("Helvetica", 16))

    def show_options(self) -> None:
        options_window = tk.Toplevel(self.master)
        options_window.title(self.language.OPTIONS_TITLE_TEXT)
        self._create_options_entries(options_window)
        self._create_save_button(options_window)

    def _create_options_entries(self, options_window: tk.Toplevel) -> None:
        self._create_option_entry(options_window, self.language.FACE_SCALE_FACTOR_TEXT, FACE_SCALE_FACTOR, 0)
        self._create_option_entry(options_window, self.language.FACE_MIN_NEIGHBOURS_TEXT, FACE_MIN_NEIGHBOURS, 1)
        self._create_option_entry(options_window, self.language.SMILE_SCALE_FACTOR_TEXT, SMILE_SCALE_FACTOR, 2)
        self._create_option_entry(options_window, self.language.SMILE_MIN_NEIGHBOURS_TEXT, SMILE_MIN_NEIGHBOURS, 3)
        self._create_option_entry(options_window, self.language.TIME_TO_START_COUNTING_TEXT, TIME_TO_START_COUNTING, 4)

    def _create_option_entry(self, options_window: tk.Toplevel, label_text: str, value: float, row: int) -> None:
        tk.Label(options_window, text=label_text).grid(row=row, column=0, padx=10, pady=5)
        entry = tk.Entry(options_window)
        entry.grid(row=row, column=1, padx=10, pady=5)
        entry.insert(0, str(value))

    def _create_save_button(self, options_window: tk.Toplevel) -> None:
        save_button = tk.Button(options_window, text=self.language.SAVE_BUTTON_TEXT, command=lambda: self.save_options(
            options_window.children['!entry'].get(), options_window.children['!entry2'].get(),
            options_window.children['!entry3'].get(), options_window.children['!entry4'].get(),
            options_window.children['!entry5'].get()
        ))
        save_button.grid(row=5, columnspan=2, padx=10, pady=10)

    def change_language(self, lang_code: str) -> None:
        if lang_code == "pl":
            self.language = lang_pl
        else:
            self.language = lang_en
        self._save_language(lang_code)
        self._refresh_ui()

    def _refresh_ui(self) -> None:
        self.master.title(self.language.TITLE_TEXT)
        self.header.config(text=self.language.TITLE_TEXT)
        self.subtitle.config(text=self.language.VERSION_TEXT)
        self.start_button.config(text=self.language.START_BUTTON_TEXT)
        self.options_button.config(text=self.language.OPTIONS_BUTTON_TEXT)
        self.exit_button.config(text=self.language.EXIT_BUTTON_TEXT)
        self.logo_label.config(text=self.language.LOGO_NOT_FOUND_TEXT)

    def save_options(self, face_scale: str, face_min_neighbours: str, smile_scale: str, smile_min_neighbours: str, time_to_start: str) -> None:
        try:
            face_scale = float(face_scale)
            face_min_neighbours = int(face_min_neighbours)
            smile_scale = float(smile_scale)
            smile_min_neighbours = int(smile_min_neighbours)
            time_to_start = float(time_to_start)

            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'config.py')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            with open(config_path, 'w') as f:
                f.write(f"# Configuration Constants\n")
                f.write(f"FACE_SCALE_FACTOR = {face_scale}\n")
                f.write(f"FACE_MIN_NEIGHBOURS = {face_min_neighbours}\n")
                f.write(f"SMILE_SCALE_FACTOR = {smile_scale}\n")
                f.write(f"SMILE_MIN_NEIGHBOURS = {smile_min_neighbours}\n")
                f.write(f"TIME_TO_START_COUNTING = {time_to_start}\n\n")
                f.write(f"# Font Configuration\n")
                f.write(f"FONT = {{\n")
                f.write(f'    "font": "HERSHEY_SIMPLEX",\n')
                f.write(f'    "scale": 1,\n')
                f.write(f'    "color": (255, 0, 0),\n')
                f.write(f'    "thickness": 3,\n')
                f.write(f'    "line_type": 2\n')
                f.write(f"}}\n")
                f.write(f"# Language Configuration\n")
                f.write(f"LANGUAGE = '{self._get_current_language_code()}'\n")

            messagebox.showinfo("Success", self.language.SUCCESS_MESSAGE_TEXT)
        except Exception as e:
            messagebox.showerror("Error", self.language.ERROR_MESSAGE_TEXT.format(error=e))

    def on_closing(self) -> None:
        if self.video_capture:
            self.video_capture.release()
        self.master.destroy()

    def _load_language(self) -> object:
        try:
            from src.config import LANGUAGE
            if LANGUAGE == 'pl':
                return lang_pl
        except ImportError:
            pass
        return lang_en

    def _save_language(self, lang_code: str) -> None:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'config.py')
        with open(config_path, 'a') as f:
            f.write(f"LANGUAGE = '{lang_code}'\n")

    def _get_current_language_code(self) -> str:
        if self.language == lang_pl:
            return 'pl'
        return 'en'


if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
