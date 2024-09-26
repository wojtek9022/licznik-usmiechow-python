import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
from src.video_capture import VideoCapture
from src.smile_detector import SmileDetector
from src.fps_calculator import FPSCalculator
from src.config import FONT
import os

class SmileCounterApp:
    # FIXME: This code needs refactoring e.g. using SOLID principles
    def __init__(self, master):
        self.master = master
        self.master.title("Smile Counter 2.0")
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script directory
        self.logo_path = os.path.join(current_dir, 'src', 'data', 'img', 'main_menu_logo.png')
        self.icon_path = os.path.join(current_dir, 'src', 'data', 'img', 'icon.ico')
        icon_image = Image.open(self.icon_path)
        icon_photo = ImageTk.PhotoImage(icon_image)
        self.master.iconphoto(True, icon_photo)
        self.master.geometry("800x600")

        # Header
        self.header = tk.Label(self.master, text="Smile Counter 2.0", font=("Helvetica", 24))
        self.header.pack(pady=20)
        self.subtitle = tk.Label(self.master, text="Version: 2.0.0", font=("Helvetica", 12))
        self.subtitle.pack(pady=5)

        # Load and display the logo image if it exists
        if os.path.exists(self.logo_path):
            self.logo_image = Image.open(self.logo_path)
            self.logo_image = self.logo_image.resize((200, 200), Image.LANCZOS)  # Resize logo if necessary
            self.logo_photo = ImageTk.PhotoImage(self.logo_image)
            self.logo_label = tk.Label(self.master, image=self.logo_photo)  # Use image in Label
        else:
            self.logo_label = tk.Label(self.master, text="[Logo not found]", font=("Helvetica", 16))  # Fallback text
        self.logo_label.pack(pady=10)

        # Frame for buttons
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=20)

        # Start button
        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_video, width=15, height=2)
        self.start_button.pack(side=tk.TOP, padx=10, pady=5)

        # Options button
        self.options_button = tk.Button(self.button_frame, text="Options", command=self.show_options, width=15, height=2)
        self.options_button.pack(side=tk.TOP, padx=10, pady=5)

        # Exit button
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.on_closing, width=15, height=2)
        self.exit_button.pack(side=tk.TOP, padx=10, pady=5)

        self.video_frame = None
        self.canvas = None
        self.video_capture = None
        self.smile_detector = SmileDetector()
        self.fps_calculator = FPSCalculator()
        self.smiles_detected = 0
        self.running = False

    def start_video(self):
        # Hide buttons and header
        self.header.pack_forget()
        self.logo_label.pack_forget()
        self.button_frame.pack_forget()

        # Frame for video
        self.video_frame = tk.Frame(self.master)
        self.video_frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.video_frame, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Start video capture
        self.video_capture = VideoCapture()
        self.running = True
        self.update_frame()

    def update_frame(self):
        # Update video frame
        if self.running:
            ret, frame = self.video_capture.read()
            if ret:
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

                frame = cv2.resize(frame, (self.canvas.winfo_width(), self.canvas.winfo_height()))
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
                self.canvas.imgtk = imgtk

                text_to_show = f"Detected smiles: {self.smiles_detected}"
                self.canvas.create_text(10, 10, anchor=tk.NW, text=text_to_show, fill="red", font=("Helvetica", 16))

                self.canvas.after(10, self.update_frame)

    def show_options(self):
        messagebox.showinfo("Options", "Options menu is not implemented yet.")

    def on_closing(self):
        if self.video_capture:
            self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
