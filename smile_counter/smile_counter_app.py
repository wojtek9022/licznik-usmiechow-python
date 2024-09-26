import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
from src.video_capture import VideoCapture
from src.smile_detector import SmileDetector
from src.fps_calculator import FPSCalculator
import os

# Import configuration constants
from src.config import FONT, FACE_SCALE_FACTOR, FACE_MIN_NEIGHBOURS, SMILE_SCALE_FACTOR, SMILE_MIN_NEIGHBOURS, TIME_TO_START_COUNTING

class SmileCounterApp:
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
        self.subtitle.pack_forget()
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
        # Create a new window for options
        options_window = tk.Toplevel(self.master)
        options_window.title("Options")

        # Option settings
        tk.Label(options_window, text="Face Scale Factor:").grid(row=0, column=0, padx=10, pady=5)
        face_scale_entry = tk.Entry(options_window)
        face_scale_entry.grid(row=0, column=1, padx=10, pady=5)
        face_scale_entry.insert(0, str(FACE_SCALE_FACTOR))  # Insert current value

        tk.Label(options_window, text="Face Min Neighbours:").grid(row=1, column=0, padx=10, pady=5)
        face_min_neighbours_entry = tk.Entry(options_window)
        face_min_neighbours_entry.grid(row=1, column=1, padx=10, pady=5)
        face_min_neighbours_entry.insert(0, str(FACE_MIN_NEIGHBOURS))  # Insert current value

        tk.Label(options_window, text="Smile Scale Factor:").grid(row=2, column=0, padx=10, pady=5)
        smile_scale_entry = tk.Entry(options_window)
        smile_scale_entry.grid(row=2, column=1, padx=10, pady=5)
        smile_scale_entry.insert(0, str(SMILE_SCALE_FACTOR))  # Insert current value

        tk.Label(options_window, text="Smile Min Neighbours:").grid(row=3, column=0, padx=10, pady=5)
        smile_min_neighbours_entry = tk.Entry(options_window)
        smile_min_neighbours_entry.grid(row=3, column=1, padx=10, pady=5)
        smile_min_neighbours_entry.insert(0, str(SMILE_MIN_NEIGHBOURS))  # Insert current value

        tk.Label(options_window, text="Time to Start Counting:").grid(row=4, column=0, padx=10, pady=5)
        time_to_start_entry = tk.Entry(options_window)
        time_to_start_entry.grid(row=4, column=1, padx=10, pady=5)
        time_to_start_entry.insert(0, str(TIME_TO_START_COUNTING))  # Insert current value

        # Button to save changes
        save_button = tk.Button(options_window, text="Save", command=lambda: self.save_options(
            face_scale_entry.get(), face_min_neighbours_entry.get(),
            smile_scale_entry.get(), smile_min_neighbours_entry.get(),
            time_to_start_entry.get()
        ))
        save_button.grid(row=5, columnspan=2, padx=10, pady=10)

    def save_options(self, face_scale, face_min_neighbours, smile_scale, smile_min_neighbours, time_to_start):
        # Convert values and save to configuration file
        try:
            face_scale = float(face_scale)
            face_min_neighbours = int(face_min_neighbours)
            smile_scale = float(smile_scale)
            smile_min_neighbours = int(smile_min_neighbours)
            time_to_start = float(time_to_start)

            # Ensure the 'src' directory exists
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'config.py')
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            # Save to file
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

            messagebox.showinfo("Success", "Options saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving options: {e}")

    def on_closing(self):
        if self.video_capture:
            self.video_capture.release()
        self.master.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SmileCounterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
