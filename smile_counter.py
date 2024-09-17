import cv2
import time

# Constants
TIME_TO_START_COUNTING = 0.8
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBOURS = 10
SMILE_SCALE_FACTOR = 1.05
SMILE_MIN_NEIGHBOURS = 15

def load_cascades():
    smile_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_smile.xml")
    face_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_frontalface_default.xml")
    return smile_cascade, face_cascade

def initialize_video_capture(source=0):
    return cv2.VideoCapture(source)

def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def detect_faces(face_cascade, gray_frame):
    return face_cascade.detectMultiScale(gray_frame, scaleFactor=FACE_SCALE_FACTOR, minNeighbors=FACE_MIN_NEIGHBOURS)

def detect_smiles(smile_cascade, gray_frame):
    return smile_cascade.detectMultiScale(gray_frame, scaleFactor=SMILE_SCALE_FACTOR, minNeighbors=SMILE_MIN_NEIGHBOURS)

def draw_rectangles(frame, coordinates, color, thickness=3):
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

# FIXME
def calculate_fps(prev_time, current_time, frames):
    if prev_time == 0:
        return 0, current_time
    return frames / (current_time - prev_time), current_time

def display_text(frame, text, position, font, font_scale, font_color, thickness, line_type):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)


def is_smile_within_face(smiles, face_coordinates):
    face_x, face_y, face_w, face_h = face_coordinates
    for (x, y, w, h) in smiles:
        if x > face_x and y > face_y and (x + w) < (face_x + face_w) and (y + h) < (face_y + face_h):
            return True 
    return False 


def draw_smile_rectangle(smiles, face_coordinates, frame):
    face_x, face_y, face_w, face_h = face_coordinates
    for (x, y, w, h) in smiles:
        # Check if inside face border
        if x > face_x and y > face_y and (x + w) < (face_x + face_w) and (y + h) < (face_y + face_h):
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)


def count_smiles(smile_detected, smile_active, last_smile_time, smiles_detected):
    current_time = time.time()
    
    if smile_detected:
        if not smile_active and (current_time - last_smile_time) > TIME_TO_START_COUNTING:
            smiles_detected += 1
            smile_active = True
    else:
        smile_active = False
        last_smile_time = current_time

    return smile_active, last_smile_time, smiles_detected

# Main function to process video and detect smiles
def process_video():
    smile_cascade, face_cascade = load_cascades()
    video = initialize_video_capture()

    # Text configuration
    # FIXME: Remove that many variables from here
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)
    font_thickness = 3
    font_line_type = 2

    smiles_detected = 0
    smile_active = False
    last_smile_time = 0
    prev_frame_time = 0
    frame_counter = 0
    fps = 0
    frames = 0

    while True:
        check, frame = video.read()
        if not check:
            break

        gray_frame = convert_to_grayscale(frame)
        faces = detect_faces(face_cascade, gray_frame)

        for (face_x, face_y, face_w, face_h) in faces:
            draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))
            smiles = detect_smiles(smile_cascade, gray_frame)
            smile_detected = is_smile_within_face(smiles, (face_x, face_y, face_w, face_h))
            smile_active, last_smile_time, smiles_detected = count_smiles(smile_detected, smile_active, last_smile_time, smiles_detected)
            if smile_detected:
                draw_smile_rectangle(smiles, (face_x, face_y, face_w, face_h), frame)

            # FPS Calculation
            new_frame_time = time.time()
            fps, prev_frame_time = calculate_fps(prev_frame_time, new_frame_time, frames)
            frames += 1

            # Display info
            h, w, _ = frame.shape
            text_position = (round(w / 4), round(h / 8))
            text_to_show = f"Detected smiles: {smiles_detected} fps: {round(fps)}"
            display_text(frame, text_to_show, text_position, font, font_scale, font_color, font_thickness, font_line_type)

        cv2.imshow('Smile Detection', frame)

        # Exit on escape key
        if cv2.waitKey(30) & 0xFF == 27:
            break

    # Clean up
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video()
