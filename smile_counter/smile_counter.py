import cv2
import time

# Constants for face and smile detection
FACE_SCALE_FACTOR = 1.1
FACE_MIN_NEIGHBOURS = 10
SMILE_SCALE_FACTOR = 1.05
SMILE_MIN_NEIGHBOURS = 30

# Time constants
TIME_TO_START_COUNTING = 0.5  # Time to detect smile continuously

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

def detect_smiles(smile_cascade, gray_face_region):
    return smile_cascade.detectMultiScale(gray_face_region, scaleFactor=SMILE_SCALE_FACTOR, minNeighbors=SMILE_MIN_NEIGHBOURS)

def draw_rectangles(frame, coordinates, color, thickness=3):
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

def calculate_fps(prev_time, current_time, frames):
    if prev_time == 0:
        return 0, current_time
    return frames / (current_time - prev_time), current_time

def display_text(frame, text, position, font, font_scale, font_color, thickness, line_type):
    cv2.putText(frame, text, position, font, font_scale, font_color, thickness, line_type)

# Function to handle smile detection and rectangle drawing
def handle_smile_and_draw(smile_detected, smile_active, last_smile_time, smiles_detected, frame, smiles, face_x, face_y):
    current_time = time.time()

    # Check if a smile has been detected continuously for TIME_TO_START_COUNTING
    if smile_detected:
        if not smile_active and (current_time - last_smile_time) > TIME_TO_START_COUNTING:
            smiles_detected += 1  # Increment smile count
            smile_active = True
    else:
        smile_active = False
        last_smile_time = current_time  # Reset smile detection timer

    # Draw blue rectangle around the largest smile immediately
    if len(smiles) > 0 and smile_active:
        # Find the largest smile (by area)
        largest_smile = max(smiles, key=lambda s: s[2] * s[3])
        sx, sy, sw, sh = largest_smile
        smile_x, smile_y = face_x + sx, face_y + sy
        draw_rectangles(frame, [(smile_x, smile_y, sw, sh)], (255, 0, 0))  # Blue rectangle

    return smile_active, last_smile_time, smiles_detected

# Main function to process video and detect smiles
def process_video():
    smile_cascade, face_cascade = load_cascades()
    video = initialize_video_capture()

    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (255, 0, 0)
    font_thickness = 3
    font_line_type = 2

    smiles_detected = 0
    smile_active = False
    last_smile_time = 0
    prev_frame_time = 0
    fps = 0
    frames = 0

    while True:
        check, frame = video.read()
        if not check:
            break

        gray_frame = convert_to_grayscale(frame)
        faces = detect_faces(face_cascade, gray_frame)

        for (face_x, face_y, face_w, face_h) in faces:
            # Draw red rectangle around face
            draw_rectangles(frame, [(face_x, face_y, face_w, face_h)], (0, 0, 255))

            # Crop the gray frame to the face region for smile detection
            face_region = gray_frame[face_y + face_h // 2:face_y + face_h, face_x:face_x + face_w]  # Lower half of the face
            smiles = detect_smiles(smile_cascade, face_region)
            
            # Adjust coordinates of smiles to match the full frame
            smiles = [(x, y + face_h // 2, w, h) for (x, y, w, h) in smiles]

            # Determine if a smile was detected in the face region
            smile_detected = len(smiles) > 0

            # Handle smile detection and rectangle drawing with timing
            smile_active, last_smile_time, smiles_detected = handle_smile_and_draw(
                smile_detected, smile_active, last_smile_time, smiles_detected, frame, smiles, face_x, face_y
            )

            # FPS calculation
            new_frame_time = time.time()
            fps, prev_frame_time = calculate_fps(prev_frame_time, new_frame_time, frames)
            frames += 1

            # Display smile count and FPS
            h, w, _ = frame.shape
            text_position = (round(w / 4), round(h / 8))
            text_to_show = f"Detected smiles: {smiles_detected} | FPS: {round(fps)}"
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
