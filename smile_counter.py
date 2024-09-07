import cv2
import time

# Constants
TIME_TO_START_COUNTING = 2

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_frontalface_default.xml")

# To capture video from webcam.
video = cv2.VideoCapture(0)

# To use a video file as input
# video = cv2.VideoCapture('filename.mp4')

# Text configuration
font = cv2.FONT_HERSHEY_SIMPLEX
text_position = (0,0)
font_scale = 1
font_color = (0,0,0)
font_thickness = 3
font_line_type = 2

smiles_detected = 0
smile_active = True
while True:
    # Read the frame
    check, frame = video.read()
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    # Draw the rectangle around each face
    for (x, y, w, h) in face:
        img = cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 0, 255), 3)
        # Detect the Smile
        smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=20)
        # Increase smile counter by number of detected smiles
        if len(smile) > 0:
            if not smile_active:
                current_time = time.time()
                if current_time - last_smile_time > TIME_TO_START_COUNTING:
                    smiles_detected += 1
                    smile_active = True
        else:
            if smile_active:
                last_smile_time = time.time()
                smile_active = False

        # Draw the rectangle around detected smile
        for x, y, w, h in smile:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Get img size
    h, w, c = frame.shape
    # Set new img position depending on img size
    text_position = (round(w/4), round(h/8))
    # Text to show
    text_to_show = "Detected smiles: " + str(smiles_detected)
    # Add smile count
    cv2.putText(img, text_to_show,
                text_position,
                font,
                font_scale,
                font_color,
                font_thickness,
                font_line_type)

    # Display output
    cv2.imshow('smile detect', frame)

    # Stop if escape key is pressed
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# Release the VideoCapture object
video.release()
cv2.destroyAllWindows()