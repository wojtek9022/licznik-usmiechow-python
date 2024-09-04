import cv2
import time

# Load the cascade
smile_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("haar_classifiers/haarcascade_frontalface_default.xml")

# To capture video from webcam.
video = cv2.VideoCapture(0)

# To use a video file as input
# video = cv2.VideoCapture('filename.mp4')

# Text configuration
font = cv2.FONT_HERSHEY_SIMPLEX
textPosition = (0,0)
fontScale = 1
fontColor = (0,0,0)
thickness = 3
lineType = 2

smiles_detected = 0
doCount = True
timerStart = 0
timerEnd = 0
TIME_TO_START_COUNTING = 1

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
        smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=10)
        # Increase smile counter by number of detected smiles
        if len(smile) > 0 and doCount == True:
            doCount = False
            timerStart = time.time()
            smiles_detected += len(smile)

        # Stop timer and let smile count
        timerEnd = time.time()
        total = timerStart - timerEnd
        if total < -TIME_TO_START_COUNTING:
            doCount = True

        # Draw the rectangle around detected smile
        for x, y, w, h in smile:
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 3)

        # Get img size
        h, w, c = frame.shape
        # Set new img position depending on img size
        textPosition = (round(w/4), round(h/8))
        # Text to show
        textToShow = "Wykryto usmiechow: " + str(smiles_detected)
        # Add smile count
        cv2.putText(img, textToShow,
                    textPosition,
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)

    # Display output
    cv2.imshow('smile detect', frame)

    # Stop if escape key is pressed
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

# Release the VideoCapture object
video.release()
cv2.destroyAllWindows()