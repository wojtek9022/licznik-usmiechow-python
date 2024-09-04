import cv2
import os

# Load the cascade
smile_cascade = cv2.CascadeClassifier("../haar_classifiers/haarcascade_smile.xml")
face_cascade = cv2.CascadeClassifier("../haar_classifiers/haarcascade_frontalface_default.xml")

# Photos path
pathname = os.path.join('G:\inzynierka', 'our_samples_19062024', 'Phone', '378x504-positive')

# Creating total path of the photo
photos = [os.path.join(pathname, photo) for photo in os.listdir(pathname)]

detected_smiles = 0

# To use a video file as input
# video = cv2.VideoCapture('filename.mp4')

for photo in photos:
    photo_name = os.path.splitext(os.path.basename(photo))[0]
    img = cv2.imread(photo)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect the faces
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
    # Draw the rectangle around each face
    for (x, y, w, h) in face:
        img = cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 3)
        # Detect the Smile
        smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=10)
        if len(smile) > 0:
            print("Wykryto w : ", photo_name)
            detected_smiles += 1
        else:
            print("Nie wykryto w : ", photo_name)
        # Draw the rectangle around detected smile
        for x, y, w, h in smile:
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    # Display output
    cv2.imshow('smile detect', img)
    filename = "./output/" + photo_name + ".jpg"
    cv2.imwrite(filename, img)

    # Stop if escape key is pressed
    key = cv2.waitKey(30) & 0xff
    if key == 27:
        break

print("Przeanalizowano ", len(photos), " zdjec")
print("Wykryto ",detected_smiles," usmiechow")
cv2.destroyAllWindows()