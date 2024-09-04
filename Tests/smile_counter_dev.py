import cv2
import os
import logging

# Logger init
logger = logging.getLogger(__name__)
def main():
    # Logger config
    logging.basicConfig(filename='./output/detection.log', level=logging.INFO)

    # Load the cascade
    smile_cascade = cv2.CascadeClassifier("../haar_classifiers/haarcascade_smile.xml")
    face_cascade = cv2.CascadeClassifier("../haar_classifiers/haarcascade_frontalface_default.xml")

    # Photos path
    pathname = "./input/378x504-positive"

    # Creating total path of the photo
    photos = [os.path.join(pathname, photo) for photo in os.listdir(pathname)]

    detected_smiles = 0

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
                logger.info("Smile detected in photo with filename: " + photo_name)
                detected_smiles += 1
            else:
                logger.info("Do not detected smile in photo with filename: " + photo_name)
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

    print("Analyzed ", len(photos), " photos")
    logger.info("Analyzed " + str(len(photos)) + " photos")
    print("Detected ", detected_smiles," smiles")
    logger.info("Detected " + str(detected_smiles) + " smiles")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()