import cv2
import os
import logging

# FIXME: Create a config file
DATASET_PHOTOS_INPUT_PATH = "./input/378x504-positive"
DEFAULT_OUTPUT_FOLDER_PATH = "./output/"
DEFAULT_OUTPUT_FILE_EXTENSION = "jpg"
LOG_OUTPUTH_PATH = DEFAULT_OUTPUT_FOLDER_PATH + "detection.log"
SMILE_CLASSIFIER_PATH = "../haar_classifiers/haarcascade_smile.xml"
FRONTAL_FACE_CLASSIFIER_PATH = "../haar_classifiers/haarcascade_frontalface_default.xml"

# FIXME: This code has too many redundant comments
# Logger init
logger = logging.getLogger(__name__)


def _create_output_path(base_dir: str, filename: str, extension: str) -> str:
    '''
    Creates proper output path for photos with detected smiles.

    :param: base_dir [M] - Base output directory path
    :param: filename [M] - Chosen name of the file.
    :param: extension [M] - Extension that will be used to save graphic file.
    :return: output_path - Created output path for the file.
    '''
    output_path = os.path.join(base_dir, f"{filename}.{extension}")
    return output_path


def main():
    # Logger config
    logging.basicConfig(filename=LOG_OUTPUTH_PATH, level=logging.INFO)

    # Load the cascade
    smile_cascade = cv2.CascadeClassifier(SMILE_CLASSIFIER_PATH)
    face_cascade = cv2.CascadeClassifier(FRONTAL_FACE_CLASSIFIER_PATH)

    # Creating total path of the photo
    photos = [os.path.join(DATASET_PHOTOS_INPUT_PATH, photo) for photo in os.listdir(DATASET_PHOTOS_INPUT_PATH)]

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
        filename = _create_output_path(DEFAULT_OUTPUT_FOLDER_PATH, photo_name, DEFAULT_OUTPUT_FILE_EXTENSION)
        cv2.imwrite(filename, img)

        # Stop if escape key is pressed
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    # NOTE: Logger should work like print if appropriate log level is set.
    logger.info("Analyzed " + str(len(photos)) + " photos")
    logger.info("Detected " + str(detected_smiles) + " smiles")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
