import cv2
import os
import logging
import time
from config_file_reader import read_config

# Read config file
config_data = read_config(file_path='config.ini', section='DetectingSmilesInPhotos')

# FIXME: This code has too many redundant comments
# Logger init
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Create handler to log to file
file_handler = logging.FileHandler(config_data['LOG_OUTPUT_PATH'])
file_handler.setLevel(logging.INFO)
# Create handler to log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)
# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

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


def save_output_images(photo_name: str, img: cv2.Mat):
    if config_data['SAVE_OUTPUT_IMAGES']:
        filename = _create_output_path(config_data['DEFAULT_OUTPUT_FOLDER_PATH'], photo_name, config_data['DEFAULT_OUTPUT_FILE_EXTENSION'])
        cv2.imwrite(filename, img)


def show_output_images(window_name: str, img: cv2.Mat):
    if config_data['SHOW_OUTPUT_IMAGES']:
        cv2.imshow(window_name, img)


def main():
    # Load the cascade
    smile_cascade = cv2.CascadeClassifier(config_data['SMILE_CLASSIFIER_PATH'])
    face_cascade = cv2.CascadeClassifier(config_data['FRONTAL_FACE_CLASSIFIER_PATH'])

    # Creating total path of the photo
    photos = [os.path.join(config_data['DATASET_PHOTOS_INPUT_PATH'], photo) for photo in os.listdir(config_data['DATASET_PHOTOS_INPUT_PATH'])]

    detected_smiles = 0

    timer_start_time = time.time()

    for photo in photos:
        photo_name = os.path.splitext(os.path.basename(photo))[0]
        img = cv2.imread(photo)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the faces
        face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        # Draw the rectangle around each face
        for (x, y, w, h) in face:
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Detect the Smile
            smile = smile_cascade.detectMultiScale(gray, scaleFactor=1.4, minNeighbors=10)
            if len(smile) > 0:
                logger.info("Smile detected in photo with filename: " + photo_name)
                detected_smiles += 1
            else:
                logger.info("Do not detected smile in photo with filename: " + photo_name)
            # Draw the rectangle around detected smile
            for x_, y_, w_, h_ in smile:
                img = cv2.rectangle(img, (x_, y_), (x_ + w_, y_ + h_), (255, 0, 0), 3)

        # Display output
        show_output_images(window_name = config_data['DEFAULT_IMAGE_WINDOW_NAME'], img = img)

        save_output_images(photo_name=photo_name, img=img)

        # Stop if escape key is pressed
        key = cv2.waitKey(30) & 0xff
        if key == 27:
            break

    timer_stop_time = time.time()
    total_detecting_time = timer_stop_time - timer_start_time

    # NOTE: Logger should work like print if appropriate log level is set.
    logger.info("Analyzed " + str(len(photos)) + " photos")
    logger.info("Detected " + str(detected_smiles) + " smiles")
    logger.info("Detecting smiles took: " + str(total_detecting_time))
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
