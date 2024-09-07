import configparser


def create_config():
    config = configparser.ConfigParser()

    # Create configuration for smile_counter_dev test
    config['DetectingSmilesInPhotos'] = {
        'DATASET_PHOTOS_INPUT_PATH' : "./input/378x504-positive",
        'DEFAULT_OUTPUT_FOLDER_PATH' : "./output/",
        'DEFAULT_OUTPUT_FILE_EXTENSION' : "jpg",
        'SMILE_CLASSIFIER_PATH' : "../haar_classifiers/haarcascade_smile.xml",
        'FRONTAL_FACE_CLASSIFIER_PATH' : "../haar_classifiers/haarcascade_frontalface_default.xml",
        'DEFAULT_IMAGE_WINDOW_NAME' : "Smile detection",
        'SAVE_OUTPUT_IMAGES' : False,
        'SHOW_OUTPUT_IMAGES' : False
    }

    # Add contant which have referece to another constant
    config['DetectingSmilesInPhotos']['LOG_OUTPUT_PATH'] = config['DetectingSmilesInPhotos']['DEFAULT_OUTPUT_FOLDER_PATH'] + "detection.log"

    with open('./config.ini', 'w') as configfile:
        config.write(configfile)


if __name__ == "__main__":
    create_config()