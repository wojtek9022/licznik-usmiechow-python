# Main menu
TITLE_TEXT = "Smile Counter"
VERSION_TEXT = "Version: 2.3.3"
START_BUTTON_TEXT = "Start"
OPTIONS_BUTTON_TEXT = "Options"
EXIT_BUTTON_TEXT = "Exit"
LOGO_NOT_FOUND_TEXT = "[Logo not found]"

# Options window
OPTIONS_TITLE_TEXT = "Options"
FACE_SCALE_FACTOR_TEXT = "Face Scale Factor:"
FACE_MIN_NEIGHBOURS_TEXT = "Face Min Neighbours:"
SMILE_SCALE_FACTOR_TEXT = "Smile Scale Factor:"
SMILE_MIN_NEIGHBOURS_TEXT = "Smile Min Neighbours:"
TIME_TO_START_COUNTING_TEXT = "Time to Start Counting:"
SAVE_BUTTON_TEXT = "Save"
OPTIONS_SAVED_TEXT = "Options saved successfully!"
ERROR_MESSAGE_TEXT = "Error saving options: {error}"
LANGUAGE_TEXT = "Language:"
DEBUG_MODE_TEXT = "Debug mode:"
COUNTED_SMILE_COOLDOWN_TIME_TEXT = "Smile Cooldown Time (seconds):"
CAMERA_SOURCE_TEXT = "Camera Source:"

# Tooltips in options window
FACE_SCALE_FACTOR_TOOLTIP = "Scaling factor for face detection. Larger values detect smaller faces but increase false positives"
FACE_MIN_NEIGHBOURS_TOOLTIP = "Minimum number of neighboring detections required for face detection. Higher values reduce false positives"
SMILE_SCALE_FACTOR_TOOLTIP = "Scaling factor for smile detection. Larger values detect smaller smiles but increase false positives"
SMILE_MIN_NEIGHBOURS_TOOLTIP = "Minimum number of neighboring detections required for smile detection. Higher values reduce false positives"
TIME_TO_START_COUNTING_TOOLTIP = "Time in seconds that a smile must be held before being counted"
COUNTED_SMILE_COOLDOWN_TIME_TOOLTIP = "Time in seconds before another smile can be counted from the same person"
CAMERA_SOURCE_TOOLTIP = "Select which camera to use for detection"
DEBUG_MODE_TOOLTIP = "Show detection rectangles and additional debug information"

# Main program
DETECTED_SMILES_TEXT = "Detected smiles: {count}"
SMILE_COUNTED_TEXT = "SMILE COUNTED 😊!"