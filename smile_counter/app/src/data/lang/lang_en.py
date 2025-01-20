# Main menu
TITLE_TEXT = "Smile Counter"
VERSION_TEXT = "Version: {version}"
START_BUTTON_TEXT = "Start"
OPTIONS_BUTTON_TEXT = "Options"
EXIT_BUTTON_TEXT = "Exit"
LOGO_NOT_FOUND_TEXT = "[Logo not found]"

# Loading messages
OPTIONS_LOADING_TEXT = "Loading options, please wait..."
VIDEO_LOADING_TEXT = "Starting camera, please wait..."

# Statistics window
STATISTICS_BUTTON_TEXT = "Statistics"
STATISTICS_TITLE_TEXT = "Smile Statistics"
TOTAL_SMILES_TEXT = "Total Smiles Detected:"
TODAY_SMILES_TEXT = "Smiles Today:"
WEEK_SMILES_TEXT = "Smiles This Week:"
MONTH_SMILES_TEXT = "Smiles This Month:"
YEAR_SMILES_TEXT = "Smiles This Year:"

# Options window
OPTIONS_TITLE_TEXT = "Options"
FACE_SCALE_FACTOR_TEXT = "Face Scale Factor:"
FACE_MIN_NEIGHBOURS_TEXT = "Face Min Neighbours:"
SMILE_SCALE_FACTOR_TEXT = "Smile Scale Factor:"
SMILE_MIN_NEIGHBOURS_TEXT = "Smile Min Neighbours:"
TIME_TO_START_COUNTING_TEXT = "Time to Start Counting (seconds):"
COUNTED_SMILE_COOLDOWN_TIME_TEXT = "Smile Cooldown Time (seconds):"
SAVE_BUTTON_TEXT = "Save"
OPTIONS_SAVED_TEXT = "Options saved successfully!"
ERROR_MESSAGE_TEXT = "Error saving options: {error}"
LANGUAGE_TEXT = "Language:"
DEBUG_MODE_TEXT = "Debug mode:"
CAMERA_SOURCE_TEXT = "Camera Source:"
APPLY_FACE_EFFECTS_TEXT = "Apply Face Effects"
AUTO_CONFIG_ADJUSTING_TEXT = "Auto-adjust Detection Settings (experimental)"
EXPORT_SMILE_FRAMES_TEXT = "Export Detected Smile"
SMILE_FRAMES_PATH_TEXT = "Detected Smile Frames Directory"
SELECT_DIRECTORY_TEXT = "Select Directory"

# Calibration texts
CALIBRATION_PROMPT_TEXT = "Press 'C' to begin parameters calibration"
CALIBRATION_SMILE_TEXT = "Calibration in progress\nPlease smile continuously 😊 while standing still."
CALIBRATION_NO_SMILE_TEXT = "Calibration in progress\nPlease do NOT smile now ❌"

# Tooltips in options window
FACE_SCALE_FACTOR_TOOLTIP = "Scaling factor for face detection.\nLarger values detect smaller faces, \
but increase false positives.\nHas moderate impact on overall detection efficiency."

FACE_MIN_NEIGHBOURS_TOOLTIP = "Minimum number of neighboring detections required for face detection.\n\
Higher values reduce false positives, but make face recognition more difficult."

SMILE_SCALE_FACTOR_TOOLTIP = "Scaling factor for smile detection.\nLarger values detect smaller smiles, \
but increase false positives.\nHas significant impact on overall detection efficiency."

SMILE_MIN_NEIGHBOURS_TOOLTIP = "Minimum number of neighboring points required for smile detection.\n\
Higher values reduce false positives, but significantly hinder smile detection."

TIME_TO_START_COUNTING_TOOLTIP = "Time in seconds that a smile must be held before being counted.\n\
Higher values eliminate sporadic false detections, but require user to maintain smile longer."

COUNTED_SMILE_COOLDOWN_TIME_TOOLTIP = "Time in seconds before another smile can be counted from the same person.\n\
Prevents counting multiple smiles in a short period of time."

CAMERA_SOURCE_TOOLTIP = "Select which camera to use for detection.\nDefault camera should be the first option.\n\
Other cameras, if present, should be available in the dropdown list."

DEBUG_MODE_TOOLTIP = "Show detection rectangles and additional debug information."

APPLY_FACE_EFFECTS_TOOLTIP = "Enable or disable funny face effects like beards and mustaches."

AUTO_CONFIG_ADJUSTING_TOOLTIP = "Automatically adjust detection parameters based on many factors."

EXPORT_SMILE_FRAMES_TOOLTIP = "Save frames as images when smiles are detected."

SMILE_FRAMES_PATH_TOOLTIP = "Directory where detected smile frames will be saved."


# Main program
DETECTED_SMILES_TEXT = "Detected smiles: {count}"
SMILE_COUNTED_TEXT = "SMILE COUNTED 😊!"