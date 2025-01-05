from typing import Dict, TypedDict

class ConfigOption(TypedDict):
    type: type
    row: int

class OptionsConfig(TypedDict):
    FACE_SCALE_FACTOR: ConfigOption
    FACE_MIN_NEIGHBOURS: ConfigOption
    SMILE_SCALE_FACTOR: ConfigOption
    SMILE_MIN_NEIGHBOURS: ConfigOption
    TIME_TO_START_COUNTING: ConfigOption