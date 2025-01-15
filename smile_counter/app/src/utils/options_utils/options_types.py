from typing import Dict, TypedDict

class ConfigOption(TypedDict):
    """
    Type definition for individual configuration option.

    Defines the structure of a single configuration option including
    its type and position in the UI layout.

    Attributes:
        type (type): Data type of the configuration value (int/float)
        row (int): Row position in the options window grid layout
    """
    type: type
    row: int

class OptionsConfig(TypedDict):
    """
    Type definition for complete configuration schema.

    Defines the structure of all available configuration options
    for the smile detection system.

    Attributes:
        FACE_SCALE_FACTOR (ConfigOption): Face detection scaling parameter
        FACE_MIN_NEIGHBOURS (ConfigOption): Face detection neighbor threshold
        SMILE_SCALE_FACTOR (ConfigOption): Smile detection scaling parameter
        SMILE_MIN_NEIGHBOURS (ConfigOption): Smile detection neighbor threshold
        TIME_TO_START_COUNTING (ConfigOption): Delay before smile counting
    """
    FACE_SCALE_FACTOR: ConfigOption
    FACE_MIN_NEIGHBOURS: ConfigOption
    SMILE_SCALE_FACTOR: ConfigOption
    SMILE_MIN_NEIGHBOURS: ConfigOption
    TIME_TO_START_COUNTING: ConfigOption
    EXPORT_SMILE_FRAMES: ConfigOption