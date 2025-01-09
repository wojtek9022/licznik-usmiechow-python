from configparser import ConfigParser
import os
from pathlib import Path
from typing import Dict, Any

class ConfigHandler:
    """
    Singleton class for managing application configuration.

    Handles reading and writing configuration values, maintaining default settings,
    and ensuring configuration file integrity. Uses INI file format with sections
    for different configuration categories.

    Attributes:
        config (ConfigParser): Parser for configuration file
        config_dir (str): Directory path for user configuration
        config_path (str): Path to user's configuration file
        default_config_path (str): Path to default configuration template
    """

    _instance = None
    
    # FIXME: This is required when adding new options to the config
    # should depend on options themself and passed, not in the config_handler
    OPTION_TYPES = {
        'FACE_SCALE_FACTOR': float,
        'FACE_MIN_NEIGHBOURS': int,
        'SMILE_SCALE_FACTOR': float,
        'SMILE_MIN_NEIGHBOURS': int,
        'TIME_TO_START_COUNTING': float,
        'COUNTED_SMILE_COOLDOWN_TIME': float,
        'DEBUG_MODE': bool,
        'APPLY_FACE_EFFECTS': bool,
        'AUTO_CONFIG_ADJUSTING': bool
    }

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigHandler, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = ConfigParser(
                comment_prefixes=('#',';'),
                allow_no_value=True,
                inline_comment_prefixes=(';',)
            )
            self.config.optionxform = str
            self.config_dir = os.path.join(str(Path.home()), '.smile_counter')
            self.config_path = os.path.join(self.config_dir, 'config.ini')
            current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets handlers directory
            app_dir = os.path.dirname(current_dir)  # Gets app directory
            self.default_config_path = os.path.join(app_dir, 'src', 'utils', 'config_utils', 'default_config.ini')
            self._ensure_config_exists()
            self.initialized = True
            
    def _load_default_config(self) -> ConfigParser:
        default_config = ConfigParser(comment_prefixes=';', allow_no_value=True)
        default_config.optionxform = str  # Preserve case sensitivity
        print(f"Loading default config from {self.default_config_path}")
        default_config.read(self.default_config_path)
        return default_config
            
    def _ensure_config_exists(self) -> None:
        self._create_config_directory()
        default_config = self._load_default_config()
        if not default_config.sections():
            raise RuntimeError("Default config is empty or corrupted")
        if not os.path.exists(self.config_path):
            self._create_initial_config(default_config)
            return
        self._update_existing_config(default_config)

    def _create_config_directory(self) -> None:
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            print(f"Created config directory: {self.config_dir}")

    def _create_initial_config(self, default_config: ConfigParser) -> None:
        print(f"Creating new config at {self.config_path}")
        with open(self.config_path, 'w') as configfile:
            default_config.write(configfile)
        self.config = default_config

    def _update_existing_config(self, default_config: ConfigParser) -> None:
        self.config.read(self.config_path)
        updated = False
        updated = self._update_missing_sections(default_config)
        updated = self._update_missing_options(default_config) or updated
        if updated:
            self._save_config()

    def _update_missing_sections(self, default_config: ConfigParser) -> bool:
        updated = False
        for section in default_config.sections():
            if not self.config.has_section(section):
                print(f"Adding section: {section}")
                self.config.add_section(section)
                updated = True
        return updated

    def _update_missing_options(self, default_config: ConfigParser) -> bool:
        updated = False
        for section in default_config.sections():
            for key, value in default_config[section].items():
                if not self.config.has_option(section, key):
                    print(f"Adding option: {section}/{key} = {value}")
                    self.config.set(section, key, str(value))
                    updated = True
        return updated

    def _save_config(self) -> None:
        print("Saving updated config")
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def _parse_bool_value(self, value: str) -> bool:
        """Parse string to boolean value."""
        return str(value).lower().strip() in ('true', '1', 'yes', 'on')

    def get_config(self) -> Any:
        """
        Retrieve current configuration values.

        Returns:
            Any: Object-like structure containing configuration values
        """
        self.config.read(self.config_path)
        
        if not self.config.has_section('Settings'):
            print("Settings section missing, reinitializing config")
            self._ensure_config_exists()
            self.config.read(self.config_path)
            
        settings = {}
        for k, v in self.config['Settings'].items():
            try:
                # Handle None values and missing semicolons
                if v is None:
                    settings[k] = ""
                elif self.OPTION_TYPES.get(k) == bool:
                    settings[k] = self._parse_bool_value(v)
                elif ';' in v:
                    settings[k] = v.split(';')[0].strip()
                else:
                    settings[k] = v.strip()
            except Exception as e:
                print(f"Error parsing config value for {k}: {e}")
                settings[k] = ""

        font = {}
        if self.config.has_section('Font'):
            for k, v in self.config['Font'].items():
                if k == 'color' and v:
                    try:
                        font[k] = eval(v)
                    except:
                        font[k] = (255, 0, 0)  # Default color
                else:
                    font[k] = v

        return type('Config', (), {**settings, 'FONT': font})
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.

        Writes changes to configuration file and ensures persistence
        of updated values.

        Args:
            updates (Dict[str, Any]): Dictionary of configuration updates
                                    where keys match config option names
        """
        self.config.read(self.config_path)
        for key, value in updates.items():
            self.config.set('Settings', key, str(value))
        self._save_config()