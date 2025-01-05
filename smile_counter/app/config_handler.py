from configparser import ConfigParser
import os
from pathlib import Path

class ConfigHandler:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigHandler, cls).__new__(cls)
        return cls._instance
        
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.config = ConfigParser(comment_prefixes=';', allow_no_value=True)
            self.config.optionxform = str
            self.config_dir = os.path.join(str(Path.home()), '.smile_counter')
            self.config_path = os.path.join(self.config_dir, 'config.ini')
            
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.default_config_path = os.path.join(current_dir, 'src', 'utils', 'default_config.ini')
            
            self._ensure_config_exists()
            self.initialized = True
            
    def _load_default_config(self) -> ConfigParser:
        default_config = ConfigParser(comment_prefixes=';', allow_no_value=True)
        default_config.optionxform = str  # Preserve case sensitivity
            
        print(f"Loading default config from {self.default_config_path}")
        default_config.read(self.default_config_path)
        
        # Debug output
        print("Default config sections:", default_config.sections())
        for section in default_config.sections():
            print(f"Section {section} items:", dict(default_config[section]))
        
        return default_config
            
    def _ensure_config_exists(self):
        # Create directories if needed
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
            print(f"Created config directory: {self.config_dir}")

        default_config = self._load_default_config()
        if not default_config.sections():
            raise RuntimeError("Default config is empty or corrupted")

        if not os.path.exists(self.config_path):
            # Create new config file
            with open(self.config_path, 'w') as configfile:
                default_config.write(configfile)
            self.config = default_config
            return

        # Load existing config
        self.config = ConfigParser(comment_prefixes=';', allow_no_value=True)
        self.config.optionxform = str
        self.config.read(self.config_path)

        if not self.config.sections():
            # Replace empty config with default
            self.config = default_config
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)
            return

        # Update missing sections/options
        updated = False
        for section in default_config.sections():
            if not self.config.has_section(section):
                self.config.add_section(section)
                updated = True
            
            # Copy missing options from default
            for key, value in default_config[section].items():
                if not self.config.has_option(section, key):
                    self.config.set(section, key, str(value))
                    updated = True

        if updated:
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)

    def get_config(self):
        # Ensure config is loaded
        self.config.read(self.config_path)
        
        if not self.config.has_section('Settings'):
            print("Settings section missing, reinitializing config")
            self._ensure_config_exists()
            self.config.read(self.config_path)
        
        settings = dict(self.config['Settings'])
        font = dict(self.config['Font']) if self.config.has_section('Font') else {}
        
        if 'color' in font:
            font['color'] = eval(font['color'])
            
        return type('Config', (), {**settings, 'FONT': font})
        
    def update_config(self, updates: dict):
        self.config.read(self.config_path)
        for key, value in updates.items():
            self.config.set('Settings', key, str(value))
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)