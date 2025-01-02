import os
import re
import shutil
import importlib.util
from typing import Dict, Any

class ConfigManager:
    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ConfigManager._initialized:
            self.app_dir = os.path.dirname(os.path.abspath(__file__))
            self.src_dir = os.path.join(self.app_dir, 'src')
            self.config_path = os.path.join(self.src_dir, 'config.py')
            self.default_config_path = os.path.join(self.src_dir, 'utils', 'default_config.py')
            
            os.makedirs(os.path.dirname(self.default_config_path), exist_ok=True)
            os.makedirs(self.src_dir, exist_ok=True)
            
            self._ensure_config_exists()
            ConfigManager._initialized = True

    def _ensure_config_exists(self) -> None:
        if not os.path.exists(self.config_path):
            shutil.copy2(self.default_config_path, self.config_path)

    def get_config(self):
        self._ensure_config_exists()
        spec = importlib.util.spec_from_file_location("config", self.config_path)
        config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config)
        return config

    def read_config(self) -> str:
        with open(self.config_path, 'r') as f:
            return f.read()

    def write_config(self, config_content: str) -> None:
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update config file with new values"""
        try:
            config_content = self.read_config()
            
            for key, value in updates.items():
                if isinstance(value, str):
                    formatted_value = f"'{value}'"
                else:
                    formatted_value = str(value)
                
                # Use lookahead/lookbehind for exact match
                pattern = f"(?<={key} = )[^\n]*"
                
                # Find all occurrences
                matches = list(re.finditer(pattern, config_content))
                if not matches:
                    print(f"Key {key} not found in config")
                    continue
                    
                # Update the last occurrence
                last_match = matches[-1]
                start, end = last_match.span()
                config_content = (
                    config_content[:start] + 
                    formatted_value +
                    config_content[end:]
                )
                
            self.write_config(config_content)
            print(f"Config updated successfully")
            
        except Exception as e:
            print(f"Error updating config: {str(e)}")
            raise
        
        self.write_config(config_content)