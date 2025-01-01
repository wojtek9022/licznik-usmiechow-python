import os
import re
from typing import Dict, Any

class ConfigManager:
    def __init__(self) -> None:
        self.config_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')
        self.config_path = os.path.join(self.config_dir, 'config.py')
        os.makedirs(self.config_dir, exist_ok=True)

    def read_config(self) -> str:
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return f.read()
        return ""

    def write_config(self, config_content: str) -> None:
        with open(self.config_path, 'w') as f:
            f.write(config_content)

    def update_config(self, updates: Dict[str, Any]) -> None:
        config_content = self.read_config()
        
        for key, value in updates.items():
            if isinstance(value, str):
                value = f"'{value}'"
            config_content = re.sub(
                f"{key} = .*\n",
                f"{key} = {value}\n",
                config_content
            )
        
        self.write_config(config_content)

    def validate_config(self, config_content: str) -> None:
        required_keys = [
            "FACE_SCALE_FACTOR",
            "FACE_MIN_NEIGHBOURS",
            "SMILE_SCALE_FACTOR", 
            "SMILE_MIN_NEIGHBOURS",
            "TIME_TO_START_COUNTING",
            "FONT",
            "LANGUAGE"
        ]
        for key in required_keys:
            if key not in config_content:
                raise ValueError(f"Missing required config key: {key}")