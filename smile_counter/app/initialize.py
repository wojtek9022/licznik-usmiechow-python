import os
from app.config_manager import ConfigManager

def ensure_app_initialized():
    config_manager = ConfigManager()
    return config_manager.get_config()