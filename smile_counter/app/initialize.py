import os
from app.config_handler import ConfigHandler

def ensure_app_initialized():
    config_handler = ConfigHandler()
    return config_handler.get_config()