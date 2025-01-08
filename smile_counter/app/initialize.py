"""
Application initialization module.

This module handles initial setup tasks like configuration loading
and validation of required resources before the main application starts.
"""

import os
from .config_handler import ConfigHandler
from typing import Any

def ensure_app_initialized() -> Any:
    """
    Initialize application configuration and resources.

    Handles first-time setup by creating necessary config files
    and directories. Loads and validates configuration settings
    before application startup.

    Returns:
        Any: Configuration object containing application settings
    """
    config_handler = ConfigHandler()
    return config_handler.get_config()