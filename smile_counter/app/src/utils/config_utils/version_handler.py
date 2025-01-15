import os
from pathlib import Path

class VersionHandler:
    @staticmethod
    def get_version() -> str:
        try:
            # Go up three levels from utils/config_utils to src, then to data
            version_path = Path(__file__).parent.parent.parent / "data" / "version.txt"
            print(f"Attempting to read version from: {version_path}")  # Debug line
            with open(version_path, "r") as f:
                return f.read().strip()
        except Exception as e:
            print(f"Error reading version from {version_path}: {e}")
            return "0.0.0"