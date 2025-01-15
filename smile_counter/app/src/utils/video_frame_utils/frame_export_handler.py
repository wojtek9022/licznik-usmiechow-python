import os
import cv2
from datetime import datetime
from app.handlers.config_handler import ConfigHandler

class FrameExportHandler:
    """Handles exporting frames when smiles are detected."""
    
    def __init__(self):
        self.config = ConfigHandler().get_config()
        self.export_dir = self._setup_export_directory()

    def _setup_export_directory(self) -> str:
        """Create export directory if it doesn't exist."""
        config_dir = os.path.dirname(ConfigHandler().config_path)
        export_dir = os.path.join(config_dir, 'detected_smiles_images')
        
        if not os.path.exists(export_dir):
            os.makedirs(export_dir)
            print(f"Created smile frames export directory: {export_dir}")
            
        return export_dir

    def export_frame(self, frame) -> None:
        """Export a frame when smile is counted."""
        if not self.config.EXPORT_SMILE_FRAMES:
            return
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'smile_{timestamp}.jpg'
        filepath = os.path.join(self.export_dir, filename)
        
        try:
            cv2.imwrite(filepath, frame)
            print(f"Exported smile frame: {filepath}")
        except Exception as e:
            print(f"Error exporting smile frame: {e}")