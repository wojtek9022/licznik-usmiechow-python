import unittest
import os
import cv2
import logging
from smile_counter.app.src.detectors.smile_detector import SmileDetector
from smile_counter.app.config_handler import ConfigHandler

class SmileDetectorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename='smile_detector_smoke_test.log'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Initialize detector
        cls.config = ConfigHandler()
        cls.detector = SmileDetector(cls.config)
        
        # Define test data paths
        cls.positive_dir = os.path.join('tests', 'data', 'positive', 'color')
        cls.negative_dir = os.path.join('tests', 'data', 'negative', 'color')

    def _get_all_images(self, root_dir):
        """Helper method to recursively get all image files from directory"""
        image_files = []
        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        return image_files

    def test_positive_samples(self):
        """Test detection on positive samples (should detect smiles)"""
        success_count = 0
        total_count = 0
        
        for img_path in self._get_all_images(self.positive_dir):
            total_count += 1
            
            # Load and process image
            frame = cv2.imread(img_path)
            if frame is None:
                self.logger.error(f"Could not read image: {img_path}")
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            smile_detected = False
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.detector.detect(frame=roi_gray)
                if len(smiles) > 0:
                    smile_detected = True
                    break
            
            if smile_detected:
                success_count += 1
                self.logger.info(f"✓ Smile correctly detected in: {os.path.basename(img_path)}")
            else:
                self.logger.warning(f"✗ Failed to detect smile in: {os.path.basename(img_path)}")

        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        self.logger.info(f"Positive samples detection rate: {success_rate:.2f}% ({success_count}/{total_count})")
        self.assertGreater(success_rate, 70.0, "Detection rate for positive samples is too low")

    def test_negative_samples(self):
        """Test detection on negative samples (should not detect smiles)"""
        success_count = 0
        total_count = 0
        
        for img_path in self._get_all_images(self.negative_dir):
            total_count += 1
            
            # Load and process image
            frame = cv2.imread(img_path)
            if frame is None:
                self.logger.error(f"Could not read image: {img_path}")
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            smile_detected = False
            for (x, y, w, h) in faces:
                roi_gray = gray[y+y+h, x+x+w]
                smiles = self.detector.detect(frame=roi_gray)
                if len(smiles) > 0:
                    smile_detected = True
                    break
            
            if not smile_detected:
                success_count += 1
                self.logger.info(f"✓ Correctly did not detect smile in negative sample: {os.path.basename(img_path)}")
            else:
                self.logger.warning(f"✗ Incorrectly detected smile in negative sample: {os.path.basename(img_path)}")

        success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
        self.logger.info(f"Negative samples correct rejection rate: {success_rate:.2f}% ({success_count}/{total_count})")
        
        # Assert reasonable rejection rate for negative samples
        self.assertGreater(success_rate, 70.0, "Rejection rate for negative samples is too low")

if __name__ == '__main__':
    unittest.main(verbosity=2)