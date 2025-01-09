import unittest
import os
import cv2
import logging
from pathlib import Path
from smile_counter.app.src.detectors.smile_detector import SmileDetector
from smile_counter.initialize import get_initialized_config

class SmileDetectorSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create output directory
        output_dir = Path(__file__).parent / 'output'
        output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=output_dir / 'smile_detector_smoke_test.log',
            filemode='w'
        )
        cls.logger = logging.getLogger(__name__)
        
        # Initialize config and detector
        cls.config = get_initialized_config()
        cls.detector = SmileDetector(cls.config)
        
        # Define test paths
        cls.test_data_dir = Path(__file__).parent / 'data'
        cls.positive_dir = cls.test_data_dir / 'positive'
        cls.negative_dir = cls.test_data_dir / 'negative'
        
        # Initialize statistics
        cls.subfolder_stats = {}
        cls.confusion_matrix = {
            'true_positive': 0,
            'false_positive': 0,
            'true_negative': 0,
            'false_negative': 0
        }

    @classmethod
    def tearDownClass(cls):
        """Print summary statistics after all tests"""
        cls.logger.info("\n" + "="*50)
        cls.logger.info("SMILE DETECTION TEST SUMMARY")
        cls.logger.info("="*50)
        
        # Print subfolder statistics
        cls.logger.info("\nSubfolder Statistics:")
        cls.logger.info("-"*30)
        for subfolder, stats in cls.subfolder_stats.items():
            detected = stats.get('detected', stats.get('not_detected', 0))
            cls.logger.info(f"\nFolder: {subfolder}")
            cls.logger.info(f"Success Rate: {stats['success_rate']:.2f}%")
            cls.logger.info(f"Detected: {detected}/{stats['total']}")
        
        # Print confusion matrix
        total = sum(cls.confusion_matrix.values())
        if total > 0:
            cls.logger.info("\nConfusion Matrix:")
            cls.logger.info("-"*30)
            cls.logger.info(f"True Positives (smile correctly detected): {cls.confusion_matrix['true_positive']}")
            cls.logger.info(f"False Positives (smile incorrectly detected): {cls.confusion_matrix['false_positive']}")
            cls.logger.info(f"True Negatives (no-smile correctly rejected): {cls.confusion_matrix['true_negative']}")
            cls.logger.info(f"False Negatives (smile missed): {cls.confusion_matrix['false_negative']}")
            
            # Calculate metrics
            accuracy = ((cls.confusion_matrix['true_positive'] + cls.confusion_matrix['true_negative']) / total) * 100
            precision = (cls.confusion_matrix['true_positive'] / 
                        (cls.confusion_matrix['true_positive'] + cls.confusion_matrix['false_positive'])) * 100
            
            cls.logger.info("\nOverall Metrics:")
            cls.logger.info("-"*30)
            cls.logger.info(f"Total images processed: {total}")
            cls.logger.info(f"Overall accuracy: {accuracy:.2f}%")
            cls.logger.info(f"Precision: {precision:.2f}%")
        
        # Add configuration information
        cls.logger.info("\nTest Configuration:")
        cls.logger.info("-"*30)
        cls.logger.info(f"Face Scale Factor: {cls.config.FACE_SCALE_FACTOR}")
        cls.logger.info(f"Face Min Neighbours: {cls.config.FACE_MIN_NEIGHBOURS}")
        cls.logger.info(f"Smile Scale Factor: {cls.config.SMILE_SCALE_FACTOR}")
        cls.logger.info(f"Smile Min Neighbours: {cls.config.SMILE_MIN_NEIGHBOURS}")
        
        # Add any other relevant configuration parameters
        cls.logger.info("\n" + "="*50)

    def _process_image(self, img_path: Path) -> bool:
        """Process single image and return if smile was detected"""
        frame = cv2.imread(str(img_path))
        if frame is None:
            self.logger.error(f"Could not read image: {img_path}")
            return False
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Use config values for face detection
        faces = self.detector.face_cascade.detectMultiScale(
            gray,
            scaleFactor=float(self.config.FACE_SCALE_FACTOR),
            minNeighbors=int(self.config.FACE_MIN_NEIGHBOURS)
        )
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            # Use config values for smile detection
            smiles = self.detector.detect(
                frame=roi_gray,
                scaleFactor=float(self.config.SMILE_SCALE_FACTOR),
                minNeighbors=int(self.config.SMILE_MIN_NEIGHBOURS)
            )
            if len(smiles) > 0:
                return True
        return False

    def _process_subfolder(self, folder_path: Path, expected_smile: bool) -> dict:
        """Process images in a subfolder and collect statistics"""
        stats = {'total': 0, 'detected': 0}
        
        for img_path in folder_path.rglob('*.[jJ][pP][gG]'):
            stats['total'] += 1
            detected = self._process_image(img_path)
            if detected:
                stats['detected'] += 1
            
            # Update confusion matrix
            if expected_smile and detected:
                self.confusion_matrix['true_positive'] += 1
            elif expected_smile and not detected:
                self.confusion_matrix['false_negative'] += 1
            elif not expected_smile and detected:
                self.confusion_matrix['false_positive'] += 1
            else:
                self.confusion_matrix['true_negative'] += 1
                
        return stats

    def test_positive_samples(self):
        """Test detection on positive samples (should detect smiles)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING POSITIVE SAMPLES (Should detect smiles)")
        self.logger.info("="*50 + "\n")
        
        total_success = 0
        total_count = 0
        
        # Process each subfolder separately
        for subfolder in self.positive_dir.rglob('*'):
            if subfolder.is_dir():
                stats = self._process_subfolder(subfolder, expected_smile=True)
                if stats['total'] > 0:
                    success_rate = (stats['detected'] / stats['total']) * 100
                    self.subfolder_stats[str(subfolder.relative_to(self.positive_dir))] = {
                        'success_rate': success_rate,
                        'detected': stats['detected'],
                        'total': stats['total']
                    }
                    total_success += stats['detected']
                    total_count += stats['total']
        
        if total_count == 0:
            self.skipTest("No test images found in positive directories")
            
        success_rate = (total_success / total_count) * 100
        self.assertGreater(success_rate, 50.0, "Detection rate for positive samples is too low")

    def test_negative_samples(self):
        """Test detection on negative samples (should not detect smiles)"""
        self.logger.info("\n" + "="*50)
        self.logger.info("TESTING NEGATIVE SAMPLES (Should NOT detect smiles)")
        self.logger.info("="*50 + "\n")
        
        total_success = 0
        total_count = 0
        
        # Process each subfolder separately
        for subfolder in self.negative_dir.rglob('*'):
            if subfolder.is_dir():
                stats = self._process_subfolder(subfolder, expected_smile=False)
                if stats['total'] > 0:
                    success_rate = ((stats['total'] - stats['detected']) / stats['total']) * 100
                    self.subfolder_stats[str(subfolder.relative_to(self.negative_dir))] = {
                        'success_rate': success_rate,
                        'not_detected': stats['total'] - stats['detected'],
                        'total': stats['total']
                    }
                    total_success += (stats['total'] - stats['detected'])
                    total_count += stats['total']
        
        if total_count == 0:
            self.skipTest("No test images found in negative directories")
            
        success_rate = (total_success / total_count) * 100
        self.assertGreater(success_rate, 50.0, "Rejection rate for negative samples is too low")

    def tearDown(self):
        """Print summary statistics after tests"""
        self.logger.info("\n=== Subfolder Statistics ===")
        for subfolder, stats in self.subfolder_stats.items():
            self.logger.info(f"{subfolder}: {stats['success_rate']:.2f}% "
                           f"({stats.get('detected', stats.get('not_detected', 0))}/{stats['total']})")
        
        self.logger.info("\n=== Overall Results ===")
        total = sum(self.confusion_matrix.values())
        if total > 0:
            self.logger.info(f"True Positives: {self.confusion_matrix['true_positive']} "
                           f"({self.confusion_matrix['true_positive']/total*100:.2f}%)")
            self.logger.info(f"False Positives: {self.confusion_matrix['false_positive']} "
                           f"({self.confusion_matrix['false_positive']/total*100:.2f}%)")
            self.logger.info(f"True Negatives: {self.confusion_matrix['true_negative']} "
                           f"({self.confusion_matrix['true_negative']/total*100:.2f}%)")
            self.logger.info(f"False Negatives: {self.confusion_matrix['false_negative']} "
                           f"({self.confusion_matrix['false_negative']/total*100:.2f}%)")

if __name__ == '__main__':
    unittest.main(verbosity=2)