"""
RetinaFace Detector - Wraps InsightFace's RetinaFace detector
"""
import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis


class RetinaFaceDetector:
    """RetinaFace detector using InsightFace backend"""
    
    def __init__(self, det_size=(640, 640)):
        """
        Initialize RetinaFace detector
        
        Args:
            det_size: Detection size (width, height)
        """
        self.det_size = det_size
        
        logging.info("Initializing RetinaFace detector...")
        import warnings
        from contextlib import redirect_stdout, redirect_stderr
        import io
        
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=det_size)
        
        logging.info("RetinaFace detector initialized successfully")
    
    def detect_faces(self, img):
        """
        Detect faces in an image
        
        Args:
            img: BGR image (numpy array)
            
        Returns:
            List of detection dictionaries with format similar to MTCNN:
            [
                {
                    'box': [x, y, width, height],
                    'confidence': float,
                    'keypoints': {
                        'left_eye': (x, y),
                        'right_eye': (x, y),
                        'nose': (x, y),
                        'mouth_left': (x, y),
                        'mouth_right': (x, y)
                    }
                }
            ]
        """
        try:
            faces = self.app.get(img)
            
            detections = []
            for face in faces:
                # Convert bbox format: [x1, y1, x2, y2] -> [x, y, width, height]
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                x, y, width, height = x1, y1, x2 - x1, y2 - y1
                
                # Get confidence (det_score)
                confidence = float(face.det_score)
                
                # Get landmarks/keypoints
                # InsightFace provides 5 landmarks: left_eye, right_eye, nose, mouth_left, mouth_right
                kps = face.kps.astype(int)
                keypoints = {
                    'left_eye': tuple(kps[0]),
                    'right_eye': tuple(kps[1]),
                    'nose': tuple(kps[2]),
                    'mouth_left': tuple(kps[3]),
                    'mouth_right': tuple(kps[4])
                }
                
                detection = {
                    'box': [x, y, width, height],
                    'confidence': confidence,
                    'keypoints': keypoints
                }
                
                detections.append(detection)
            
            return detections
            
        except Exception as e:
            logging.error(f"RetinaFace detection failed: {e}")
            return []
    
    def detect_faces_with_visualization(self, img):
        """
        Detect faces and return annotated image
        
        Args:
            img: BGR image (numpy array)
            
        Returns:
            Tuple of (annotated_image, detection_count)
        """
        detections = self.detect_faces(img)
        
        if not detections:
            return img, 0
        
        result_img = img.copy()
        
        for detection in detections:
            x, y, width, height = detection['box']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(
                result_img,
                f"Conf: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            
            # Draw landmarks
            keypoints = detection['keypoints']
            landmarks = {
                'left_eye': (keypoints['left_eye'], (255, 0, 0)),
                'right_eye': (keypoints['right_eye'], (0, 255, 255)),
                'nose': (keypoints['nose'], (0, 0, 255)),
                'mouth_left': (keypoints['mouth_left'], (255, 255, 0)),
                'mouth_right': (keypoints['mouth_right'], (255, 0, 255)),
            }
            
            for name, (point, color) in landmarks.items():
                cv2.circle(result_img, point, 5, color, -1)
                cv2.putText(
                    result_img,
                    name.replace('_', ' ').title(),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
        
        return result_img, len(detections)
