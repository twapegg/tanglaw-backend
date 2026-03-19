"""
RetinaFace Detector - Wraps InsightFace's RetinaFace detector
"""
import cv2
import numpy as np
import logging
from insightface.app import FaceAnalysis

from config import RETINAFACE_MODEL_CANDIDATES

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
        
        self.app = None
        last_error = None
        model_candidates = RETINAFACE_MODEL_CANDIDATES or ("buffalo_l",)

        for model_name in model_candidates:
            try:
                with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    warnings.simplefilter("ignore")
                    self.app = FaceAnalysis(
                        name=model_name,
                        allowed_modules=["detection"],
                        providers=["CPUExecutionProvider"],
                    )
                    self.app.prepare(ctx_id=0, det_size=det_size)

                logging.info(f"RetinaFace detector initialized with model pack '{model_name}'")
                break
            except Exception as e:
                last_error = e
                self.app = None
                logging.warning(f"RetinaFace model pack '{model_name}' unavailable: {e}")

        if self.app is None:
            raise RuntimeError(f"Failed to initialize RetinaFace detector: {last_error}")
    
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
    
    def detect_faces_with_visualization(self, img, include_confidences=False):
        """
        Detect faces and return annotated image
        
        Args:
            img: BGR image (numpy array)
            
        Returns:
            Tuple of:
                - (annotated_image, detection_count) when include_confidences=False
                - (annotated_image, detection_count, confidences) when include_confidences=True
        """
        detections = self.detect_faces(img)
        
        if not detections:
            if include_confidences:
                return img, 0, []
            return img, 0
        
        result_img = img.copy()
        
        confidences = []
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = max(0.60, min(0.95, min(result_img.shape[:2]) / 700.0))
        label_text_thickness = 2
        label_pad_x = 8
        label_pad_y = 6
        label_bg_color = (210, 245, 210)
        label_border_color = (130, 105, 65)
        label_text_color = (72, 45, 18)
        box_color = (34, 132, 72)

        for detection in detections:
            x, y, width, height = detection['box']
            confidence = float(detection['confidence'])
            confidences.append(max(0.0, min(1.0, confidence)))
            
            # Draw bounding box
            cv2.rectangle(result_img, (x, y), (x + width, y + height), box_color, 3)
            
            # Draw print-friendly confidence label.
            label = f"Conf {confidence * 100:.1f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, label_font, label_font_scale, label_text_thickness
            )
            text_x = max(
                label_pad_x,
                min(x + 2, result_img.shape[1] - text_width - label_pad_x - 1),
            )
            candidate_baseline = y - 8
            min_baseline = text_height + baseline + label_pad_y + 1
            if candidate_baseline < min_baseline:
                candidate_baseline = y + text_height + baseline + label_pad_y + 2
            text_baseline = min(
                result_img.shape[0] - baseline - label_pad_y - 1,
                max(min_baseline, candidate_baseline),
            )
            bg_x1 = max(0, text_x - label_pad_x)
            bg_y1 = max(0, text_baseline - text_height - baseline - label_pad_y)
            bg_x2 = min(result_img.shape[1] - 1, text_x + text_width + label_pad_x)
            bg_y2 = min(result_img.shape[0] - 1, text_baseline + baseline + label_pad_y)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (bg_x2, bg_y2), label_bg_color, -1)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (bg_x2, bg_y2), label_border_color, 2)
            accent_x2 = min(bg_x2, bg_x1 + 4)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (accent_x2, bg_y2), box_color, -1)
            cv2.putText(
                result_img,
                label,
                (
                    text_x,
                    max(text_height, min(text_baseline, result_img.shape[0] - baseline - 1)),
                ),
                label_font,
                label_font_scale,
                label_text_color,
                label_text_thickness,
                cv2.LINE_AA,
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
        
        if include_confidences:
            return result_img, len(detections), confidences
        return result_img, len(detections)
