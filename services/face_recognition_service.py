import logging
from typing import Dict, Optional

from models.face_database import face_database

try:
    from services.arcface_recognizer import ArcFaceRecognizer
    ARCFACE_AVAILABLE = True
except ImportError:
    ARCFACE_AVAILABLE = False
    logging.warning("ArcFaceRecognizer not found. Install: pip install insightface onnxruntime")

try:
    from services.mobilefacenet_recognizer import MobileFaceNetRecognizer
    MOBILEFACENET_AVAILABLE = True
except ImportError:
    MOBILEFACENET_AVAILABLE = False
    logging.warning("MobileFaceNetRecognizer not found. Check mobilefacenet dependencies")


class FaceRecognitionService:
    
    def __init__(self, model_type: str = 'arcface'):
        """
        Initialize face recognition service
        
        Args:
            model_type: 'arcface' or 'mobilefacenet' (Note: MobileFaceNet requires pretrained weights)
        """
        self.model_type = model_type
        self.recognizer = None
        self.arcface_recognizer = None
        self.mobilefacenet_recognizer = None
        self._initialize_recognizers()
        
        # Log warning about MobileFaceNet
        if model_type == 'mobilefacenet':
            logging.warning("MobileFaceNet selected but may not have pretrained weights. Consider using ArcFace for better accuracy.")
    
    def _initialize_recognizers(self):
        """Initialize all available recognizers"""
        # Initialize ArcFace
        if ARCFACE_AVAILABLE:
            try:
                self.arcface_recognizer = ArcFaceRecognizer()
                logging.info("ArcFace recognizer initialized")
            except Exception as e:
                logging.error(f"Failed to initialize ArcFace: {e}")
        
        # Initialize MobileFaceNet
        if MOBILEFACENET_AVAILABLE:
            try:
                self.mobilefacenet_recognizer = MobileFaceNetRecognizer()
                logging.info("MobileFaceNet recognizer initialized")
            except Exception as e:
                logging.error(f"Failed to initialize MobileFaceNet: {e}")
        
        # Set default recognizer
        self.set_model(self.model_type)
    
    def set_model(self, model_type: str) -> bool:
        """
        Set the active model
        
        Args:
            model_type: 'arcface' or 'mobilefacenet'
            
        Returns:
            True if successful, False otherwise
        """
        if model_type == 'arcface':
            if self.arcface_recognizer is not None:
                self.recognizer = self.arcface_recognizer
                self.model_type = 'arcface'
                logging.info("Switched to ArcFace model")
                return True
            else:
                logging.warning("ArcFace model not available")
                return False
        elif model_type == 'mobilefacenet':
            if self.mobilefacenet_recognizer is not None:
                self.recognizer = self.mobilefacenet_recognizer
                self.model_type = 'mobilefacenet'
                logging.info("Switched to MobileFaceNet model")
                return True
            else:
                logging.warning("MobileFaceNet model not available")
                return False
        else:
            logging.error(f"Unknown model type: {model_type}")
            return False
    
    def get_current_model(self) -> str:
        """Get the currently active model type"""
        return self.model_type
    
    def get_available_models(self) -> Dict:
        """Get list of available models"""
        return {
            "arcface": ARCFACE_AVAILABLE and self.arcface_recognizer is not None,
            "mobilefacenet": MOBILEFACENET_AVAILABLE and self.mobilefacenet_recognizer is not None
        }
    
    def is_available(self) -> bool:
        return self.recognizer is not None
    
    def enroll_face(self, image_path: str, name: str, model_type: Optional[str] = None) -> Dict:
        """
        Enroll a face in the database
        
        Args:
            image_path: Path to the image file
            name: Person's name
            model_type: Optional model type to use (if None, uses current model)
            
        Returns:
            Dictionary with enrollment result
        """
        # Use specified model or default to current
        if model_type and model_type != self.model_type:
            original_model = self.model_type
            if not self.set_model(model_type):
                return {
                    "success": False,
                    "error": f"Model {model_type} not available"
                }
            # Restore after enrollment
            restore_model = True
        else:
            restore_model = False
            original_model = None
        
        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }
        
        try:
            self.recognizer.add_face(image_path, name)
            
            return {
                "success": True,
                "name": name,
                "model": self.model_type,
                "message": f"Successfully enrolled {name} using {self.model_type}",
                "total_faces": len(self.recognizer.known_faces)
            }
            
        except Exception as e:
            logging.error(f"Failed to enroll face: {e}")
            return {
                "success": False,
                "error": str(e),
                "status_code": 400 if isinstance(e, ValueError) else 500
            }
        finally:
            if restore_model and original_model:
                self.set_model(original_model)
    
    def recognize_face(
        self,
        image_path: str,
        threshold: float = 0.4,
        model_type: Optional[str] = None
    ) -> Dict:
        """
        Recognize faces in an image.
        
        Args:
            image_path: Path to the image file
            threshold: Recognition confidence threshold
            model_type: Optional model type to use (if None, uses current model)
            
        Returns:
            Dictionary with recognition results
        """
        # Use specified model or default to current
        if model_type and model_type != self.model_type:
            original_model = self.model_type
            if not self.set_model(model_type):
                return {
                    "success": False,
                    "error": f"Model {model_type} not available"
                }
            restore_model = True
        else:
            restore_model = False
            original_model = None
        
        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }
        
        try:
            # Perform recognition with threshold
            # Both ArcFace and MobileFaceNet now use normalized similarity scores (0-1)
            results = self.recognizer.recognize_face(image_path, threshold=threshold)
            logging.info(f"Recognizer returned {len(results)} results")
            
            if not results:
                return {
                    "success": True,
                    "found": False,
                    "model": self.model_type,
                    "message": "No faces detected or no matches found"
                }
            
            # Process results
            faces = []
            for name, similarity, bbox in results:
                face_data = {
                    "name": name,
                    "confidence": float(similarity),
                    "recognized": name != "Unknown",
                    "bbox": bbox  # Already converted to list in recognizer
                }
                logging.info(f"Processed face data: {face_data}")
                faces.append(face_data)
            
            final_result = {
                "success": True,
                "found": True,
                "model": self.model_type,
                "faces": faces,
                "count": len(faces)
            }
            logging.info(f"Final recognition result: {final_result}")
            return final_result
            
        except Exception as e:
            logging.error(f"Failed to recognize face: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if restore_model and original_model:
                self.set_model(original_model)
    
    def draw_annotated_image(self, image_path: str, faces: list) -> bytes:
        """
        Draw bounding boxes and names on the image.
        
        Args:
            image_path: Path to the original image
            faces: List of face data with name, confidence, and bbox
            
        Returns:
            Annotated image as bytes
        """
        import cv2
        import numpy as np
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not read image")
        
        for face in faces:
            bbox = face['bbox']
            name = face['name']
            confidence = face['confidence']
            
            # Ensure bbox is a list with 4 elements
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                logging.warning(f"Invalid bbox format for {name}: {bbox}, skipping")
                continue
            
            # Draw bounding box
            try:
                x1, y1, x2, y2 = [int(coord) for coord in bbox]
            except (ValueError, TypeError) as e:
                logging.warning(f"Could not parse bbox {bbox}: {e}, skipping")
                continue
                
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for recognized, red for unknown
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            
            # Prepare text
            if name != "Unknown":
                text = f"{name} ({confidence*100:.1f}%)"
            else:
                text = "Unknown"
            
            # Calculate text size and background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
            
            # Draw text background
            text_y = y1 - 10 if y1 - 10 > text_height else y1 + text_height + 10
            cv2.rectangle(img, (x1, text_y - text_height - baseline), 
                         (x1 + text_width, text_y + baseline), color, -1)
            
            # Draw text
            cv2.putText(img, text, (x1, text_y), font, font_scale, (255, 255, 255), thickness)
        
        # Encode image to bytes
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()

    def verify_face(
        self,
        image_path: str,
        name: str,
        threshold: float = 0.4,
        model_type: Optional[str] = None
    ) -> Dict:
        """
        Verify a claimed identity (1:1).

        Args:
            image_path: Path to the image file
            name: Claimed identity to verify
            threshold: Verification threshold (distance for ArcFace, similarity for MobileFaceNet)
            model_type: Optional model type to use (if None, uses current model)

        Returns:
            Dictionary with verification results
        """
        if model_type and model_type != self.model_type:
            original_model = self.model_type
            if not self.set_model(model_type):
                return {
                    "success": False,
                    "error": f"Model {model_type} not available"
                }
            restore_model = True
        else:
            restore_model = False
            original_model = None

        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }

        try:
            if not hasattr(self.recognizer, "verify_face"):
                return {
                    "success": False,
                    "error": "Verification not supported for current model"
                }

            result = self.recognizer.verify_face(
                image_path,
                name,
                threshold=threshold
            )

            if result.get("success"):
                result["model"] = self.model_type
                result["name"] = name

            return result

        except Exception as e:
            logging.error(f"Failed to verify face: {e}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            if restore_model and original_model:
                self.set_model(original_model)
    
    def list_enrolled_faces(self) -> Dict:
        """
        Get list of all enrolled faces.
        
        Returns:
            Dictionary with list of names
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }
        
        return {
            "success": True,
            "model": self.model_type,
            "names": list(self.recognizer.known_faces.keys()),
            "count": len(self.recognizer.known_faces)
        }
    
    def remove_face(self, name: str) -> Dict:
        """
        Remove a face from the database.
        
        Args:
            name: Person's name to remove
            
        Returns:
            Dictionary with operation result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }
        
        # Use the recognizer's remove_face method if available
        if hasattr(self.recognizer, 'remove_face'):
            success = self.recognizer.remove_face(name)
        else:
            # Fallback: manually delete and save
            if name in self.recognizer.known_faces:
                del self.recognizer.known_faces[name]
                success = True
                # Save database if using MobileFaceNet (has _save_database method)
                if hasattr(self.recognizer, '_save_database'):
                    self.recognizer._save_database()
            else:
                success = False
        
        if success:
            return {
                "success": True,
                "model": self.model_type,
                "message": f"Successfully removed {name}",
                "remaining_faces": len(self.recognizer.known_faces)
            }
        else:
            return {
                "success": False,
                "error": f"Face '{name}' not found in database"
            }
    
    def clear_database(self) -> Dict:
        """
        Clear all faces from the database.
        
        Returns:
            Dictionary with operation result
        """
        if not self.is_available():
            return {
                "success": False,
                "error": "Face recognition service not available"
            }
        
        if hasattr(self.recognizer, 'clear_database'):
            self.recognizer.clear_database()
        else:
            self.recognizer.known_faces = {}
        
        return {
            "success": True,
            "model": self.model_type,
            "message": "Database cleared successfully"
        }


# Global face recognition service instance
face_recognition_service = FaceRecognitionService()
