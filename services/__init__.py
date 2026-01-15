"""Services module."""
from .image_processor import darken_image, enhance_classical, enhance_deep
from .face_detector import detect_faces
from .face_recognition_service import face_recognition_service

# Try to import recognizers (optional)
_exports = [
    'darken_image', 'enhance_classical', 'enhance_deep', 
    'detect_faces', 'face_recognition_service'
]

try:
    from .arcface_recognizer import ArcFaceRecognizer
    _exports.append('ArcFaceRecognizer')
except ImportError:
    pass

try:
    from .mobilefacenet_recognizer import MobileFaceNetRecognizer
    _exports.append('MobileFaceNetRecognizer')
except ImportError:
    pass

__all__ = _exports

