"""
Image utilities for conversion and encoding.
"""
import base64
import cv2

from config import JPEG_QUALITY


def image_to_base64(img):
    """
    Convert OpenCV image to base64 string.
    
    Args:
        img: OpenCV image (BGR format)
        
    Returns:
        Base64 encoded string
    """
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64
