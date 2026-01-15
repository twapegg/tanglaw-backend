"""
File utilities for handling file uploads and validation.
"""
from werkzeug.utils import secure_filename

from config import ALLOWED_EXTENSIONS


def allowed_file(filename):
    """
    Check if file extension is allowed.
    
    Args:
        filename: Name of the file to check
        
    Returns:
        Boolean indicating if file type is allowed
    """
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_secure_filename(filename, request_id):
    """
    Generate a secure unique filename.
    
    Args:
        filename: Original filename
        request_id: Unique request identifier
        
    Returns:
        Secure unique filename
    """
    secure_name = secure_filename(filename)
    return f"{request_id}_{secure_name}"
