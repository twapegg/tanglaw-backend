"""
Configuration settings for the Image Processing Server.
"""
import os

# Directories
UPLOAD_FOLDER = "temp_uploads"
OUTPUT_FOLDER = "temp_outputs"

# File upload settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

# Zero-DCE model settings
ZERO_DCE_BASE_PATH = os.path.join(os.getcwd(), "Zero-DCE", "Zero-DCE_code")
ZERO_DCE_MODEL_PATH = os.path.join(ZERO_DCE_BASE_PATH, "snapshots", "Epoch99.pth")

# Image processing defaults
DEFAULT_GAMMA = 1.1
DEFAULT_CURVE = 1.4
DEFAULT_NOISE_AMOUNT = 0.005
DEFAULT_BRIGHT_FACTOR = 0.3

# Enhancement settings
CLAHE_CLIP_LIMIT = 2
CLAHE_TILE_GRID_SIZE = (8, 8)
ENHANCEMENT_GAMMA = 1.5
DENOISE_STRENGTH = 5

# JPEG quality for encoding
JPEG_QUALITY = 95

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

# Logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "server.log"
