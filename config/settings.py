"""
Configuration settings for the Image Processing Server.
"""
import os

# Directories
UPLOAD_FOLDER = "temp_uploads"
OUTPUT_FOLDER = "temp_outputs"

# File upload settings
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 256 * 1024 * 1024  # 256MB (increase for video uploads)

# Zero-DCE model settings
ZERO_DCE_BASE_PATH = os.path.join(os.getcwd(), "Zero-DCE", "Zero-DCE_code")
ZERO_DCE_MODEL_PATH = os.path.join(ZERO_DCE_BASE_PATH, "snapshots", "Epoch99.pth")

# Image processing defaults
DEFAULT_GAMMA = 1.1
DEFAULT_CURVE = 1.4
DEFAULT_NOISE_AMOUNT = 0.005
DEFAULT_BRIGHT_FACTOR = 0.3

# Enhancement settings (gentler classic)
CLAHE_CLIP_LIMIT = 2.6
CLAHE_TILE_GRID_SIZE = (8, 8)
ENHANCEMENT_GAMMA = 1.5
DENOISE_STRENGTH = 6

# Post-enhancement boost for deep model output
DEEP_POST_CLAHE_CLIP_LIMIT = 2.5
DEEP_POST_GAMMA = 1.4

ENHANCEMENT_LUMA_LOW = 70
ENHANCEMENT_LUMA_VERY_LOW = 45
ENHANCEMENT_CLAHE_CLIP_LIMIT_LOW = 3.1
ENHANCEMENT_CLAHE_CLIP_LIMIT_VERY_LOW = 4.2
ENHANCEMENT_GAMMA_LOW = 1.6
ENHANCEMENT_GAMMA_VERY_LOW = 2.1
ENHANCEMENT_DENOISE_LOW = 5
ENHANCEMENT_DENOISE_VERY_LOW = 7
ENHANCEMENT_GAIN_LOW = 1.04
ENHANCEMENT_GAIN_VERY_LOW = 1.08
ENHANCEMENT_BIAS_LOW = 4
ENHANCEMENT_BIAS_VERY_LOW = 7
ENHANCEMENT_DEEP_POSTBOOST_LUMA = 60
ENHANCEMENT_DEEP_POSTBOOST_GAMMA = 1.7
ENHANCEMENT_DEEP_POSTBOOST_GAIN = 1.06
ENHANCEMENT_DEEP_POSTBOOST_BIAS = 6

# JPEG quality for encoding
JPEG_QUALITY = 95

# Server defaults
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

# Logging
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FILE = "server.log"
