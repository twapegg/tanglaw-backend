#!/usr/bin/env python3
"""
Image Processing Pipeline Server
Flask-based REST API for image processing with darkening, enhancement, and face detection.

Usage:
    python server.py [--port PORT] [--host HOST]
"""

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import uuid
import time
import logging
from datetime import datetime
import io
import base64
import argparse
import sys
from pathlib import Path
import torch
from mtcnn import MTCNN

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = "temp_uploads"
OUTPUT_FOLDER = "temp_outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "tiff", "webp"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("server.log"), logging.StreamHandler(sys.stdout)],
)

# Initialize models
detector = None
deep_model = None
device = None


def initialize_models():
    """Initialize MTCNN detector and Zero-DCE model."""
    global detector, deep_model, device

    try:
        detector = MTCNN()
        logging.info("MTCNN detector initialized")
    except Exception as e:
        logging.error(f"Failed to initialize MTCNN: {e}")

    try:
        zero_dce_path = os.path.join(os.getcwd(), "Zero-DCE", "Zero-DCE_code")
        if os.path.exists(zero_dce_path):
            if zero_dce_path not in sys.path:
                sys.path.append(zero_dce_path)

            from model import enhance_net_nopool

            model_path = os.path.join(zero_dce_path, "snapshots", "Epoch99.pth")
            if os.path.exists(model_path):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                deep_model = enhance_net_nopool().to(device)
                deep_model.load_state_dict(torch.load(model_path, map_location=device))
                deep_model.eval()
                logging.info(f"Zero-DCE model loaded on {device}")
            else:
                logging.warning("Zero-DCE model file not found")
        else:
            logging.warning("Zero-DCE directory not found")
    except Exception as e:
        logging.error(f"Failed to load Zero-DCE model: {e}")


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def darken_image(
    img_array, percent, noise_amount=0.005, gamma=1.1, curve=1.4, bright_factor=0.3
):
    """Darken an image."""
    arr = img_array.astype("float32") / 255.0
    linear = arr**gamma
    linear_dark = linear * (1 - percent / 100.0)
    linear_dark = linear_dark**curve
    linear_dark = linear_dark * (1 - bright_factor * linear)

    noise = np.random.normal(0.0, noise_amount, linear_dark.shape)
    linear_noisy = np.clip(linear_dark + noise, 0.0, 1.0)

    srgb = np.clip(linear_noisy ** (1.0 / gamma), 0, 1)
    out = (srgb * 255).astype("uint8")
    return out


def enhance_classical(img, gamma=1.5, denoise_strength=5):
    """Apply classical enhancement (CLAHE + gamma + denoising)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    gamma_inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** gamma_inv * 255 for i in range(256)]).astype(
        "uint8"
    )
    enhanced_gamma = cv2.LUT(enhanced, table)

    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced_gamma,
        None,
        h=denoise_strength,
        hColor=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return denoised


def enhance_deep(img):
    """Apply deep learning enhancement using Zero-DCE."""
    if deep_model is None or device is None:
        return None

    try:
        img_norm = img.astype(np.float32) / 255.0
        img_input = (
            torch.from_numpy(np.transpose(img_norm, (2, 0, 1))).unsqueeze(0).to(device)
        )

        with torch.no_grad():
            _, enhanced, _ = deep_model(img_input)

        enhanced_img = enhanced.cpu().numpy()
        enhanced_img = np.clip(
            np.transpose(enhanced_img[0], (1, 2, 0)) * 255, 0, 255
        ).astype(np.uint8)

        return enhanced_img
    except Exception as e:
        logging.error(f"Deep enhancement failed: {e}")
        return None


def detect_faces(img):
    """Detect faces and draw landmarks."""
    if detector is None:
        return img, 0

    try:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(rgb_image)

        if not detections:
            return img, 0

        result_img = img.copy()

        for detection in detections:
            x, y, width, height = detection["box"]
            confidence = detection["confidence"]

            cv2.rectangle(result_img, (x, y), (x + width, y + height), (0, 255, 0), 2)
            cv2.putText(
                result_img,
                f"Conf: {confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            keypoints = detection["keypoints"]
            landmarks = {
                "left_eye": (keypoints["left_eye"], (255, 0, 0)),
                "right_eye": (keypoints["right_eye"], (0, 255, 255)),
                "nose": (keypoints["nose"], (0, 0, 255)),
                "mouth_left": (keypoints["mouth_left"], (255, 255, 0)),
                "mouth_right": (keypoints["mouth_right"], (255, 0, 255)),
            }

            for name, (point, color) in landmarks.items():
                cv2.circle(result_img, point, 5, color, -1)
                cv2.putText(
                    result_img,
                    name.replace("_", " ").title(),
                    (point[0] + 10, point[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )

        return result_img, len(detections)
    except Exception as e:
        logging.error(f"Face detection failed: {e}")
        return img, 0


def image_to_base64(img):
    """Convert OpenCV image to base64 string."""
    _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    img_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_base64


def process_single_image(image_path):
    """Process a single image through the entire pipeline."""
    start_time = time.time()

    # Dictionary to collect all processed images
    processed_images = {}

    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image from {image_path}")

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # 1. Baseline - original image with face detection
        baseline_detected, baseline_faces = detect_faces(original)
        processed_images["baseline"] = {
            "type": "baseline",
            "label": "Baseline",
            "darkening_level": 0,
            "enhancement": "none",
            "faces_detected": baseline_faces,
            "image": image_to_base64(baseline_detected),
        }

        # Process darkening levels
        darkened_50 = darken_image(original_rgb, 50)
        darkened_50_bgr = cv2.cvtColor(darkened_50, cv2.COLOR_RGB2BGR)

        darkened_80 = darken_image(original_rgb, 80)
        darkened_80_bgr = cv2.cvtColor(darkened_80, cv2.COLOR_RGB2BGR)

        # 2. Darkened (50%)
        darkened_50_detected, darkened_50_faces = detect_faces(darkened_50_bgr)
        processed_images["darkened_50"] = {
            "type": "50_darkened",
            "label": "Darkened (50%)",
            "darkening_level": 50,
            "enhancement": "none",
            "faces_detected": darkened_50_faces,
            "image": image_to_base64(darkened_50_detected),
        }

        # 3. Darkened (80%)
        darkened_80_detected, darkened_80_faces = detect_faces(darkened_80_bgr)
        processed_images["darkened_80"] = {
            "type": "80_darkened",
            "label": "Darkened (80%)",
            "darkening_level": 80,
            "enhancement": "none",
            "faces_detected": darkened_80_faces,
            "image": image_to_base64(darkened_80_detected),
        }

        # 4. Enhanced (Classical, 50%)
        classical_50_enhanced = enhance_classical(darkened_50_bgr)
        classical_50_detected, classical_50_faces = detect_faces(classical_50_enhanced)
        processed_images["classical_50"] = {
            "type": "50_darkened_classical",
            "label": "Enhanced (Classical, 50%)",
            "darkening_level": 50,
            "enhancement": "classical",
            "faces_detected": classical_50_faces,
            "image": image_to_base64(classical_50_detected),
        }

        # 5. Enhanced (Classical, 80%)
        classical_80_enhanced = enhance_classical(darkened_80_bgr)
        classical_80_detected, classical_80_faces = detect_faces(classical_80_enhanced)
        processed_images["classical_80"] = {
            "type": "80_darkened_classical",
            "label": "Enhanced (Classical, 80%)",
            "darkening_level": 80,
            "enhancement": "classical",
            "faces_detected": classical_80_faces,
            "image": image_to_base64(classical_80_detected),
        }

        # 6 & 7. Deep learning enhancement
        if deep_model is not None:
            deep_50_enhanced = enhance_deep(darkened_50_bgr)
            if deep_50_enhanced is not None:
                deep_50_detected, deep_50_faces = detect_faces(deep_50_enhanced)
                processed_images["deep_50"] = {
                    "type": "50_darkened_deep",
                    "label": "Enhanced (Deep, 50%)",
                    "darkening_level": 50,
                    "enhancement": "deep",
                    "faces_detected": deep_50_faces,
                    "image": image_to_base64(deep_50_detected),
                }

            deep_80_enhanced = enhance_deep(darkened_80_bgr)
            if deep_80_enhanced is not None:
                deep_80_detected, deep_80_faces = detect_faces(deep_80_enhanced)
                processed_images["deep_80"] = {
                    "type": "80_darkened_deep",
                    "label": "Enhanced (Deep, 80%)",
                    "darkening_level": 80,
                    "enhancement": "deep",
                    "faces_detected": deep_80_faces,
                    "image": image_to_base64(deep_80_detected),
                }

        # Return in specific order
        ordered_results = []
        order = [
            "baseline",
            "darkened_50",
            "darkened_80",
            "classical_50",
            "classical_80",
            "deep_50",
            "deep_80",
        ]
        for key in order:
            if key in processed_images:
                ordered_results.append(processed_images[key])

        processing_time = time.time() - start_time

        return {
            "success": True,
            "processing_time": round(processing_time, 2),
            "total_images": len(ordered_results),
            "images": ordered_results,
        }

    except Exception as e:
        logging.error(f"Error processing image: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


@app.route("/")
def index():
    """Root endpoint with API information."""
    return jsonify(
        {
            "service": "Image Processing Pipeline API",
            "version": "1.0.0",
            "endpoints": {
                "/": "API information",
                "/health": "Health check",
                "/process_image": "POST - Process an image through the pipeline",
            },
            "status": "running",
        }
    )


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "mtcnn": detector is not None,
                "zero_dce": deep_model is not None,
            },
        }
    )


@app.route("/process_image", methods=["POST"])
def process_image():
    """
    Process an image through the complete pipeline.

    Expects:
        - multipart/form-data with 'image' field containing the image file

    Returns:
        JSON with list of processed images in base64 format
    """
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Processing request")

    # Check if image file is present
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]

    # Check if file is selected
    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    # Check if file type is allowed
    if not allowed_file(file.filename):
        return (
            jsonify(
                {
                    "success": False,
                    "error": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                }
            ),
            400,
        )

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{request_id}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        logging.info(f"[{request_id}] File saved: {filename}")

        # Process the image
        result = process_single_image(filepath)

        # Cleanup uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")

        if result["success"]:
            logging.info(
                f"[{request_id}] Processing complete: {result['total_images']} images generated in {result['processing_time']}s"
            )
            return jsonify(result), 200
        else:
            logging.error(f"[{request_id}] Processing failed: {result['error']}")
            return jsonify(result), 500

    except Exception as e:
        logging.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/process_image_files", methods=["POST"])
def process_image_files():
    """
    Process an image and return files instead of base64.
    Returns a ZIP file containing all processed images.
    """
    import zipfile

    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Processing request (file mode)")

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        unique_filename = f"{request_id}_{filename}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)

        # Load and process image
        original = cv2.imread(filepath)
        if original is None:
            return jsonify({"success": False, "error": "Could not read image"}), 400

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Create output directory for this request
        output_dir = os.path.join(app.config["OUTPUT_FOLDER"], request_id)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(filename)[0]

        # 1. Baseline - original image with face detection
        baseline_detected, _ = detect_faces(original)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_baseline_detected.jpg"),
            baseline_detected,
        )

        # Process darkening
        darkened_50 = darken_image(original_rgb, 50)
        darkened_50_bgr = cv2.cvtColor(darkened_50, cv2.COLOR_RGB2BGR)

        darkened_80 = darken_image(original_rgb, 80)
        darkened_80_bgr = cv2.cvtColor(darkened_80, cv2.COLOR_RGB2BGR)

        # 2. Darkened (50%)
        darkened_50_detected, _ = detect_faces(darkened_50_bgr)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_50_darkened_detected.jpg"),
            darkened_50_detected,
        )

        # 3. Darkened (80%)
        darkened_80_detected, _ = detect_faces(darkened_80_bgr)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_80_darkened_detected.jpg"),
            darkened_80_detected,
        )

        # 4. Enhanced (Classical, 50%)
        classical_50_enhanced = enhance_classical(darkened_50_bgr)
        classical_50_detected, _ = detect_faces(classical_50_enhanced)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_50_darkened_classical_detected.jpg"),
            classical_50_detected,
        )

        # 5. Enhanced (Classical, 80%)
        classical_80_enhanced = enhance_classical(darkened_80_bgr)
        classical_80_detected, _ = detect_faces(classical_80_enhanced)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_80_darkened_classical_detected.jpg"),
            classical_80_detected,
        )

        # 6 & 7. Deep learning enhancement
        if deep_model is not None:
            deep_50_enhanced = enhance_deep(darkened_50_bgr)
            if deep_50_enhanced is not None:
                deep_50_detected, _ = detect_faces(deep_50_enhanced)
                cv2.imwrite(
                    os.path.join(
                        output_dir, f"{base_name}_50_darkened_deep_detected.jpg"
                    ),
                    deep_50_detected,
                )

            deep_80_enhanced = enhance_deep(darkened_80_bgr)
            if deep_80_enhanced is not None:
                deep_80_detected, _ = detect_faces(deep_80_enhanced)
                cv2.imwrite(
                    os.path.join(
                        output_dir, f"{base_name}_80_darkened_deep_detected.jpg"
                    ),
                    deep_80_detected,
                )

        # Create ZIP file
        zip_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{request_id}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)

        # Cleanup
        os.remove(filepath)
        import shutil

        shutil.rmtree(output_dir)

        return send_file(
            zip_path, as_attachment=True, download_name=f"processed_{filename}.zip"
        )

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Image Processing Pipeline Server")
    parser.add_argument(
        "--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=5000, help="Port to bind to (default: 5000)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    logging.info("Initializing models...")
    initialize_models()
    logging.info("Server initialization complete")

    logging.info(f"Starting server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
