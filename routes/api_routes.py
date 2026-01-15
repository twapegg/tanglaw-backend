import os
import cv2
import time
import uuid
import logging
import zipfile
import shutil
from flask import Blueprint, jsonify, request, send_file

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS
from models import model_manager
from services import darken_image, enhance_classical, enhance_deep, detect_faces
from utils import allowed_file, get_secure_filename, image_to_base64


api = Blueprint('api', __name__)


def process_single_image(image_path):
    start_time = time.time()
    processed_images = {}

    try:
        # Load original image
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Could not read image from {image_path}")

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        baseline_detected, baseline_faces = detect_faces(original)
        processed_images["baseline"] = {
            "type": "baseline",
            "label": "Baseline",
            "darkening_level": 0,
            "enhancement": "none",
            "faces_detected": baseline_faces,
            "image": image_to_base64(baseline_detected),
        }

        darkened_50 = darken_image(original_rgb, 50)
        darkened_50_bgr = cv2.cvtColor(darkened_50, cv2.COLOR_RGB2BGR)

        darkened_80 = darken_image(original_rgb, 80)
        darkened_80_bgr = cv2.cvtColor(darkened_80, cv2.COLOR_RGB2BGR)

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
        if model_manager.is_zero_dce_available():
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
        order = ["baseline", "darkened_50", "darkened_80", "classical_50", 
                 "classical_80", "deep_50", "deep_80"]
        ordered_results = [processed_images[key] for key in order if key in processed_images]

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


@api.route("/")
def index():
    """Root endpoint with API information."""
    return jsonify({
        "service": "Image Processing Pipeline API",
        "version": "2.0.0",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/process_image": "POST - Process an image through the pipeline",
            "/process_image_files": "POST - Process and return images as ZIP file",
            "/recognition/health": "GET - Face recognition service health",
            "/recognition/enroll": "POST - Enroll a new face",
            "/recognition/recognize": "POST - Recognize faces in an image",
            "/recognition/list": "GET - List all enrolled faces",
            "/recognition/remove/<name>": "DELETE - Remove a face from database",
            "/recognition/clear": "POST - Clear all faces from database"
        },
        "status": "running",
    })


@api.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "mtcnn": model_manager.is_mtcnn_available(),
            "zero_dce": model_manager.is_zero_dce_available(),
        },
    })


@api.route("/process_image", methods=["POST"])
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

    # Validate request
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    try:
        # Save uploaded file
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        logging.info(f"[{request_id}] File saved: {file.filename}")

        # Process the image
        result = process_single_image(filepath)

        # Cleanup uploaded file
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")

        if result["success"]:
            logging.info(
                f"[{request_id}] Processing complete: {result['total_images']} "
                f"images in {result['processing_time']}s"
            )
            return jsonify(result), 200
        else:
            logging.error(f"[{request_id}] Processing failed: {result['error']}")
            return jsonify(result), 500

    except Exception as e:
        logging.error(f"[{request_id}] Unexpected error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@api.route("/process_image_files", methods=["POST"])
def process_image_files():
    """
    Process an image and return files instead of base64.
    Returns a ZIP file containing all processed images.
    """
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Processing request (file mode)")

    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files["image"]

    if file.filename == "" or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    try:
        # Save uploaded file
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Load and process image
        original = cv2.imread(filepath)
        if original is None:
            return jsonify({"success": False, "error": "Could not read image"}), 400

        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

        # Create output directory for this request
        output_dir = os.path.join(OUTPUT_FOLDER, request_id)
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(file.filename))[0]

        # 1. Baseline
        baseline_detected, _ = detect_faces(original)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_baseline_detected.jpg"),
            baseline_detected
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
            darkened_50_detected
        )

        # 3. Darkened (80%)
        darkened_80_detected, _ = detect_faces(darkened_80_bgr)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_80_darkened_detected.jpg"),
            darkened_80_detected
        )

        # 4. Enhanced (Classical, 50%)
        classical_50_enhanced = enhance_classical(darkened_50_bgr)
        classical_50_detected, _ = detect_faces(classical_50_enhanced)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_50_darkened_classical_detected.jpg"),
            classical_50_detected
        )

        # 5. Enhanced (Classical, 80%)
        classical_80_enhanced = enhance_classical(darkened_80_bgr)
        classical_80_detected, _ = detect_faces(classical_80_enhanced)
        cv2.imwrite(
            os.path.join(output_dir, f"{base_name}_80_darkened_classical_detected.jpg"),
            classical_80_detected
        )

        # 6 & 7. Deep learning enhancement
        if model_manager.is_zero_dce_available():
            deep_50_enhanced = enhance_deep(darkened_50_bgr)
            if deep_50_enhanced is not None:
                deep_50_detected, _ = detect_faces(deep_50_enhanced)
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_50_darkened_deep_detected.jpg"),
                    deep_50_detected
                )

            deep_80_enhanced = enhance_deep(darkened_80_bgr)
            if deep_80_enhanced is not None:
                deep_80_detected, _ = detect_faces(deep_80_enhanced)
                cv2.imwrite(
                    os.path.join(output_dir, f"{base_name}_80_darkened_deep_detected.jpg"),
                    deep_80_detected
                )

        # Create ZIP file
        zip_path = os.path.join(OUTPUT_FOLDER, f"{request_id}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, file)

        # Cleanup
        os.remove(filepath)
        shutil.rmtree(output_dir)

        return send_file(
            zip_path,
            as_attachment=True,
            download_name=f"processed_{base_name}.zip"
        )

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@api.route('/enhance', methods=['POST'])
def enhance_image():
    """
    Enhance a low-light image without darkening.
    
    Expects:
        - multipart/form-data with:
            - 'image': image file
            - 'method' (optional): 'classical' or 'deep' (default: 'deep')
    
    Returns:
        JSON with enhanced image in base64
    """
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Enhancement request")

    if 'image' not in request.files:
        return jsonify({
            "success": False,
            "error": "No image file provided"
        }), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "No file selected"
        }), 400

    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400

    # Get enhancement method
    method = request.form.get('method', 'deep').lower()
    if method not in ['classical', 'deep']:
        return jsonify({
            "success": False,
            "error": "Invalid enhancement method. Use 'classical' or 'deep'"
        }), 400

    try:
        # Save uploaded file
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        logging.info(f"[{request_id}] Enhancing image with {method} method")

        # Load image
        original = cv2.imread(filepath)
        if original is None:
            raise ValueError(f"Could not read image from {filepath}")

        # Apply enhancement directly to original image (no darkening)
        if method == 'deep':
            if not model_manager.is_zero_dce_available():
                raise ValueError("Zero-DCE model not available")
            enhanced = enhance_deep(original)
            if enhanced is None:
                raise ValueError("Deep enhancement failed")
        else:  # classical
            enhanced = enhance_classical(original)

        # Convert to base64
        enhanced_base64 = image_to_base64(enhanced)

        # Cleanup
        os.remove(filepath)

        return jsonify({
            "success": True,
            "method": method,
            "enhanced_image": enhanced_base64
        }), 200

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        # Cleanup on error
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({"success": False, "error": str(e)}), 500
