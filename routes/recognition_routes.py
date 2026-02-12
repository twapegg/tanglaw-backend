import os
import uuid
import logging
import time
import cv2
import numpy as np
from flask import Blueprint, request, jsonify, send_file, make_response

from config import UPLOAD_FOLDER, OUTPUT_FOLDER, ALLOWED_EXTENSIONS
from services.face_recognition_service import face_recognition_service
from services import (
    darken_image,
    detect_faces,
    enhance_classical,
    enhance_deep,
    is_detector_available,
)
from utils import allowed_file, get_secure_filename, image_to_base64


recognition = Blueprint('recognition', __name__, url_prefix='/recognition')


@recognition.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET,POST,PUT,DELETE,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    response.headers["Access-Control-Expose-Headers"] = (
        "Content-Disposition,X-Processing-Time-Ms,X-Processing-Time-Sec,"
        "X-Frame-Count,X-Processing-Speed-Fps,X-Source-Fps,X-Detector"
    )
    return response


@recognition.route('/video', methods=['OPTIONS'])
def video_options():
    return make_response("", 204)


@recognition.route('/health')
def health():
    available_models = face_recognition_service.get_available_models()
    current_model = face_recognition_service.get_current_model()
    
    return jsonify({
        "status": "healthy" if face_recognition_service.is_available() else "unavailable",
        "service": "face_recognition",
        "available": face_recognition_service.is_available(),
        "current_model": current_model,
        "available_models": available_models
    })


@recognition.route('/enroll', methods=['POST'])
def enroll_face():
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Face enrollment request")
    
    if not face_recognition_service.is_available():
        return jsonify({"success": False, "error": "Service not available"}), 503
    
    files = []
    if 'images' in request.files:
        files = request.files.getlist('images')
    elif 'image' in request.files:
        files = [request.files['image']]
    else:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    if not files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    
    model_type = request.form.get('model', None)
    
    valid_files = []
    invalid_files = []
    for file in files:
        if file.filename == '':
            invalid_files.append({
                "filename": "",
                "error": "No file selected",
                "success": False
            })
            continue
        if not allowed_file(file.filename):
            invalid_files.append({
                "filename": file.filename,
                "error": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}',
                "success": False
            })
            continue
        valid_files.append(file)
    
    if not valid_files:
        return jsonify({
            "success": False,
            "error": "No valid image files provided",
            "details": invalid_files
        }), 400
    
    try:
        results = []
        success_count = 0
        
        for file in valid_files:
            filename = get_secure_filename(file.filename, request_id)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            logging.info(f"[{request_id}] Enrolling: {name} ({file.filename}) (model: {model_type or 'default'})")
            
            result = face_recognition_service.enroll_face(filepath, name, model_type=model_type)
            
            try:
                os.remove(filepath)
            except Exception as e:
                logging.warning(f"Failed to remove temp file: {e}")
            
            entry = {
                "filename": file.filename,
                "success": bool(result.get("success")),
            }
            if result.get("success"):
                entry["message"] = result.get("message")
                success_count += 1
            else:
                entry["error"] = result.get("error")
                entry["status_code"] = result.get("status_code")
            results.append(entry)
        
        results.extend(invalid_files)
        
        failed_count = len(results) - success_count
        overall_success = success_count > 0
        
        response = {
            "success": overall_success,
            "name": name,
            "model": model_type or face_recognition_service.get_current_model(),
            "total": len(results),
            "enrolled": success_count,
            "failed": failed_count,
            "results": results
        }
        
        if overall_success:
            status_code = 200 if failed_count == 0 else 207
            logging.info(f"[{request_id}] Enrollment complete: {success_count} succeeded, {failed_count} failed")
            return jsonify(response), status_code
        
        logging.error(f"[{request_id}] Enrollment failed for all images")
        return jsonify(response), 400
            
    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@recognition.route('/recognize', methods=['POST'])
def recognize_face():
    """
    Recognize faces in an image.
    
    Expects:
        - mu- 'model' (optional): 'arcface' or 'mobilefacenet' (uses current model if not specified)
            
    Returns:
        JSON with recognition results
    """
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Face recognition request")
    
    # Check if service is available
    if not face_recognition_service.is_available():
        return jsonify({
            "success": False,
            "error": "Service not available"
        }), 503
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400
    
    model_type = request.form.get('model', None)
    current_model = model_type if model_type else face_recognition_service.get_current_model()
    default_threshold = 0.70 if current_model == 'mobilefacenet' else 0.4
    threshold = float(request.form.get('threshold', default_threshold))
    
    logging.info(f"[{request_id}] Model: {current_model}, Threshold: {threshold}")
    
    try:
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logging.info(f"[{request_id}] Recognizing faces (model: {model_type or 'default'})")
        
        result = face_recognition_service.recognize_face(filepath, threshold=threshold, model_type=model_type)
        logging.info(f"[{request_id}] Result: success={result.get('success')}, found={result.get('found')}, count={result.get('count')}")
        
        if result.get("success") and result.get("found") and result.get("faces"):
            try:
                import base64
                annotated_bytes = face_recognition_service.draw_annotated_image(filepath, result["faces"])
                if annotated_bytes:
                    result["annotated_image"] = base64.b64encode(annotated_bytes).decode('utf-8')
                    logging.info(f"[{request_id}] Annotated image generated")
            except Exception as e:
                logging.error(f"[{request_id}] Annotation failed: {e}", exc_info=True)
                result["annotation_error"] = str(e)
        
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")
        
        if result["success"]:
            if result.get("found"):
                logging.info(f"[{request_id}] Found {result.get('count', 0)} face(s)")
            else:
                logging.info(f"[{request_id}] No faces found")
            return jsonify(result), 200
        else:
            logging.error(f"[{request_id}] Recognition failed: {result.get('error')}")
            return jsonify(result), 500
            
    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@recognition.route('/verify', methods=['POST'])
def verify_face():
    """
    Verify a claimed identity (1:1).

    Expects:
        - multipart/form-data with:
            - 'image': image file
            - 'name': claimed identity
            - 'model' (optional): 'arcface' or 'mobilefacenet'
            - 'threshold' (optional)
    """
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Face verification request")

    if not face_recognition_service.is_available():
        return jsonify({"success": False, "error": "Service not available"}), 503

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400

    model_type = request.form.get('model', None)
    current_model = model_type if model_type else face_recognition_service.get_current_model()
    default_threshold = 0.60 if current_model == 'mobilefacenet' else 0.4
    threshold = float(request.form.get('threshold', default_threshold))

    logging.info(f"[{request_id}] Model: {current_model}, Threshold: {threshold}, Name: {name}")

    try:
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        result = face_recognition_service.verify_face(
            filepath,
            name,
            threshold=threshold,
            model_type=model_type
        )

        if result.get("success") and result.get("found") and result.get("faces"):
            try:
                import base64
                annotated_faces = []
                for face in result["faces"]:
                    confidence = face.get("similarity")
                    if confidence is None and "distance" in face:
                        confidence = max(0.0, 1.0 - (face["distance"] / 2.0))
                    annotated_faces.append({
                        "name": name if face.get("match") else "Unknown",
                        "confidence": confidence if confidence is not None else 0.0,
                        "bbox": face.get("bbox")
                    })

                annotated_bytes = face_recognition_service.draw_annotated_image(
                    filepath,
                    annotated_faces
                )
                if annotated_bytes:
                    result["annotated_image"] = base64.b64encode(annotated_bytes).decode('utf-8')
                    logging.info(f"[{request_id}] Annotated image generated")
            except Exception as e:
                logging.error(f"[{request_id}] Annotation failed: {e}", exc_info=True)
                result["annotation_error"] = str(e)

        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")

        if result.get("success"):
            return jsonify(result), 200
        else:
            status_code = result.get("status_code", 500)
            return jsonify(result), status_code

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@recognition.route('/detect', methods=['POST'])
def detect_only():
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Face detection request")

    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400

    file = request.files['image']

    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400

    detector_type = request.form.get('detector', 'mtcnn').strip().lower()
    if detector_type not in ['mtcnn', 'retinaface']:
        return jsonify({"success": False, "error": "Invalid detector. Use 'mtcnn' or 'retinaface'."}), 400
    if not is_detector_available(detector_type):
        return jsonify({
            "success": False,
            "error": f"{detector_type} detector is not available"
        }), 503

    try:
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        if img is None:
            return jsonify({"success": False, "error": "Could not read image"}), 400

        detected_img, count = detect_faces(img, detector=detector_type)
        annotated_base64 = image_to_base64(detected_img)

        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")

        return jsonify({
            "success": True,
            "count": count,
            "detector": detector_type,
            "annotated_image": annotated_base64
        }), 200

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@recognition.route('/video', methods=['POST'])
def process_video():
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Video processing request")
    processing_start = time.perf_counter()

    if 'video' not in request.files:
        return jsonify({"success": False, "error": "No video file provided"}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    enhancement = request.form.get('enhancement', 'none').lower()
    pipeline = request.form.get('pipeline', 'recognize').lower()
    detector_type = request.form.get('detector', 'mtcnn').strip().lower()
    darken_level_raw = request.form.get('darken_level', '50')
    model_type = request.form.get('model', None)
    threshold = request.form.get('threshold', None)

    if enhancement not in ['none', 'classical', 'deep']:
        return jsonify({"success": False, "error": "Invalid enhancement type"}), 400
    if pipeline not in ['detect', 'recognize', 'darken_only']:
        return jsonify({"success": False, "error": "Invalid pipeline mode"}), 400
    if detector_type not in ['mtcnn', 'retinaface']:
        return jsonify({"success": False, "error": "Invalid detector. Use 'mtcnn' or 'retinaface'."}), 400
    if pipeline == 'detect' and not is_detector_available(detector_type):
        return jsonify({
            "success": False,
            "error": f"{detector_type} detector is not available"
        }), 503
    try:
        darken_level = int(darken_level_raw)
    except ValueError:
        return jsonify({"success": False, "error": "Invalid darken level"}), 400
    if darken_level not in [50, 80]:
        return jsonify({"success": False, "error": "Darken level must be 50 or 80"}), 400

    try:
        filename = get_secure_filename(file.filename, request_id)
        input_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(input_path)

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return jsonify({"success": False, "error": "Could not read video"}), 400

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0:
            fps = 30.0

        output_name = (
            f"darkened_{darken_level}_{request_id}.mp4"
            if pipeline == 'darken_only'
            else f"annotated_{request_id}.mp4"
        )
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        if not writer.isOpened():
            cap.release()
            return jsonify({"success": False, "error": "Could not write output video"}), 500

        tmp_frame_path = os.path.join(UPLOAD_FOLDER, f"{request_id}_frame.jpg")
        current_model = model_type if model_type else face_recognition_service.get_current_model()
        default_threshold = 0.70 if current_model == 'mobilefacenet' else 0.4
        resolved_threshold = float(threshold) if threshold is not None else default_threshold
        frame_count = 0
        deep_fallback_logged = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if pipeline == 'darken_only':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                darkened_rgb = darken_image(frame_rgb, darken_level)
                darkened_bgr = cv2.cvtColor(darkened_rgb, cv2.COLOR_RGB2BGR)
                writer.write(darkened_bgr)
                frame_count += 1
                continue

            processed_frame = frame
            if enhancement == 'classical':
                processed_frame = enhance_classical(processed_frame)
            elif enhancement == 'deep':
                enhanced = enhance_deep(processed_frame)
                if enhanced is None:
                    # Keep pipeline running when deep enhancement is unavailable/fails.
                    if not deep_fallback_logged:
                        logging.warning(
                            f"[{request_id}] Deep enhancement unavailable; continuing without enhancement"
                        )
                        deep_fallback_logged = True
                else:
                    processed_frame = enhanced

            if pipeline == 'detect':
                annotated_frame, _ = detect_faces(processed_frame, detector=detector_type)
                writer.write(annotated_frame)
            else:
                cv2.imwrite(tmp_frame_path, processed_frame)
                result = face_recognition_service.recognize_face(
                    tmp_frame_path,
                    threshold=resolved_threshold,
                    model_type=model_type
                )
                if result.get("success") and result.get("found") and result.get("faces"):
                    annotated_bytes = face_recognition_service.draw_annotated_image(
                        tmp_frame_path,
                        result["faces"]
                    )
                    annotated_frame = cv2.imdecode(
                        np.frombuffer(annotated_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    writer.write(annotated_frame)
                else:
                    writer.write(processed_frame)

            frame_count += 1

        cap.release()
        writer.release()

        try:
            os.remove(input_path)
        except Exception as e:
            logging.warning(f"Failed to remove temp video: {e}")
        try:
            if os.path.exists(tmp_frame_path):
                os.remove(tmp_frame_path)
        except Exception as e:
            logging.warning(f"Failed to remove temp frame: {e}")

        processing_time_sec = time.perf_counter() - processing_start
        processing_speed_fps = (
            frame_count / processing_time_sec if processing_time_sec > 0 else 0.0
        )
        logging.info(
            f"[{request_id}] Video processing complete: {frame_count} frames in {processing_time_sec:.3f}s"
        )

        response = send_file(
            output_path,
            as_attachment=True,
            download_name=output_name,
            mimetype="video/mp4"
        )
        response.headers["X-Processing-Time-Ms"] = str(int(processing_time_sec * 1000))
        response.headers["X-Processing-Time-Sec"] = f"{processing_time_sec:.3f}"
        response.headers["X-Frame-Count"] = str(frame_count)
        response.headers["X-Processing-Speed-Fps"] = f"{processing_speed_fps:.3f}"
        response.headers["X-Source-Fps"] = f"{fps:.3f}"
        if pipeline == 'detect':
            response.headers["X-Detector"] = detector_type
        return response

    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@recognition.route('/list', methods=['GET'])
def list_faces():
    model_type = request.args.get('model', None)
    if model_type:
        original_model = face_recognition_service.model_type
        if not face_recognition_service.set_model(model_type):
            return jsonify({"success": False, "error": f"Model {model_type} not available"}), 400
        result = face_recognition_service.list_enrolled_faces()
        face_recognition_service.set_model(original_model)
    else:
        result = face_recognition_service.list_enrolled_faces()
    return jsonify(result), 200


@recognition.route('/remove/<name>', methods=['DELETE'])
def remove_face(name):
    logging.info(f"Removing face: {name}")
    model_type = request.args.get('model', None)
    
    if model_type:
        original_model = face_recognition_service.model_type
        if not face_recognition_service.set_model(model_type):
            return jsonify({"success": False, "error": f"Model {model_type} not available"}), 400
        result = face_recognition_service.remove_face(name)
        face_recognition_service.set_model(original_model)
    else:
        result = face_recognition_service.remove_face(name)
    
    return jsonify(result), 200 if result["success"] else 404


@recognition.route('/clear', methods=['POST'])
def clear_database():
    logging.warning("Clearing face database")
    model_type = request.args.get('model', None)
    
    if model_type:
        original_model = face_recognition_service.model_type
        if not face_recognition_service.set_model(model_type):
            return jsonify({"success": False, "error": f"Model {model_type} not available"}), 400
        result = face_recognition_service.clear_database()
        face_recognition_service.set_model(original_model)
    else:
        result = face_recognition_service.clear_database()
    
    return jsonify(result), 200


@recognition.route('/models', methods=['GET'])
def get_models():
    available_models = face_recognition_service.get_available_models()
    current_model = face_recognition_service.get_current_model()
    
    return jsonify({
        "success": True,
        "current_model": current_model,
        "available_models": available_models,
        "models": {
            "arcface": {
                "name": "ArcFace",
                "description": "High accuracy production model",
                "available": available_models.get("arcface", False)
            },
            "mobilefacenet": {
                "name": "MobileFaceNet",
                "description": "Lightweight mobile-optimized model",
                "available": available_models.get("mobilefacenet", False)
            }
        }
    }), 200


@recognition.route('/models/switch', methods=['POST'])
def switch_model():
    data = request.get_json()
    
    if not data or 'model' not in data:
        return jsonify({"success": False, "error": "Model type is required"}), 400
    
    model_type = data['model']
    
    if model_type not in ['arcface', 'mobilefacenet']:
        return jsonify({"success": False, "error": "Invalid model type"}), 400
    
    success = face_recognition_service.set_model(model_type)
    
    if success:
        logging.info(f"Switched to {model_type}")
        return jsonify({
            "success": True,
            "message": f"Switched to {model_type}",
            "current_model": face_recognition_service.get_current_model()
        }), 200
    else:
        return jsonify({"success": False, "error": f"Model {model_type} not available"}), 503


@recognition.route('/compare', methods=['POST'])
def compare_models():
    request_id = str(uuid.uuid4())[:8]
    logging.info(f"[{request_id}] Model comparison request")
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"success": False, "error": "Invalid file"}), 400
    
    threshold = float(request.form.get('threshold', 0.4))
    
    try:
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logging.info(f"[{request_id}] Comparing models")
        
        results = {}
        available_models = face_recognition_service.get_available_models()
        
        for model_name, is_available in available_models.items():
            if is_available:
                result = face_recognition_service.recognize_face(filepath, threshold=threshold, model_type=model_name)
                results[model_name] = result
        
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")
        
        return jsonify({"success": True, "results": results, "comparison_id": request_id}), 200
        
    except Exception as e:
        logging.error(f"[{request_id}] Error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500
