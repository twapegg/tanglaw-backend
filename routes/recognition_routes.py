import os
import uuid
import logging
from flask import Blueprint, request, jsonify

from config import UPLOAD_FOLDER, ALLOWED_EXTENSIONS
from services.face_recognition_service import face_recognition_service
from utils import allowed_file, get_secure_filename


recognition = Blueprint('recognition', __name__, url_prefix='/recognition')


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
