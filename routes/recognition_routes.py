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
    
    if 'image' not in request.files:
        return jsonify({"success": False, "error": "No image file provided"}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({
            "success": False,
            "error": f'File type not allowed. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
        }), 400
    
    name = request.form.get('name', '').strip()
    if not name:
        return jsonify({"success": False, "error": "Name is required"}), 400
    
    model_type = request.form.get('model', None)
    
    try:
        filename = get_secure_filename(file.filename, request_id)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        logging.info(f"[{request_id}] Enrolling: {name} (model: {model_type or 'default'})")
        
        result = face_recognition_service.enroll_face(filepath, name, model_type=model_type)
        
        try:
            os.remove(filepath)
        except Exception as e:
            logging.warning(f"Failed to remove temp file: {e}")
        
        if result["success"]:
            logging.info(f"[{request_id}] Successfully enrolled: {name}")
            return jsonify(result), 200
        else:
            logging.error(f"[{request_id}] Enrollment failed: {result.get('error')}")
            return jsonify(result), 500
            
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
