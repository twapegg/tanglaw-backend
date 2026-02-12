import logging
import cv2

from models import model_manager

_retinaface_detector = None
_retinaface_init_attempted = False
_retinaface_init_error = None


def _get_retinaface_detector():
    global _retinaface_detector, _retinaface_init_attempted, _retinaface_init_error

    if _retinaface_detector is not None:
        return _retinaface_detector

    if _retinaface_init_attempted:
        return None

    _retinaface_init_attempted = True
    try:
        from .retinaface_detector import RetinaFaceDetector

        _retinaface_detector = RetinaFaceDetector()
        return _retinaface_detector
    except Exception as e:
        _retinaface_init_error = str(e)
        logging.error(f"Failed to initialize RetinaFace detector: {e}")
        return None


def is_detector_available(detector="mtcnn"):
    detector_name = (detector or "mtcnn").strip().lower()
    if detector_name == "mtcnn":
        return model_manager.is_mtcnn_available()
    if detector_name == "retinaface":
        return _get_retinaface_detector() is not None
    return False


def detect_faces(img, detector="mtcnn"):
    detector_name = (detector or "mtcnn").strip().lower()

    if detector_name == "retinaface":
        retinaface = _get_retinaface_detector()
        if retinaface is None:
            error_details = (
                _retinaface_init_error if _retinaface_init_error else "not initialized"
            )
            raise RuntimeError(f"RetinaFace detector unavailable: {error_details}")
        return retinaface.detect_faces_with_visualization(img)

    if detector_name != "mtcnn":
        raise ValueError("Invalid detector. Use 'mtcnn' or 'retinaface'.")

    if not model_manager.is_mtcnn_available():
        return img, 0

    try:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = model_manager.detector.detect_faces(rgb_image)

        if not detections:
            return img, 0

        result_img = img.copy()

        valid_count = 0

        for detection in detections:
            box = detection.get("box")
            keypoints = detection.get("keypoints", {})
            confidence = float(detection.get("confidence", 0.0))

            if not box or len(box) != 4:
                continue

            x, y, width, height = [int(v) for v in box]

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

            landmarks = {
                "left_eye": (keypoints.get("left_eye"), (255, 0, 0)),
                "right_eye": (keypoints.get("right_eye"), (0, 255, 255)),
                "nose": (keypoints.get("nose"), (0, 0, 255)),
                "mouth_left": (keypoints.get("mouth_left"), (255, 255, 0)),
                "mouth_right": (keypoints.get("mouth_right"), (255, 0, 255)),
            }

            for name, (point, color) in landmarks.items():
                if point is None or len(point) != 2:
                    continue
                px, py = int(point[0]), int(point[1])
                cv2.circle(result_img, (px, py), 5, color, -1)
                cv2.putText(
                    result_img,
                    name.replace("_", " ").title(),
                    (px + 10, py),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    color,
                    1,
                )
            valid_count += 1

        return result_img, valid_count
        
    except Exception as e:
        logging.error(f"Face detection failed: {e}")
        return img, 0
