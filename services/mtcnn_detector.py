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


def detect_faces(img, detector="mtcnn", include_confidences=False):
    detector_name = (detector or "mtcnn").strip().lower()

    if detector_name == "retinaface":
        retinaface = _get_retinaface_detector()
        if retinaface is None:
            error_details = (
                _retinaface_init_error if _retinaface_init_error else "not initialized"
            )
            raise RuntimeError(f"RetinaFace detector unavailable: {error_details}")
        return retinaface.detect_faces_with_visualization(
            img,
            include_confidences=include_confidences,
        )

    if detector_name != "mtcnn":
        raise ValueError("Invalid detector. Use 'mtcnn' or 'retinaface'.")

    if not model_manager.is_mtcnn_available():
        if include_confidences:
            return img, 0, []
        return img, 0

    try:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = model_manager.detector.detect_faces(rgb_image)

        if not detections:
            if include_confidences:
                return img, 0, []
            return img, 0

        result_img = img.copy()

        valid_count = 0
        confidences = []
        label_font = cv2.FONT_HERSHEY_SIMPLEX
        label_font_scale = max(0.75, min(1.15, min(result_img.shape[:2]) / 600.0))
        label_text_thickness = 2
        label_pad_x = 14
        label_pad_y = 10
        label_bg_color = (210, 245, 210)       # light green background (BGR)
        label_border_color = (130, 105, 65)    # muted border that fits the palette
        label_text_color = (72, 45, 18)        # dark, easy-to-read text color
        box_color = (48, 190, 60)

        for detection in detections:
            box = detection.get("box")
            keypoints = detection.get("keypoints", {})
            confidence = float(detection.get("confidence", 0.0))

            if not box or len(box) != 4:
                continue

            x, y, width, height = [int(v) for v in box]

            cv2.rectangle(result_img, (x, y), (x + width, y + height), box_color, 4)
            label = f"Conf {confidence * 100:.1f}%"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, label_font, label_font_scale, label_text_thickness
            )
            text_x = max(
                label_pad_x,
                min(x + 2, result_img.shape[1] - text_width - label_pad_x - 1),
            )
            candidate_baseline = y - 8
            min_baseline = text_height + baseline + label_pad_y + 1
            if candidate_baseline < min_baseline:
                candidate_baseline = y + text_height + baseline + label_pad_y + 2
            text_baseline = min(
                result_img.shape[0] - baseline - label_pad_y - 1,
                max(min_baseline, candidate_baseline),
            )
            bg_x1 = max(0, text_x - label_pad_x)
            bg_y1 = max(0, text_baseline - text_height - baseline - label_pad_y)
            bg_x2 = min(result_img.shape[1] - 1, text_x + text_width + label_pad_x)
            bg_y2 = min(result_img.shape[0] - 1, text_baseline + baseline + label_pad_y)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (bg_x2, bg_y2), label_bg_color, -1)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (bg_x2, bg_y2), label_border_color, 2)
            accent_x2 = min(bg_x2, bg_x1 + 5)
            cv2.rectangle(result_img, (bg_x1, bg_y1), (accent_x2, bg_y2), box_color, -1)
            cv2.putText(
                result_img,
                label,
                (
                    text_x,
                    max(text_height, min(text_baseline, result_img.shape[0] - baseline - 1)),
                ),
                label_font,
                label_font_scale,
                label_text_color,
                label_text_thickness,
                cv2.LINE_AA,
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
            confidences.append(max(0.0, min(1.0, confidence)))
            valid_count += 1

        if include_confidences:
            return result_img, valid_count, confidences
        return result_img, valid_count
        
    except Exception as e:
        logging.error(f"Face detection failed: {e}")
        if include_confidences:
            return img, 0, []
        return img, 0
