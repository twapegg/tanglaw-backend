import cv2

from models import model_manager


def detect_faces(img):
    if not model_manager.is_mtcnn_available():
        return img, 0

    try:
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        detections = model_manager.detector.detect_faces(rgb_image)

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
