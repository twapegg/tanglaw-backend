"""MobileFaceNet Recognizer using InsightFace pretrained models"""

import cv2
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

CONDITION_DIRS = {
    "50_enhanced_classic",
    "50_enhanced_deep",
    "80_enhanced_classic",
    "80_enhanced_deep",
    "50_darkened",
    "80_darkened",
}
BASELINE_CONDITION = "baseline"

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logging.warning("InsightFace not available")


class MobileFaceNetRecognizer:
    
    def __init__(self, det_size=(640, 640), database_path='database/mobilefacenet'):
        if not INSIGHTFACE_AVAILABLE:
            raise ImportError("InsightFace required: pip install insightface")
        
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        self.known_faces = {}
        
        try:
            logging.info("Initializing MobileFaceNet (antelopev2)...")
            import warnings
            from contextlib import redirect_stdout, redirect_stderr
            import io
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                self.app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=det_size)
            logging.info("MobileFaceNet initialized with pretrained weights")
        except Exception as e:
            logging.error(f"MobileFaceNet init failed: {str(e) or 'Unknown error'}, using buffalo_l fallback")
            import warnings
            from contextlib import redirect_stdout, redirect_stderr
            import io
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
                self.app.prepare(ctx_id=0, det_size=det_size)
        
        self._load_database()
        logging.info(f"Loaded {len(self.known_faces)} faces from database")
    
    def add_face(self, image_path: str, name: str, condition: str = None) -> bool:
        logging.info(f"Adding face: {name}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected")
        
        if len(faces) > 1:
            raise ValueError("Multiple faces detected; please provide an image with exactly one face.")
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = self._normalize_embedding(face.embedding)
        resolved_condition = condition or self._infer_condition(image_path)
        self._ensure_entry(name)
        if resolved_condition not in self.known_faces[name]:
            self.known_faces[name][resolved_condition] = []
        self.known_faces[name][resolved_condition].append(embedding.tolist())
        self._save_database()
        logging.info(
            f"Added {name} [{resolved_condition}] (total: {len(self.known_faces[name][resolved_condition])} embeddings)"
        )
        return True
    
    def recognize_face(self, image_path: str, threshold: float = 0.6) -> List[Tuple[str, float, List[int]]]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        faces = self.app.get(img)
        logging.info(f"Detected {len(faces)} faces")
        
        if len(faces) == 0:
            return []
        
        results = []
        for i, face in enumerate(faces):
            embedding = face.embedding
            bbox = face.bbox.astype(int).tolist()
            
            best_match = "Unknown"
            best_similarity = 0.0
            
            embedding = self._normalize_embedding(embedding)
            for name, entry in self.known_faces.items():
                condition_groups = self._get_condition_groups(entry)
                for condition, embeddings in condition_groups.items():
                    template = self._compute_template(embeddings)
                    if template is None:
                        continue
                    similarity = self._compute_similarity(embedding, template)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
            
            if best_similarity < threshold:
                best_match = "Unknown"
                confidence = 0.0
            else:
                confidence = float(best_similarity)
                logging.info(f"Face {i}: {best_match} ({confidence:.3f})")
            
            results.append((best_match, confidence, bbox))
        
        return results

    def verify_face(self, image_path: str, name: str, threshold: float = 0.6):
        if name not in self.known_faces:
            return {
                "success": False,
                "error": f"Face '{name}' not found in database",
                "status_code": 404
            }

        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")

        faces = self.app.get(img)
        if len(faces) == 0:
            return {
                "success": True,
                "found": False,
                "verified": False,
                "metric": "cosine_similarity",
                "threshold": threshold,
                "count": 0,
                "faces": [],
                "best": None
            }
        if len(faces) > 1:
            return {
                "success": False,
                "error": "Multiple faces detected; please provide an image with exactly one face.",
                "status_code": 400
            }

        condition_groups = self._get_condition_groups(self.known_faces[name])
        templates = []
        for condition, embeddings in condition_groups.items():
            template = self._compute_template(embeddings)
            if template is not None:
                templates.append((condition, template))

        if not templates:
            return {
                "success": False,
                "error": f"No embeddings found for '{name}'",
                "status_code": 404
            }

        results = []
        best_face = None
        best_similarity = -1.0

        for face in faces:
            embedding = self._normalize_embedding(face.embedding)
            bbox = face.bbox.astype(int).tolist()

            face_best_similarity = -1.0
            best_condition = None
            for condition, template in templates:
                similarity = self._compute_similarity(embedding, template)
                if similarity > face_best_similarity:
                    face_best_similarity = similarity
                    best_condition = condition

            is_match = face_best_similarity >= threshold

            face_result = {
                "bbox": bbox,
                "similarity": float(face_best_similarity),
                "match": is_match,
                "condition": best_condition
            }
            results.append(face_result)

            if face_best_similarity > best_similarity:
                best_similarity = face_best_similarity
                best_face = face_result

        return {
            "success": True,
            "found": True,
            "verified": bool(best_face and best_face["match"]),
            "metric": "cosine_similarity",
            "threshold": threshold,
            "count": len(results),
            "faces": results,
            "best": best_face
        }
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        emb1_norm = self._normalize_embedding(embedding1)
        emb2_norm = self._normalize_embedding(embedding2)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(np.clip(similarity, 0.0, 1.0))

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        if not isinstance(embedding, np.ndarray):
            embedding = np.array(embedding, dtype=np.float32)
        embedding = embedding.flatten()
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm

    def _compute_template(self, embeddings: list) -> np.ndarray:
        if not embeddings:
            return None
        normalized = [self._normalize_embedding(e) for e in embeddings]
        mean = np.mean(normalized, axis=0)
        return self._normalize_embedding(mean)

    def _infer_condition(self, image_path: str) -> str:
        try:
            parts = Path(image_path).parts
        except Exception:
            parts = []
        for part in parts:
            if part in CONDITION_DIRS:
                return part
        return BASELINE_CONDITION

    def _ensure_entry(self, name: str) -> None:
        if name not in self.known_faces:
            self.known_faces[name] = {}
            return
        entry = self.known_faces[name]
        if isinstance(entry, dict):
            return
        self.known_faces[name] = {BASELINE_CONDITION: self._to_list(entry)}

    def _get_condition_groups(self, entry):
        if isinstance(entry, dict):
            return entry
        return {BASELINE_CONDITION: self._to_list(entry)}

    def _to_list(self, entry):
        if isinstance(entry, np.ndarray):
            return [entry.tolist()]
        if isinstance(entry, list):
            return entry
        return []
    
    def _save_database(self):
        db_file = self.database_path / 'face_database.pkl'
        with open(db_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
        logging.debug(f"Database saved: {len(self.known_faces)} people")
    
    def _load_database(self):
        db_file = self.database_path / 'face_database.pkl'
        if db_file.exists():
            try:
                with open(db_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                logging.debug(f"Loaded database: {len(self.known_faces)} people")
            except Exception as e:
                logging.error(f"Error loading database: {e}")
                self.known_faces = {}
        else:
            logging.debug("No existing database, starting fresh")
            self.known_faces = {}
    
    def remove_face(self, name: str) -> bool:
        if name in self.known_faces:
            del self.known_faces[name]
            logging.info(f"Removed face: {name}")
            self._save_database()
            return True
        return False
    
    def clear_database(self):
        self.known_faces = {}
        self._save_database()
        logging.info("Database cleared")
