import cv2
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple
from insightface.app import FaceAnalysis

CONDITION_DIRS = {
    "50_enhanced_classic",
    "50_enhanced_deep",
    "80_enhanced_classic",
    "80_enhanced_deep",
    "50_darkened",
    "80_darkened",
}
BASELINE_CONDITION = "baseline"


class ArcFaceRecognizer:
    
    def __init__(self, det_size=(640, 640), database_path='database/arcface'):
        self.known_faces = {}
        self.det_size = det_size
        self.database_path = Path(database_path)
        self.database_path.mkdir(parents=True, exist_ok=True)
        
        logging.info("Initializing InsightFace ArcFace (buffalo_l)...")
        import warnings
        from contextlib import redirect_stdout, redirect_stderr
        import io
        with warnings.catch_warnings(), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=det_size)
        logging.info("ArcFace initialized successfully")
        
        self._load_database()
        logging.info(f"Loaded database: {len(self.known_faces)} people")
    
    def add_face(self, image_path: str, name: str, condition: str = None) -> bool:
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
        logging.info(
            f"Added {name} [{resolved_condition}] (total: {len(self.known_faces[name][resolved_condition])} embeddings)"
        )
        
        self._save_database()
        return True
    
    def recognize_face(self, image_path: str, threshold: float = 0.4) -> List[Tuple[str, float, List[int]]]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            logging.info("No faces detected")
            return []
        
        results = []
        for face in faces:
            embedding = face.embedding
            bbox = face.bbox.astype(int).tolist()
            
            best_match = "Unknown"
            best_distance = float('inf')
            
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            embedding = self._normalize_embedding(embedding)
            
            if len(self.known_faces) > 0:
                for name, entry in self.known_faces.items():
                    condition_groups = self._get_condition_groups(entry)
                    for condition, embeddings in condition_groups.items():
                        template = self._compute_template(embeddings)
                        if template is None:
                            continue
                        try:
                            distance = self._cosine_distance(embedding, template)
                            if distance < best_distance:
                                best_distance = distance
                                best_match = name
                        except Exception as e:
                            logging.error(f"Error comparing embeddings for {name} [{condition}]: {e}")
                            continue
            
            if best_distance > threshold:
                best_match = "Unknown"
                confidence = 0.0
            else:
                confidence = max(0.0, 1.0 - (best_distance / 2.0))
            
            results.append((best_match, confidence, bbox))
            logging.info(f"Face: {best_match} ({confidence:.2f})")
        
        return results

    def verify_face(self, image_path: str, name: str, threshold: float = 0.4):
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
                "metric": "cosine_distance",
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
        best_distance = float("inf")

        for face in faces:
            embedding = self._normalize_embedding(face.embedding)
            bbox = face.bbox.astype(int).tolist()

            face_best_distance = float("inf")
            best_condition = None
            for condition, template in templates:
                distance = self._cosine_distance(embedding, template)
                if distance < face_best_distance:
                    face_best_distance = distance
                    best_condition = condition

            similarity = max(0.0, 1.0 - (face_best_distance / 2.0))
            is_match = face_best_distance <= threshold

            face_result = {
                "bbox": bbox,
                "distance": float(face_best_distance),
                "similarity": float(similarity),
                "match": is_match,
                "condition": best_condition
            }
            results.append(face_result)

            if face_best_distance < best_distance:
                best_distance = face_best_distance
                best_face = face_result

        return {
            "success": True,
            "found": True,
            "verified": bool(best_face and best_face["match"]),
            "metric": "cosine_distance",
            "threshold": threshold,
            "count": len(results),
            "faces": results,
            "best": best_face
        }
    
    def _cosine_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        if not isinstance(emb1, np.ndarray):
            emb1 = np.array(emb1, dtype=np.float32)
        if not isinstance(emb2, np.ndarray):
            emb2 = np.array(emb2, dtype=np.float32)
        
        emb1 = emb1.flatten()
        emb2 = emb2.flatten()
        
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 2.0
        
        emb1 = emb1 / norm1
        emb2 = emb2 / norm2
        similarity = float(np.dot(emb1, emb2))
        return 1.0 - similarity

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
    
    def get_face_count(self) -> int:
        return len(self.known_faces)
    
    def remove_face(self, name: str) -> bool:
        if name in self.known_faces:
            del self.known_faces[name]
            logging.info(f"Removed face: {name}")
            self._save_database()
            return True
        return False
    
    def clear_all(self):
        self.known_faces = {}
        logging.info("Cleared all faces")
        self._save_database()
    
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
    
    def _save_database(self):
        db_file = self.database_path / 'face_database.pkl'
        with open(db_file, 'wb') as f:
            pickle.dump(self.known_faces, f)
        logging.debug(f"Database saved: {len(self.known_faces)} people")
