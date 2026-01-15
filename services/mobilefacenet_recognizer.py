"""MobileFaceNet Recognizer using InsightFace pretrained models"""

import cv2
import pickle
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

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
    
    def add_face(self, image_path: str, name: str) -> bool:
        logging.info(f"Adding face: {name}")
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        
        faces = self.app.get(img)
        if len(faces) == 0:
            raise ValueError("No face detected")
        
        if len(faces) > 1:
            logging.warning("Multiple faces detected, using largest")
        
        face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
        embedding = face.embedding
        
        if name not in self.known_faces:
            self.known_faces[name] = []
        
        self.known_faces[name].append(embedding)
        self._save_database()
        logging.info(f"Added {name} (total: {len(self.known_faces[name])} embeddings)")
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
            
            for name, embeddings in self.known_faces.items():
                if not isinstance(embeddings, list):
                    embeddings = [embeddings]
                
                for known_embedding in embeddings:
                    similarity = self._compute_similarity(embedding, known_embedding)
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
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        emb1_norm = embedding1 / (np.linalg.norm(embedding1) + 1e-8)
        emb2_norm = embedding2 / (np.linalg.norm(embedding2) + 1e-8)
        similarity = np.dot(emb1_norm, emb2_norm)
        return float(np.clip(similarity, 0.0, 1.0))
    
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
