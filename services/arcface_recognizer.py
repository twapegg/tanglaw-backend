import cv2
import numpy as np
import logging
import pickle
from pathlib import Path
from typing import List, Tuple
from insightface.app import FaceAnalysis


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
    
    def add_face(self, image_path: str, name: str) -> bool:
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
        
        if isinstance(self.known_faces[name], np.ndarray):
            self.known_faces[name] = self.known_faces[name].tolist() if self.known_faces[name].size > 0 else []
        
        self.known_faces[name].append(embedding.tolist() if isinstance(embedding, np.ndarray) else embedding)
        logging.info(f"Added {name} (total: {len(self.known_faces[name])} embeddings)")
        
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
            best_similarity = float('inf')
            
            if not isinstance(embedding, np.ndarray):
                embedding = np.array(embedding)
            
            if len(self.known_faces) > 0:
                for name, embeddings in self.known_faces.items():
                    if not isinstance(embeddings, list):
                        embeddings = [embeddings]
                    
                    for known_embedding in embeddings:
                        if not isinstance(known_embedding, np.ndarray):
                            known_embedding = np.array(known_embedding, dtype=np.float32)
                        
                        try:
                            similarity = self._cosine_distance(embedding, known_embedding)
                            if similarity < best_similarity:
                                best_similarity = similarity
                                best_match = name
                        except Exception as e:
                            logging.error(f"Error comparing embeddings for {name}: {e}")
                            continue
            
            if best_similarity > threshold:
                best_match = "Unknown"
                confidence = 0.0
            else:
                confidence = max(0.0, 1.0 - (best_similarity / 2.0))
            
            results.append((best_match, confidence, bbox))
            logging.info(f"Face: {best_match} ({confidence:.2f})")
        
        return results
    
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
