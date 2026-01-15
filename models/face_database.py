import os
import pickle
import logging
from typing import Dict, Optional, Any


class FaceDatabase:
    
    def __init__(self, database_path: str = "face_database.pkl"):
        self.database_path = database_path
        self.known_faces: Dict = {}
        
    def load(self) -> bool:
        if os.path.exists(self.database_path):
            try:
                with open(self.database_path, "rb") as f:
                    self.known_faces = pickle.load(f)
                logging.info(
                    f"Loaded {len(self.known_faces)} face(s) from {self.database_path}"
                )
                logging.info(f"Names in database: {list(self.known_faces.keys())}")
                return True
            except Exception as e:
                logging.error(f"Failed to load face database: {e}")
                return False
        else:
            logging.info(f"No existing database found at {self.database_path}")
            return False
    
    def save(self) -> bool:
        try:
            with open(self.database_path, "wb") as f:
                pickle.dump(self.known_faces, f)
            logging.info(f"Database saved to {self.database_path}")
            return True
        except Exception as e:
            logging.error(f"Failed to save face database: {e}")
            return False
    
    def add_face(self, name: str, embedding) -> bool:
        self.known_faces[name] = embedding
        logging.info(f"Added/updated face for: {name}")
        return True
    
    def remove_face(self, name: str) -> bool:
        if name in self.known_faces:
            del self.known_faces[name]
            logging.info(f"Removed face: {name}")
            return True
        return False
    
    def get_face(self, name: str) -> Optional[Any]:
        return self.known_faces.get(name)
    
    def list_all(self) -> list:
        """
        Get list of all names in database.
        
        Returns:
            List of names
        """
        return list(self.known_faces.keys())
    
    def count(self) -> int:
        """
        Get count of faces in database.
        
        Returns:
            Number of faces
        """
        return len(self.known_faces)
    
    def clear(self):
        """Clear all faces from database."""
        self.known_faces = {}
        logging.info("Database cleared")


# Global face database instance
face_database = FaceDatabase()
