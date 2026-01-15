"""
Model Manager - Handles initialization and management of ML models.
"""
import os
import sys
import logging
import torch
from mtcnn import MTCNN

from config import ZERO_DCE_BASE_PATH, ZERO_DCE_MODEL_PATH


class ModelManager:
    """Manages ML models (MTCNN and Zero-DCE)."""
    
    def __init__(self):
        self.detector = None
        self.deep_model = None
        self.device = None
        
    def initialize_all(self):
        """Initialize all models."""
        self._initialize_mtcnn()
        self._initialize_zero_dce()
        
    def _initialize_mtcnn(self):
        """Initialize MTCNN face detector."""
        try:
            self.detector = MTCNN()
            logging.info("MTCNN detector initialized")
        except Exception as e:
            logging.error(f"Failed to initialize MTCNN: {e}")
            
    def _initialize_zero_dce(self):
        """Initialize Zero-DCE enhancement model."""
        try:
            if not os.path.exists(ZERO_DCE_BASE_PATH):
                logging.warning("Zero-DCE directory not found")
                return
                
            if ZERO_DCE_BASE_PATH not in sys.path:
                sys.path.append(ZERO_DCE_BASE_PATH)
                
            from model import enhance_net_nopool
            
            if not os.path.exists(ZERO_DCE_MODEL_PATH):
                logging.warning("Zero-DCE model file not found")
                return
                
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.deep_model = enhance_net_nopool().to(self.device)
            self.deep_model.load_state_dict(
                torch.load(ZERO_DCE_MODEL_PATH, map_location=self.device)
            )
            self.deep_model.eval()
            logging.info(f"Zero-DCE model loaded on {self.device}")
            
        except Exception as e:
            logging.error(f"Failed to load Zero-DCE model: {e}")
            
    def is_mtcnn_available(self):
        """Check if MTCNN detector is available."""
        return self.detector is not None
        
    def is_zero_dce_available(self):
        """Check if Zero-DCE model is available."""
        return self.deep_model is not None and self.device is not None


# Global model manager instance
model_manager = ModelManager()
