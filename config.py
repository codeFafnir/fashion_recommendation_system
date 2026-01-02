"""
Configuration module for Fashion Recommendation System
Shared configuration used across all stages
"""

from pathlib import Path
from datetime import timedelta
import numpy as np

# Optional torch import for neural network stages
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class Config:
    """Base configuration for the recommendation system"""
    
    # Paths
    DATA_PATH = Path('/Users/raghu/Desktop/Quarter_1/CSE_258R/assignment2/h-and-m-personalized-fashion-recommendations')
    OUTPUT_PATH = Path('/Users/raghu/Desktop/Quarter_1/CSE_258R/assignment2/fashion_recommender_candidate_generation_2')
    MODEL_PATH = OUTPUT_PATH / 'models'
    
    # Create directories
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)
    MODEL_PATH.mkdir(exist_ok=True, parents=True)
    
    # Temporal settings
    TOTAL_WEEKS = 24
    N_TRAIN_WEEKS = 11
    N_VAL_WEEKS = 1
    
    # Sampling settings
    TARGET_USERS = 50000
    MIN_USER_PURCHASES = 1
    MIN_ITEM_PURCHASES = 5
    INCLUDE_COLD_START = True
    COLD_START_RATIO = 0.15
    COLD_START_MAX_PURCHASES = 1
    STRATIFY_BY_ACTIVITY = True
    ACTIVITY_BINS = [1, 3, 8, 20, 50, 10000]
    ACTIVITY_LABELS = ['low', 'medium', 'high', 'very_high', 'extreme']
    
    # Candidate generation settings
    N_REPURCHASE_CANDIDATES = 12
    N_POPULARITY_CANDIDATES = 25
    POPULARITY_WINDOW_WEEKS = 2
    N_COPURCHASE_CANDIDATES = 15
    MIN_ITEM_SUPPORT = 3
    MAX_ITEM_NEIGHBORS = 50
    N_USERKNN_CANDIDATES = 10
    N_SIMILAR_USERS = 30
    N_CATEGORY_CANDIDATES = 10
    N_TOP_CATEGORY_ITEMS = 20
    
    # Feature extraction settings
    CHUNK_SIZE = 50000
    RECENT_DAYS = 7
    MEDIUM_DAYS = 21
    
    # Random state
    RANDOM_STATE = 42


class LightGBMConfig:
    """Configuration for LightGBM training"""
    
    MODEL_PATH = Config.MODEL_PATH
    
    # Training settings
    N_ESTIMATORS = 500
    EARLY_STOPPING_ROUNDS = 100
    VERBOSE_EVAL = 50
    RANDOM_STATE = 42


class NeuralTowerConfig:
    """Configuration for Two-Tower Neural Network training"""
    
    DATA_PATH = Config.OUTPUT_PATH
    MODEL_PATH = Config.MODEL_PATH
    
    # Device configuration (set at runtime when torch is available)
    DEVICE = None
    
    @classmethod
    def get_device(cls):
        """Get the appropriate device for PyTorch"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for neural network training")
        
        if cls.DEVICE is None:
            if torch.backends.mps.is_available():
                cls.DEVICE = torch.device('mps')
            elif torch.cuda.is_available():
                cls.DEVICE = torch.device('cuda')
            else:
                cls.DEVICE = torch.device('cpu')
        return cls.DEVICE
    
    # Training settings
    BATCH_SIZE = 4096
    N_EPOCHS = 30
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 2e-4
    EARLY_STOPPING_PATIENCE = 5
    VALIDATION_FREQ = 1
    
    # Two-Tower Model architecture
    # User Tower: processes user-level features
    USER_EMBEDDING_DIM = 128
    # Item Tower: processes item features + image embeddings combined
    ITEM_EMBEDDING_DIM = 128
    # Fusion layer dimensions
    FUSION_HIDDEN_DIMS = [256, 128, 64]
    DROPOUT_RATE = 0.3
    
    # Feature prefixes for tower assignment
    # User features go to User Tower
    USER_FEATURE_PREFIXES = [
        'n_', 'avg_', 'std_', 'min_', 'max_', 'days_', 'purchase_',
        'exploration_', 'age', 'FN', 'Active', 'unique_'
    ]
    # Item features go to Item Tower (includes image embeddings)
    ITEM_FEATURE_PREFIXES = [
        'product_', 'graphical_', 'colour_', 'perceived_', 'department_',
        'index_', 'section_', 'garment_', 'popularity_', 'sales_', 'buyers_'
    ]
    # Image features are treated as item features in two-tower architecture
    IMAGE_FEATURE_PREFIXES = ['image_emb_']
    
    RANDOM_STATE = 42


class EDAConfig:
    """Configuration for EDA and visualization"""
    
    DATA_PATH = Config.DATA_PATH
    PROCESSED_PATH = Config.OUTPUT_PATH
    MODEL_PATH = Config.MODEL_PATH
    OUTPUT_DIR = Path('/Users/raghu/Desktop/Quarter_1/CSE_258R/assignment2/eda_plots')
    
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class EvaluationConfig:
    """Configuration for evaluation"""
    
    MODEL_PATH = Config.MODEL_PATH
    K_VALUES = [1, 3, 5, 10, 12]


# Default config instance
config = Config()

