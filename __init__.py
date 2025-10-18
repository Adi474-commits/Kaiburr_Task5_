"""
Consumer Complaint Text Classification Package
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data_loader import load_data, split_data
from .preprocessing import TextPreprocessor
from .feature_engineering import FeatureExtractor
from .evaluation import ModelEvaluator
from .predict import ComplaintClassifier

__all__ = [
    'load_data',
    'split_data',
    'TextPreprocessor',
    'FeatureExtractor',
    'ModelEvaluator',
    'ComplaintClassifier'
]
