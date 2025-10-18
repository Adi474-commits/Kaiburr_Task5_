"""
Machine Learning Models Module
Implements various classification models for complaint categorization
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
from typing import Dict, Any, Optional
import logging
import joblib
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Train and manage multiple classification models
    """
    
    def __init__(self, use_class_weights: bool = True):
        """
        Initialize model trainer
        
        Args:
            use_class_weights: Whether to use class weights for imbalanced data
        """
        self.use_class_weights = use_class_weights
        self.models = {}
        self.training_times = {}
        self.model_configs = self._get_default_configs()
        
        logger.info("ModelTrainer initialized")
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Get default model configurations
        
        Returns:
            Dictionary of model configurations
        """
        return {
            'logistic_regression': {
                'C': 1.0,
                'max_iter': 1000,
                'solver': 'lbfgs',
                'multi_class': 'multinomial',
                'random_state': 42,
                'n_jobs': -1
            },
            'naive_bayes': {
                'alpha': 1.0
            },
            'svm': {
                'C': 1.0,
                'max_iter': 1000,
                'random_state': 42
            },
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'random_state': 42,
                'n_jobs': -1
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'random_state': 42,
                'n_jobs': -1,
                'eval_metric': 'mlogloss'
            }
        }
    
    def _compute_class_weights(self, y: np.ndarray) -> Dict[int, float]:
        """
        Compute class weights for imbalanced data
        
        Args:
            y: Label array
            
        Returns:
            Dictionary of class weights
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def train_logistic_regression(self, X_train: np.ndarray, y_train: np.ndarray,
                                  config: Optional[Dict] = None) -> LogisticRegression:
        """
        Train Logistic Regression model
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training Logistic Regression...")
        
        if config is None:
            config = self.model_configs['logistic_regression'].copy()
        
        if self.use_class_weights:
            config['class_weight'] = 'balanced'
        
        model = LogisticRegression(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['logistic_regression'] = model
        self.training_times['logistic_regression'] = training_time
        
        logger.info(f"Logistic Regression trained in {training_time:.2f}s")
        return model
    
    def train_naive_bayes(self, X_train: np.ndarray, y_train: np.ndarray,
                         config: Optional[Dict] = None) -> MultinomialNB:
        """
        Train Naive Bayes model
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training Naive Bayes...")
        
        if config is None:
            config = self.model_configs['naive_bayes'].copy()
        
        model = MultinomialNB(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['naive_bayes'] = model
        self.training_times['naive_bayes'] = training_time
        
        logger.info(f"Naive Bayes trained in {training_time:.2f}s")
        return model
    
    def train_svm(self, X_train: np.ndarray, y_train: np.ndarray,
                  config: Optional[Dict] = None) -> LinearSVC:
        """
        Train SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training SVM...")
        
        if config is None:
            config = self.model_configs['svm'].copy()
        
        if self.use_class_weights:
            config['class_weight'] = 'balanced'
        
        model = LinearSVC(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['svm'] = model
        self.training_times['svm'] = training_time
        
        logger.info(f"SVM trained in {training_time:.2f}s")
        return model
    
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray,
                           config: Optional[Dict] = None) -> RandomForestClassifier:
        """
        Train Random Forest model
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training Random Forest...")
        
        if config is None:
            config = self.model_configs['random_forest'].copy()
        
        if self.use_class_weights:
            config['class_weight'] = 'balanced'
        
        model = RandomForestClassifier(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.models['random_forest'] = model
        self.training_times['random_forest'] = training_time
        
        logger.info(f"Random Forest trained in {training_time:.2f}s")
        return model
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray,
                     config: Optional[Dict] = None) -> xgb.XGBClassifier:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training labels
            config: Model configuration
            
        Returns:
            Trained model
        """
        logger.info("Training XGBoost...")
        
        if config is None:
            config = self.model_configs['xgboost'].copy()
        
        if self.use_class_weights:
            class_weights = self._compute_class_weights(y_train)
            sample_weights = np.array([class_weights[label] for label in y_train])
        else:
            sample_weights = None
        
        model = xgb.XGBClassifier(**config)
        
        start_time = time.time()
        model.fit(X_train, y_train, sample_weight=sample_weights)
        training_time = time.time() - start_time
        
        self.models['xgboost'] = model
        self.training_times['xgboost'] = training_time
        
        logger.info(f"XGBoost trained in {training_time:.2f}s")
        return model
    
    def train_all(self, X_train: np.ndarray, y_train: np.ndarray,
                  models_to_train: Optional[list] = None):
        """
        Train all specified models
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_train: List of model names to train (None for all)
        """
        if models_to_train is None:
            models_to_train = ['logistic_regression', 'naive_bayes', 'svm', 
                             'random_forest', 'xgboost']
        
        logger.info(f"Training {len(models_to_train)} models...")
        
        for model_name in models_to_train:
            try:
                if model_name == 'logistic_regression':
                    self.train_logistic_regression(X_train, y_train)
                elif model_name == 'naive_bayes':
                    self.train_naive_bayes(X_train, y_train)
                elif model_name == 'svm':
                    self.train_svm(X_train, y_train)
                elif model_name == 'random_forest':
                    self.train_random_forest(X_train, y_train)
                elif model_name == 'xgboost':
                    self.train_xgboost(X_train, y_train)
                else:
                    logger.warning(f"Unknown model: {model_name}")
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
        
        logger.info("All models trained successfully")
    
    def get_model(self, model_name: str):
        """
        Get trained model by name
        
        Args:
            model_name: Name of the model
            
        Returns:
            Trained model
        """
        return self.models.get(model_name)
    
    def save_model(self, model_name: str, filepath: str):
        """
        Save a trained model to file
        
        Args:
            model_name: Name of the model
            filepath: Path to save the model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        joblib.dump(self.models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def save_all_models(self, output_dir: str):
        """
        Save all trained models
        
        Args:
            output_dir: Directory to save models
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name in self.models:
            filepath = os.path.join(output_dir, f"{model_name}.pkl")
            self.save_model(model_name, filepath)
    
    @staticmethod
    def load_model(filepath: str):
        """
        Load a model from file
        
        Args:
            filepath: Path to the model file
            
        Returns:
            Loaded model
        """
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    X_train, y_train = make_classification(
        n_samples=1000, n_features=100, n_classes=4, 
        n_informative=50, random_state=42
    )
    
    trainer = ModelTrainer(use_class_weights=True)
    trainer.train_all(X_train, y_train)
    
    print(f"Trained models: {list(trainer.models.keys())}")
    print(f"Training times: {trainer.training_times}")
