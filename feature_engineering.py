"""
Feature Engineering Module
Implements various feature extraction techniques for text classification
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from typing import Tuple, Optional
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract features from preprocessed text for classification
    """
    
    def __init__(self,
                 method: str = 'tfidf',
                 max_features: int = 5000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 min_df: int = 5,
                 max_df: float = 0.8,
                 use_svd: bool = False,
                 svd_components: int = 300):
        """
        Initialize feature extractor
        
        Args:
            method: Feature extraction method ('tfidf' or 'count')
            max_features: Maximum number of features to extract
            ngram_range: Range of n-grams to consider
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            use_svd: Whether to apply dimensionality reduction
            svd_components: Number of SVD components
        """
        self.method = method
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.use_svd = use_svd
        self.svd_components = svd_components
        
        self.vectorizer = None
        self.svd = None
        
        logger.info(f"FeatureExtractor initialized with method: {method}")
    
    def fit(self, texts: pd.Series):
        """
        Fit vectorizer on training texts
        
        Args:
            texts: Series of preprocessed texts
        """
        logger.info(f"Fitting {self.method} vectorizer...")
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df,
                sublinear_tf=True
            )
        elif self.method == 'count':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                min_df=self.min_df,
                max_df=self.max_df
            )
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit vectorizer
        X = self.vectorizer.fit_transform(texts)
        logger.info(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        logger.info(f"Feature matrix shape: {X.shape}")
        
        # Fit SVD if requested
        if self.use_svd:
            logger.info(f"Fitting SVD with {self.svd_components} components...")
            self.svd = TruncatedSVD(n_components=self.svd_components, random_state=42)
            self.svd.fit(X)
            logger.info(f"Explained variance: {self.svd.explained_variance_ratio_.sum():.3f}")
        
        return self
    
    def transform(self, texts: pd.Series) -> np.ndarray:
        """
        Transform texts to feature vectors
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit() first.")
        
        X = self.vectorizer.transform(texts)
        
        if self.use_svd and self.svd is not None:
            X = self.svd.transform(X)
        
        return X
    
    def fit_transform(self, texts: pd.Series) -> np.ndarray:
        """
        Fit vectorizer and transform texts
        
        Args:
            texts: Series of preprocessed texts
            
        Returns:
            Feature matrix
        """
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self) -> list:
        """
        Get feature names from vectorizer
        
        Returns:
            List of feature names
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        
        if hasattr(self.vectorizer, 'get_feature_names_out'):
            return self.vectorizer.get_feature_names_out()
        else:
            return self.vectorizer.get_feature_names()
    
    def get_top_features(self, n: int = 20, category: Optional[int] = None,
                        y: Optional[np.ndarray] = None,
                        X: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Get top features for each category
        
        Args:
            n: Number of top features to return
            category: Specific category to analyze (None for all)
            y: Labels array
            X: Feature matrix
            
        Returns:
            DataFrame with top features
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted.")
        
        if self.use_svd:
            logger.warning("Top features not available when using SVD")
            return pd.DataFrame()
        
        feature_names = self.get_feature_names()
        
        if y is None or X is None:
            return pd.DataFrame({'feature': feature_names[:n]})
        
        # Calculate mean TF-IDF for each category
        results = []
        
        categories = [category] if category is not None else np.unique(y)
        
        for cat in categories:
            mask = y == cat
            X_cat = X[mask]
            
            if hasattr(X_cat, 'toarray'):
                X_cat = X_cat.toarray()
            
            mean_tfidf = np.mean(X_cat, axis=0).flatten()
            top_indices = np.argsort(mean_tfidf)[::-1][:n]
            
            for idx in top_indices:
                results.append({
                    'category': cat,
                    'feature': feature_names[idx],
                    'score': mean_tfidf[idx]
                })
        
        return pd.DataFrame(results)
    
    def save(self, filepath: str):
        """
        Save feature extractor to file
        
        Args:
            filepath: Path to save the model
        """
        joblib.dump({
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'params': {
                'method': self.method,
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'min_df': self.min_df,
                'max_df': self.max_df,
                'use_svd': self.use_svd,
                'svd_components': self.svd_components
            }
        }, filepath)
        logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load feature extractor from file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            FeatureExtractor instance
        """
        data = joblib.load(filepath)
        
        extractor = cls(**data['params'])
        extractor.vectorizer = data['vectorizer']
        extractor.svd = data['svd']
        
        logger.info(f"Feature extractor loaded from {filepath}")
        return extractor


def extract_text_features(df: pd.DataFrame, 
                         text_column: str = 'processed_text') -> pd.DataFrame:
    """
    Extract additional text-based features
    
    Args:
        df: DataFrame with text
        text_column: Name of text column
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Length features
    df['text_length'] = df[text_column].str.len()
    df['word_count'] = df[text_column].str.split().str.len()
    df['avg_word_length'] = df[text_column].apply(
        lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0
    )
    
    # Unique word ratio
    df['unique_word_ratio'] = df[text_column].apply(
        lambda x: len(set(str(x).split())) / len(str(x).split()) if len(str(x).split()) > 0 else 0
    )
    
    # Capital letter ratio (before preprocessing)
    if 'text' in df.columns:
        df['capital_ratio'] = df['text'].apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
        )
    
    return df


if __name__ == "__main__":
    # Example usage
    texts = pd.Series([
        "credit report error incorrect information",
        "debt collector harassment phone calls",
        "loan approval denied unfair",
        "mortgage payment issue bank"
    ])
    
    extractor = FeatureExtractor(method='tfidf', max_features=100)
    X = extractor.fit_transform(texts)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Top features: {extractor.get_feature_names()[:10]}")
