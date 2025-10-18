"""
Data Loading and Preprocessing Module
Handles loading data from various sources and initial data preparation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import yaml
from typing import Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise


def load_data(file_path: str, 
              text_column: str = 'Consumer complaint narrative',
              label_column: str = 'Product',
              sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Load consumer complaint data from CSV file
    
    Args:
        file_path: Path to the data file
        text_column: Name of the column containing complaint text
        label_column: Name of the column containing product category
        sample_size: Number of samples to load (None for all)
        
    Returns:
        DataFrame with complaint text and labels
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel.")
        
        logger.info(f"Loaded {len(df)} records")
        
        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} records")
        
        # Check if required columns exist
        if text_column not in df.columns:
            logger.warning(f"Column '{text_column}' not found. Available columns: {df.columns.tolist()}")
            # Try to find a similar column
            text_cols = [col for col in df.columns if 'complaint' in col.lower() or 'narrative' in col.lower()]
            if text_cols:
                text_column = text_cols[0]
                logger.info(f"Using column '{text_column}' as text column")
        
        if label_column not in df.columns:
            logger.warning(f"Column '{label_column}' not found. Available columns: {df.columns.tolist()}")
            # Try to find a similar column
            label_cols = [col for col in df.columns if 'product' in col.lower() or 'category' in col.lower()]
            if label_cols:
                label_column = label_cols[0]
                logger.info(f"Using column '{label_column}' as label column")
        
        # Create clean dataset with text and labels
        df_clean = df[[text_column, label_column]].copy()
        df_clean.columns = ['text', 'label']
        
        # Remove missing values
        initial_size = len(df_clean)
        df_clean = df_clean.dropna()
        logger.info(f"Removed {initial_size - len(df_clean)} rows with missing values")
        
        # Remove empty strings
        df_clean = df_clean[df_clean['text'].str.strip() != '']
        
        logger.info(f"Final dataset size: {len(df_clean)} records")
        logger.info(f"Unique labels: {df_clean['label'].nunique()}")
        
        return df_clean
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def map_categories(df: pd.DataFrame, 
                   category_mapping: Optional[dict] = None) -> pd.DataFrame:
    """
    Map product categories to numerical labels
    
    Args:
        df: DataFrame with 'label' column
        category_mapping: Custom mapping dictionary (optional)
        
    Returns:
        DataFrame with mapped labels and original labels
    """
    if category_mapping is None:
        # Default mapping based on task requirements
        category_mapping = {
            'Credit reporting, repair, or other': 0,
            'Credit reporting': 0,
            'Credit reporting, credit repair services, or other personal consumer reports': 0,
            'Debt collection': 1,
            'Consumer Loan': 2,
            'Vehicle loan or lease': 2,
            'Student loan': 2,
            'Mortgage': 3
        }
    
    df = df.copy()
    
    # Map labels
    df['label_encoded'] = df['label'].map(category_mapping)
    
    # Handle unmapped categories
    unmapped = df[df['label_encoded'].isna()]
    if len(unmapped) > 0:
        logger.warning(f"Found {len(unmapped)} unmapped categories:")
        logger.warning(f"Unique unmapped categories: {unmapped['label'].unique()}")
        
        # Assign to category 0 (other) for unmapped
        df['label_encoded'] = df['label_encoded'].fillna(0)
    
    df['label_name'] = df['label']
    
    logger.info(f"Category distribution:\n{df['label_encoded'].value_counts().sort_index()}")
    
    return df


def split_data(df: pd.DataFrame, 
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42,
               stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
        stratify: Whether to stratify split by labels
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info(f"Splitting data: test_size={test_size}, val_size={val_size}")
    
    stratify_col = df['label_encoded'] if stratify else None
    
    # Split into train+val and test
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        stratify=stratify_col
    )
    
    # Split train+val into train and val
    stratify_col_train = train_val_df['label_encoded'] if stratify else None
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=random_state,
        stratify=stratify_col_train
    )
    
    logger.info(f"Train size: {len(train_df)}")
    logger.info(f"Validation size: {len(val_df)}")
    logger.info(f"Test size: {len(test_df)}")
    
    return train_df, val_df, test_df


def save_processed_data(train_df: pd.DataFrame,
                       val_df: pd.DataFrame,
                       test_df: pd.DataFrame,
                       output_dir: str = 'data/processed/'):
    """
    Save processed data splits to CSV files
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        test_df: Test DataFrame
        output_dir: Output directory path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)
    test_df.to_csv(f'{output_dir}/test.csv', index=False)
    
    logger.info(f"Saved processed data to {output_dir}")


if __name__ == "__main__":
    # Example usage
    config = load_config()
    
    # Load data (you'll need to download the dataset first)
    df = load_data(
        config['data']['raw_data_path'],
        sample_size=10000  # Use sample for testing
    )
    
    # Map categories
    df = map_categories(df)
    
    # Split data
    train_df, val_df, test_df = split_data(
        df,
        test_size=config['data']['test_size'],
        val_size=config['data']['validation_size'],
        random_state=config['data']['random_state']
    )
    
    # Save processed data
    save_processed_data(train_df, val_df, test_df, config['data']['processed_data_path'])
