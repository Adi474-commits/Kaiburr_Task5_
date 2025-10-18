"""
Training Script
Train multiple classification models on consumer complaint data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import yaml
import logging

from src.data_loader import load_data, map_categories, split_data, save_processed_data, load_config
from src.preprocessing import TextPreprocessor
from src.feature_engineering import FeatureExtractor
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train complaint classification models')
    
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to input data file')
    parser.add_argument('--output_dir', type=str, default='data/models',
                       help='Directory to save trained models')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Number of samples to use (for testing)')
    parser.add_argument('--models', type=str, nargs='+', 
                       default=['logistic_regression', 'naive_bayes', 'svm', 
                               'random_forest', 'xgboost'],
                       help='Models to train')
    parser.add_argument('--skip_preprocessing', action='store_true',
                       help='Skip preprocessing if already done')
    
    return parser.parse_args()


def main():
    """Main training pipeline"""
    args = parse_args()
    
    # Load configuration
    logger.info("Loading configuration...")
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data_path:
        config['data']['raw_data_path'] = args.data_path
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load data
    logger.info("=" * 60)
    logger.info("STEP 1: Loading Data")
    logger.info("=" * 60)
    
    if not args.skip_preprocessing:
        df = load_data(
            config['data']['raw_data_path'],
            sample_size=args.sample_size
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
        processed_dir = Path(config['data']['processed_data_path'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        save_processed_data(train_df, val_df, test_df, str(processed_dir))
    else:
        logger.info("Loading preprocessed data...")
        processed_dir = Path(config['data']['processed_data_path'])
        train_df = pd.read_csv(processed_dir / 'train.csv')
        val_df = pd.read_csv(processed_dir / 'val.csv')
        test_df = pd.read_csv(processed_dir / 'test.csv')
    
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    
    # Step 2: Text Preprocessing
    logger.info("=" * 60)
    logger.info("STEP 2: Text Preprocessing")
    logger.info("=" * 60)
    
    preprocessor = TextPreprocessor(
        lowercase=config['preprocessing']['lowercase'],
        remove_urls=config['preprocessing']['remove_urls'],
        remove_emails=config['preprocessing']['remove_emails'],
        remove_special_chars=config['preprocessing']['remove_special_chars'],
        remove_numbers=config['preprocessing']['remove_numbers'],
        remove_stopwords=config['preprocessing']['remove_stopwords'],
        lemmatize=config['preprocessing']['lemmatize'],
        min_word_length=config['preprocessing']['min_word_length']
    )
    
    train_df = preprocessor.preprocess_dataframe(train_df)
    val_df = preprocessor.preprocess_dataframe(val_df)
    test_df = preprocessor.preprocess_dataframe(test_df)
    
    # Save preprocessor
    joblib.dump(preprocessor, output_dir / 'preprocessor.pkl')
    logger.info(f"Preprocessor saved to {output_dir / 'preprocessor.pkl'}")
    
    # Step 3: Feature Engineering
    logger.info("=" * 60)
    logger.info("STEP 3: Feature Engineering")
    logger.info("=" * 60)
    
    feature_extractor = FeatureExtractor(
        method=config['features']['use_tfidf'] and 'tfidf' or 'count',
        max_features=config['features']['tfidf_max_features'],
        ngram_range=tuple(config['features']['tfidf_ngram_range']),
        min_df=config['features']['tfidf_min_df'],
        max_df=config['features']['tfidf_max_df']
    )
    
    X_train = feature_extractor.fit_transform(train_df['processed_text'])
    X_val = feature_extractor.transform(val_df['processed_text'])
    X_test = feature_extractor.transform(test_df['processed_text'])
    
    y_train = train_df['label_encoded'].values
    y_val = val_df['label_encoded'].values
    y_test = test_df['label_encoded'].values
    
    logger.info(f"Training features shape: {X_train.shape}")
    
    # Save feature extractor
    feature_extractor.save(str(output_dir / 'feature_extractor.pkl'))
    
    # Step 4: Model Training
    logger.info("=" * 60)
    logger.info("STEP 4: Training Models")
    logger.info("=" * 60)
    
    trainer = ModelTrainer(use_class_weights=config['training']['use_class_weights'])
    trainer.train_all(X_train, y_train, models_to_train=args.models)
    
    # Save all models
    if config['training']['save_models']:
        trainer.save_all_models(str(output_dir))
    
    # Step 5: Model Evaluation
    logger.info("=" * 60)
    logger.info("STEP 5: Evaluating Models")
    logger.info("=" * 60)
    
    evaluator = ModelEvaluator()
    
    # Evaluate on validation set
    logger.info("\nValidation Set Results:")
    val_results = evaluator.evaluate_multiple_models(trainer.models, X_val, y_val)
    print(val_results.to_string(index=False))
    
    # Evaluate on test set
    logger.info("\nTest Set Results:")
    evaluator_test = ModelEvaluator()
    test_results = evaluator_test.evaluate_multiple_models(trainer.models, X_test, y_test)
    print(test_results.to_string(index=False))
    
    # Save results
    results_dir = Path('results/reports')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    val_results.to_csv(results_dir / 'validation_results.csv', index=False)
    test_results.to_csv(results_dir / 'test_results.csv', index=False)
    
    # Find best model
    best_model_name = test_results.loc[test_results['F1-Score'].idxmax(), 'Model']
    best_model = trainer.get_model(best_model_name)
    
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"Test F1-Score: {test_results.loc[test_results['Model'] == best_model_name, 'F1-Score'].values[0]:.4f}")
    
    # Save best model separately
    joblib.dump(best_model, output_dir / 'best_model.pkl')
    logger.info(f"Best model saved to {output_dir / 'best_model.pkl'}")
    
    # Save training metadata
    metadata = {
        'best_model': best_model_name,
        'training_config': config,
        'dataset_stats': {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'n_features': X_train.shape[1]
        },
        'training_times': trainer.training_times,
        'test_results': test_results.to_dict('records')
    }
    
    import json
    with open(output_dir / 'training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
