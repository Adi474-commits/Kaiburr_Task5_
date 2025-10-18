"""
Evaluation Script
Evaluate trained models and generate visualizations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging

from src.evaluation import ModelEvaluator
from src.feature_engineering import FeatureExtractor
from src.preprocessing import TextPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate complaint classification models')
    
    parser.add_argument('--model_dir', type=str, default='data/models',
                       help='Directory containing trained models')
    parser.add_argument('--model_name', type=str, default='best_model',
                       help='Name of model to evaluate')
    parser.add_argument('--data_path', type=str, default='data/processed/test.csv',
                       help='Path to test data')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate visualization plots')
    
    return parser.parse_args()


def main():
    """Main evaluation pipeline"""
    args = parse_args()
    
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    reports_dir = output_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    logger.info(f"Loading test data from {args.data_path}...")
    test_df = pd.read_csv(args.data_path)
    
    # Load preprocessor and feature extractor
    logger.info("Loading preprocessor and feature extractor...")
    preprocessor = joblib.load(model_dir / 'preprocessor.pkl')
    feature_extractor = FeatureExtractor.load(str(model_dir / 'feature_extractor.pkl'))
    
    # Preprocess if needed
    if 'processed_text' not in test_df.columns:
        logger.info("Preprocessing text...")
        test_df = preprocessor.preprocess_dataframe(test_df)
    
    # Extract features
    logger.info("Extracting features...")
    X_test = feature_extractor.transform(test_df['processed_text'])
    y_test = test_df['label_encoded'].values
    
    # Load model(s)
    logger.info(f"Loading model: {args.model_name}...")
    model_path = model_dir / f'{args.model_name}.pkl'
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Available models:")
        for f in model_dir.glob('*.pkl'):
            if f.stem not in ['preprocessor', 'feature_extractor']:
                logger.info(f"  - {f.stem}")
        return
    
    model = joblib.load(model_path)
    
    # Evaluate model
    logger.info("=" * 60)
    logger.info("EVALUATING MODEL")
    logger.info("=" * 60)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, args.model_name)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1-Score:  {results['f1_score']:.4f}")
    if results['roc_auc']:
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"Inference Time: {results['inference_time']:.4f}s")
    print(f"{'='*60}\n")
    
    # Per-class metrics
    print("Per-Class Metrics:")
    print("-" * 60)
    category_names = {
        0: "Credit reporting, repair, or other",
        1: "Debt collection",
        2: "Consumer Loan",
        3: "Mortgage"
    }
    
    for i in range(len(results['precision_per_class'])):
        print(f"\n{category_names.get(i, f'Category {i}')}:")
        print(f"  Precision: {results['precision_per_class'][i]:.4f}")
        print(f"  Recall:    {results['recall_per_class'][i]:.4f}")
        print(f"  F1-Score:  {results['f1_per_class'][i]:.4f}")
    
    # Generate classification report
    logger.info("\nGenerating classification report...")
    report = evaluator.generate_classification_report(
        args.model_name,
        save_path=str(reports_dir / f'{args.model_name}_classification_report.txt')
    )
    print("\n" + report)
    
    # Generate visualizations
    if args.generate_plots:
        logger.info("\nGenerating visualizations...")
        
        # Confusion matrix
        logger.info("Plotting confusion matrix...")
        evaluator.plot_confusion_matrix(
            args.model_name,
            save_path=str(figures_dir / f'{args.model_name}_confusion_matrix.png')
        )
        
        # If multiple models are available, create comparison plots
        all_models = {}
        for model_file in model_dir.glob('*.pkl'):
            if model_file.stem not in ['preprocessor', 'feature_extractor', 'best_model']:
                try:
                    m = joblib.load(model_file)
                    all_models[model_file.stem] = m
                except:
                    pass
        
        if len(all_models) > 1:
            logger.info(f"Evaluating {len(all_models)} models for comparison...")
            evaluator_all = ModelEvaluator()
            comparison_df = evaluator_all.evaluate_multiple_models(all_models, X_test, y_test)
            
            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            print(comparison_df.to_string(index=False))
            
            # Save comparison
            comparison_df.to_csv(reports_dir / 'model_comparison.csv', index=False)
            
            # Plot comparison
            logger.info("Plotting model comparison...")
            evaluator_all.plot_model_comparison(
                save_path=str(figures_dir / 'model_comparison.png')
            )
            
            # Plot ROC curves
            logger.info("Plotting ROC curves...")
            try:
                evaluator_all.plot_roc_curves(
                    y_test,
                    save_path=str(figures_dir / 'roc_curves.png')
                )
            except Exception as e:
                logger.warning(f"Could not plot ROC curves: {e}")
    
    # Save detailed results
    logger.info("\nSaving results...")
    results_summary = {
        'model': args.model_name,
        'accuracy': float(results['accuracy']),
        'precision': float(results['precision']),
        'recall': float(results['recall']),
        'f1_score': float(results['f1_score']),
        'roc_auc': float(results['roc_auc']) if results['roc_auc'] else None,
        'inference_time': float(results['inference_time']),
        'test_samples': len(y_test)
    }
    
    import json
    with open(reports_dir / f'{args.model_name}_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save confusion matrix
    cm_df = pd.DataFrame(
        results['confusion_matrix'],
        index=[category_names.get(i, f'Category {i}') for i in range(len(results['confusion_matrix']))],
        columns=[category_names.get(i, f'Category {i}') for i in range(len(results['confusion_matrix']))]
    )
    cm_df.to_csv(reports_dir / f'{args.model_name}_confusion_matrix.csv')
    
    logger.info("=" * 60)
    logger.info("EVALUATION COMPLETE!")
    logger.info(f"Results saved to {reports_dir}")
    if args.generate_plots:
        logger.info(f"Figures saved to {figures_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
