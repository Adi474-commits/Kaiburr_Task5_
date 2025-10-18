"""
Prediction Script
Make predictions on new complaint text using trained models
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd
import json
from pathlib import Path
import logging

from src.predict import ComplaintClassifier, load_classifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Predict complaint categories')
    
    parser.add_argument('--model_dir', type=str, default='data/models',
                       help='Directory containing trained model')
    parser.add_argument('--model_name', type=str, default='best_model',
                       help='Name of model to use')
    parser.add_argument('--text', type=str, default=None,
                       help='Single text to classify')
    parser.add_argument('--input_file', type=str, default=None,
                       help='CSV file with texts to classify')
    parser.add_argument('--text_column', type=str, default='text',
                       help='Name of text column in input file')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file for predictions')
    parser.add_argument('--explain', action='store_true',
                       help='Provide explanation for predictions')
    
    return parser.parse_args()


def main():
    """Main prediction pipeline"""
    args = parse_args()
    
    # Load classifier
    logger.info(f"Loading classifier from {args.model_dir}...")
    classifier = load_classifier(args.model_dir, args.model_name)
    
    # Single text prediction
    if args.text:
        logger.info("Predicting single text...")
        
        if args.explain:
            result = classifier.explain_prediction(args.text)
        else:
            result = classifier.predict(args.text)
        
        print("\n" + "="*60)
        print("PREDICTION RESULT")
        print("="*60)
        print(f"\nInput Text:")
        print(f"  {result['text'][:200]}{'...' if len(result['text']) > 200 else ''}")
        print(f"\nPredicted Category:")
        print(f"  {result['category_name']} (ID: {result['category_id']})")
        
        if 'confidence' in result:
            print(f"\nConfidence: {result['confidence']:.4f}")
        
        if 'all_probabilities' in result:
            print("\nAll Category Probabilities:")
            for category, prob in result['all_probabilities'].items():
                print(f"  {category:40s}: {prob:.4f}")
        
        if 'important_features' in result:
            print("\nImportant Features:")
            for feature in result['important_features'][:10]:
                if 'contribution' in feature:
                    print(f"  {feature['feature']:20s}: {feature['contribution']:+.4f}")
                else:
                    print(f"  {feature['feature']:20s}: {feature.get('importance', 0):.4f}")
        
        print("="*60 + "\n")
        
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Result saved to {args.output_file}")
    
    # Batch prediction from file
    elif args.input_file:
        logger.info(f"Loading texts from {args.input_file}...")
        
        # Load input file
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
        elif args.input_file.endswith('.xlsx'):
            df = pd.read_excel(args.input_file)
        else:
            raise ValueError("Input file must be CSV or Excel")
        
        if args.text_column not in df.columns:
            raise ValueError(f"Column '{args.text_column}' not found in input file")
        
        texts = df[args.text_column].tolist()
        
        logger.info(f"Predicting {len(texts)} texts...")
        results = classifier.predict_batch(texts)
        
        # Add predictions to dataframe
        df['predicted_category_id'] = [r['category_id'] for r in results]
        df['predicted_category_name'] = [r['category_name'] for r in results]
        df['confidence'] = [r.get('confidence', None) for r in results]
        
        # Save results
        if args.output_file:
            output_path = Path(args.output_file)
        else:
            input_path = Path(args.input_file)
            output_path = input_path.parent / f"{input_path.stem}_predictions{input_path.suffix}"
        
        if str(output_path).endswith('.csv'):
            df.to_csv(output_path, index=False)
        else:
            df.to_excel(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"\nTotal texts processed: {len(texts)}")
        print("\nCategory Distribution:")
        print(df['predicted_category_name'].value_counts().to_string())
        
        if 'confidence' in df.columns:
            print(f"\nAverage Confidence: {df['confidence'].mean():.4f}")
            print(f"Min Confidence: {df['confidence'].min():.4f}")
            print(f"Max Confidence: {df['confidence'].max():.4f}")
        
        print("="*60 + "\n")
    
    # Interactive mode
    else:
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("Enter complaint text (or 'quit' to exit):\n")
        
        while True:
            text = input("> ")
            
            if text.lower() in ['quit', 'exit', 'q']:
                break
            
            if not text.strip():
                continue
            
            result = classifier.predict(text)
            
            print(f"\nPredicted Category: {result['category_name']}")
            if 'confidence' in result:
                print(f"Confidence: {result['confidence']:.4f}")
            print()


if __name__ == "__main__":
    main()
