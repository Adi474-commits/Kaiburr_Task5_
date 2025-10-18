"""
Model Evaluation Module
Comprehensive evaluation metrics and visualization for classification models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, auc
)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import logging
import os
from typing import Dict, List, Optional, Any
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and compare multiple classification models
    """
    
    def __init__(self, category_names: Optional[Dict[int, str]] = None):
        """
        Initialize model evaluator
        
        Args:
            category_names: Mapping of category IDs to names
        """
        if category_names is None:
            self.category_names = {
                0: "Credit reporting, repair, or other",
                1: "Debt collection",
                2: "Consumer Loan",
                3: "Mortgage"
            }
        else:
            self.category_names = category_names
        
        self.results = {}
        logger.info("ModelEvaluator initialized")
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Evaluate a single model
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = time.time() - start_time
        
        # Probabilities (if available)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = model.predict_proba(X_test)
            elif hasattr(model, 'decision_function'):
                y_proba = model.decision_function(X_test)
            else:
                y_proba = None
        except:
            y_proba = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)
        
        # ROC-AUC (if probabilities available)
        roc_auc = None
        if y_proba is not None:
            try:
                y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
                if y_test_bin.shape[1] > 1:
                    roc_auc = roc_auc_score(y_test_bin, y_proba, average='weighted', multi_class='ovr')
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Store results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
            'roc_auc': roc_auc,
            'inference_time': inference_time,
            'predictions': y_pred,
            'probabilities': y_proba,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.results[model_name] = results
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        
        return results
    
    def evaluate_multiple_models(self, models: Dict[str, Any], 
                                 X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Evaluate multiple models
        
        Args:
            models: Dictionary of model_name -> model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            DataFrame with comparison results
        """
        results_list = []
        
        for model_name, model in models.items():
            result = self.evaluate_model(model, X_test, y_test, model_name)
            results_list.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'] if result['roc_auc'] else 'N/A',
                'Inference Time (s)': result['inference_time']
            })
        
        comparison_df = pd.DataFrame(results_list)
        comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
        
        return comparison_df
    
    def cross_validate_model(self, model, X: np.ndarray, y: np.ndarray,
                            cv: int = 5, model_name: str = "model") -> Dict[str, float]:
        """
        Perform cross-validation
        
        Args:
            model: Model to evaluate
            X: Features
            y: Labels
            cv: Number of folds
            model_name: Name of the model
            
        Returns:
            Dictionary of cross-validation scores
        """
        logger.info(f"Cross-validating {model_name} with {cv} folds...")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
        
        results = {
            'model_name': model_name,
            'cv_mean': scores.mean(),
            'cv_std': scores.std(),
            'cv_scores': scores
        }
        
        logger.info(f"{model_name} - CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
        
        return results
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None,
                             figsize: tuple = (10, 8)):
        """
        Plot confusion matrix for a model
        
        Args:
            model_name: Name of the model
            save_path: Path to save the figure
            figsize: Figure size
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        cm = self.results[model_name]['confusion_matrix']
        
        plt.figure(figsize=figsize)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=[self.category_names[i] for i in range(len(cm))],
                   yticklabels=[self.category_names[i] for i in range(len(cm))])
        plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path: Optional[str] = None,
                            figsize: tuple = (12, 6)):
        """
        Plot comparison of all models
        
        Args:
            save_path: Path to save the figure
            figsize: Figure size
        """
        if not self.results:
            raise ValueError("No results to plot")
        
        # Prepare data
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(self.results.keys())
        
        data = {metric: [self.results[model][metric] for model in model_names]
                for metric in metrics}
        
        # Create plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.replace('_', ' ').title())
        
        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_test: np.ndarray, save_path: Optional[str] = None,
                       figsize: tuple = (10, 8)):
        """
        Plot ROC curves for all models
        
        Args:
            y_test: True labels
            save_path: Path to save the figure
            figsize: Figure size
        """
        plt.figure(figsize=figsize)
        
        # Binarize labels
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        n_classes = y_test_bin.shape[1]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.results)))
        
        for (model_name, result), color in zip(self.results.items(), colors):
            if result['probabilities'] is None:
                continue
            
            # Compute ROC curve for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], 
                                             result['probabilities'][:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            
            # Compute micro-average ROC curve
            fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), 
                                                     result['probabilities'].ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            
            plt.plot(fpr["micro"], tpr["micro"], color=color, lw=2,
                    label=f'{model_name} (AUC = {roc_auc["micro"]:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_classification_report(self, model_name: str,
                                       save_path: Optional[str] = None) -> str:
        """
        Generate detailed classification report
        
        Args:
            model_name: Name of the model
            save_path: Path to save the report
            
        Returns:
            Classification report string
        """
        if model_name not in self.results:
            raise ValueError(f"No results found for {model_name}")
        
        result = self.results[model_name]
        y_true = None  # Would need to store this in results
        y_pred = result['predictions']
        
        # Generate report
        target_names = [self.category_names[i] for i in sorted(self.category_names.keys())]
        
        # Create a mock y_true from confusion matrix for the report
        cm = result['confusion_matrix']
        y_true = []
        y_pred_list = []
        for true_label in range(len(cm)):
            for pred_label in range(len(cm)):
                count = cm[true_label, pred_label]
                y_true.extend([true_label] * count)
                y_pred_list.extend([pred_label] * count)
        
        report = classification_report(y_true, y_pred_list, 
                                      target_names=target_names,
                                      digits=4)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(f"Classification Report - {model_name}\n")
                f.write("=" * 60 + "\n\n")
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report
    
    def save_results(self, output_dir: str):
        """
        Save all evaluation results
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save summary
        summary_data = []
        for model_name, result in self.results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1_score'],
                'ROC-AUC': result['roc_auc'],
                'Inference Time': result['inference_time']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
        
        logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=4, 
                               n_informative=15, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                         random_state=42)
    
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(model, X_test, y_test, "Random Forest")
    
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
