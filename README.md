# Consumer Complaint Text Classification

A comprehensive machine learning project for multi-class text classification of consumer complaints into categories: Credit reporting/repair, Debt collection, Consumer Loan, and Mortgage.

## Author
Adithya N Reddy
BL.EN.U4EAC22075
adithyasnr@gmail.com


##  Project Overview

This project performs text classification on consumer complaint data using Natural Language Processing (NLP) and Machine Learning techniques. The goal is to automatically categorize consumer complaints into one of four categories:

| Category ID | Category Name |
|-------------|--------------|
| 0 | Credit reporting, repair, or other |
| 1 | Debt collection |
| 2 | Consumer Loan |
| 3 | Mortgage |

##  Dataset

**Source**: [Consumer Complaint Database](https://catalog.data.gov/dataset/consumer-complaint-database)

The dataset contains consumer complaints submitted to the Consumer Financial Protection Bureau (CFPB). Each complaint includes:
- Complaint narrative/text
- Product category
- Company information
- Date submitted
- Issue type
- And more...

##  Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone <repository-url>
cd Task_5
```

2. **Create a virtual environment** (recommended)
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install required packages**
```bash
pip install -r requirements.txt
```

4. **Download NLTK data** (if not already downloaded)
```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

##  Project Structure

```
Task_5/
│
├── data/
│   ├── raw/                    # Raw data files
│   ├── processed/              # Processed data files
│   └── models/                 # Saved models
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   └── 02_model_experiments.ipynb  # Model experimentation
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading utilities
│   ├── preprocessing.py       # Text preprocessing functions
│   ├── feature_engineering.py # Feature extraction
│   ├── models.py              # Model definitions
│   ├── evaluation.py          # Model evaluation metrics
│   └── predict.py             # Prediction utilities
│
├── scripts/
│   ├── train.py               # Model training script
│   ├── evaluate.py            # Model evaluation script
│   └── predict.py             # Prediction script
│
├── results/
│   ├── figures/               # Visualization outputs
│   └── reports/               # Performance reports
│
├── requirements.txt           # Python dependencies
├── config.yaml               # Configuration file
├── .gitignore
└── README.md
```

##  Usage

### 1. Exploratory Data Analysis
Explore the dataset and understand the distribution:
```bash
jupyter notebook notebooks/01_eda.ipynb
```

### 2. Train Models
Train multiple classification models:
```bash
python scripts/train.py --data_path data/raw/complaints.csv --output_dir data/models
```

### 3. Evaluate Models
Evaluate trained models on test set:
```bash
python scripts/evaluate.py --model_path data/models/best_model.pkl --data_path data/processed/test.csv
```

### 4. Make Predictions
Predict category for new complaints:
```bash
python scripts/predict.py --model_path data/models/best_model.pkl --text "Your complaint text here"
```

Or use the Python API:
```python
from src.predict import ComplaintClassifier

classifier = ComplaintClassifier('data/models/best_model.pkl')
prediction = classifier.predict("I have an issue with my credit report")
print(f"Category: {prediction['category']}")
print(f"Confidence: {prediction['confidence']:.2%}")
```

##  Methodology

### Step 1: Exploratory Data Analysis and Feature Engineering
- Load and inspect the dataset
- Analyze text length distribution
- Visualize class distribution
- Identify imbalanced classes
- Generate word clouds for each category
- Analyze n-grams (unigrams, bigrams, trigrams)

### Step 2: Text Pre-Processing
- Convert text to lowercase
- Remove special characters and numbers
- Remove URLs and email addresses
- Tokenization
- Remove stopwords
- Lemmatization/Stemming
- Handle missing values

### Step 3: Selection of Multi-Classification Models
The following models are implemented and compared:
1. **Logistic Regression** (with TF-IDF)
2. **Naive Bayes** (MultinomialNB)
3. **Support Vector Machine** (SVM)
4. **Random Forest Classifier**
5. **XGBoost**
6. **LSTM Neural Network** (Deep Learning)
7. **BERT** (Transfer Learning - Optional)

### Step 4: Comparison of Model Performance
Models are compared using:
- Accuracy
- Precision, Recall, F1-Score (per class)
- Confusion Matrix
- ROC-AUC curves
- Training time
- Inference time

### Step 5: Model Evaluation
Comprehensive evaluation including:
- Cross-validation scores
- Classification reports
- Error analysis
- Feature importance (for tree-based models)
- Misclassification analysis

### Step 6: Prediction
- Load the best performing model
- Preprocess new complaint text
- Generate predictions with confidence scores
- Provide category labels

##  Model Performance

After training on 6,265 consumer complaints across 4 categories, here are the performance metrics:

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (s) |
|-------|----------|-----------|--------|----------|-------------------|
| **SVM** ⭐ | **86.67%** | **0.8678** | **0.8667** | **0.8666** | **0.79** |
| Logistic Regression | 86.11% | 0.8617 | 0.8611 | 0.8613 | 0.41 |
| XGBoost | 85.79% | 0.8603 | 0.8579 | 0.8562 | 123.95 |
| Random Forest | 85.08% | 0.8557 | 0.8508 | 0.8485 | 6.01 |
| Naive Bayes | 83.40% | 0.8359 | 0.8340 | 0.8315 | 0.01 |

###  Best Model: Support Vector Machine (SVM)

The **SVM model** achieved the highest F1-Score of **0.8666** with excellent balance between precision and recall across all categories. Detailed classification report:

| Category | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Credit reporting, repair, or other | 0.8702 | 0.9095 | 0.8894 | 619 |
| Debt collection | 0.8703 | 0.8193 | 0.8440 | 393 |
| Consumer Loan | 0.7414 | 0.7679 | 0.7544 | 112 |
| Mortgage | 0.9583 | 0.8915 | 0.9237 | 129 |
| **Weighted Avg** | **0.8678** | **0.8667** | **0.8666** | **1,253** |

### Dataset Distribution

The training dataset consisted of:
- **Credit reporting, repair, or other**: 3,096 complaints (49.4%)
- **Debt collection**: 1,963 complaints (31.3%)
- **Consumer Loan**: 561 complaints (9.0%)
- **Mortgage**: 645 complaints (10.3%)

##  Results & Visualizations

<img width="475" height="183" alt="Screenshot 2025-10-18 114902" src="https://github.com/user-attachments/assets/447941bb-3df0-4c57-bffd-e163326deda8" />


### 1. Category Distribution

<!-- Paste category distribution image here -->
![Category Distribution]<img width="4530" height="1772" alt="01_category_distribution" src="https://github.com/user-attachments/assets/3446c90e-e6cf-432b-9e92-d829a8e247ce" />


The bar chart and pie chart show the distribution of complaints across the four categories. Credit reporting complaints dominate the dataset at nearly 50%, followed by debt collection at 31.3%. Consumer loans and mortgage complaints make up the remaining ~20% of the dataset.

---

### 2. Word Clouds by Category

<!-- Paste word clouds image here -->
![Word Clouds]<img width="5970" height="4074" alt="02_wordclouds" src="https://github.com/user-attachments/assets/163e263f-5921-474b-a266-4d2f0e4a9289" />


Word clouds for each complaint category reveal the most frequent terms:
- **Credit reporting**: Terms like "credit", "report", "account", "information", "bureau" are prominent
- **Debt collection**: Words such as "debt", "payment", "collection", "call", "company" dominate
- **Consumer Loan**: Shows terms like "loan", "payment", "vehicle", "car", "student"
- **Mortgage**: Features "mortgage", "home", "loan", "property", "foreclosure"

These visualizations help identify key distinguishing features between complaint categories.

---

### 3. Model Performance Comparison

<!-- Paste model comparison chart image here -->
![Model Comparison]<img width="4170" height="2074" alt="03_model_comparison" src="https://github.com/user-attachments/assets/3a637c37-ef44-48bc-a1ae-cc08d8cf83ab" />


This chart compares all five models across four metrics: Accuracy, Precision, Recall, and F1-Score. The visualization clearly shows:
- **SVM** (red bars) performs best across all metrics
- **Logistic Regression** (cyan bars) is a close second
- All models achieve >83% accuracy, demonstrating the effectiveness of TF-IDF features
- SVM provides the best balance between precision and recall

---

### 4. Confusion Matrices

<!-- Paste confusion matrices image here -->
![Confusion Matrices]<img width="5983" height="3863" alt="04_confusion_matrices" src="https://github.com/user-attachments/assets/33bb9345-1f0c-46d1-a8df-2232f188b837" />


Confusion matrices for all five models reveal classification patterns:
- **Credit reporting** (Category 0) has the highest prediction accuracy due to larger sample size
- **Mortgage** (Category 3) shows high precision with minimal misclassifications
- **Consumer Loan** (Category 2) has slightly lower accuracy due to smaller sample size
- Most misclassifications occur between similar categories (e.g., Consumer Loan vs Mortgage)
- SVM's confusion matrix shows the most balanced performance across all categories

The diagonal values (correct predictions) are highest for SVM, confirming its superior performance.

---

###  Generated Reports

All detailed reports are available in the `results/reports/` directory:
- **`model_comparison.csv`**: Complete metrics for all models
- **`best_model_report.txt`**: Detailed classification report for the best model (SVM)





