"""
Demo Script - Test the classification pipeline with sample data
This generates synthetic data to test the complete pipeline without downloading the real dataset
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path

# Sample complaint texts for each category
sample_data = {
    0: [  # Credit reporting
        "I found errors on my credit report that need to be corrected",
        "My credit score is incorrect and I want it fixed",
        "There are unauthorized inquiries on my credit file",
        "Credit bureau won't remove incorrect information",
        "I disputed items on my credit report but nothing changed"
    ],
    1: [  # Debt collection
        "Debt collector keeps calling me at work",
        "I'm being harassed by collection agency",
        "They won't verify the debt they claim I owe",
        "Collector threatened to sue me for old debt",
        "Getting calls for someone else's debt"
    ],
    2: [  # Consumer Loan
        "My auto loan interest rate is too high",
        "Bank denied my personal loan application unfairly",
        "Loan servicer won't work with me on payments",
        "Hidden fees on my vehicle loan",
        "Student loan company won't return my calls"
    ],
    3: [  # Mortgage
        "Mortgage company lost my payment documents",
        "Home loan modification was denied without reason",
        "Escrow account has errors in calculations",
        "Foreclosure process started without proper notice",
        "Mortgage servicer won't accept my payments"
    ]
}

# Generate more samples by repeating and slight variations
np.random.seed(42)

texts = []
labels = []
label_names = []

category_names = {
    0: "Credit reporting, repair, or other",
    1: "Debt collection",
    2: "Consumer Loan",
    3: "Mortgage"
}

# Generate 200 samples per category (800 total)
for category_id, complaints in sample_data.items():
    for _ in range(200):
        # Pick a random base complaint
        base_text = np.random.choice(complaints)
        # Add some variation
        variations = [
            base_text,
            base_text + " and I need help resolving this issue",
            base_text + " please investigate this matter",
            "I am writing to complain about " + base_text.lower(),
            base_text + " this has been going on for months"
        ]
        text = np.random.choice(variations)
        
        texts.append(text)
        labels.append(category_id)
        label_names.append(category_names[category_id])

# Create DataFrame
df = pd.DataFrame({
    'text': texts,
    'label': label_names,
    'label_encoded': labels
})

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
output_path = Path('data/raw/complaints_demo.csv')
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print(f"✅ Created demo dataset: {output_path}")
print(f"📊 Total samples: {len(df)}")
print(f"\n📈 Category Distribution:")
print(df['label'].value_counts())
print(f"\n🎯 Now run:")
print(f"   python scripts/train.py --data_path {output_path}")
