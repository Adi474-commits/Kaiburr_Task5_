"""
Sample Large Dataset - Create a smaller subset for faster processing
This script takes the large 1.5GB dataset and creates a 500MB sample
"""

import pandas as pd
import numpy as np
from pathlib import Path

print("=" * 60)
print("SAMPLING LARGE DATASET")
print("=" * 60)

# Read the full dataset in chunks to avoid memory issues
data_path = Path('data/raw/complaints.csv')

print(f"\n📂 Reading dataset: {data_path}")
print("⏳ This may take a moment...")

# First, count total rows
total_rows = sum(1 for _ in open(data_path, encoding='utf-8', errors='ignore')) - 1
print(f"📊 Total rows in dataset: {total_rows:,}")

# Calculate sample size (approximately 1/3 of data for ~500MB)
sample_size = min(100000, total_rows // 3)  # 100k rows or 1/3, whichever is smaller
print(f"🎯 Sampling {sample_size:,} rows")

# Read with sampling
skip_rows = sorted(np.random.choice(range(1, total_rows), total_rows - sample_size, replace=False))

print("📖 Loading sampled data...")
df = pd.read_csv(
    data_path,
    skiprows=skip_rows,
    encoding='utf-8',
    on_bad_lines='skip',
    low_memory=False
)

print(f"✅ Loaded {len(df):,} rows")
print(f"💾 Columns: {len(df.columns)}")

# Save sampled dataset
output_path = Path('data/raw/complaints_sample.csv')
print(f"\n💾 Saving sampled dataset to: {output_path}")
df.to_csv(output_path, index=False)

# Get file size
file_size_mb = output_path.stat().st_size / (1024 * 1024)
print(f"✅ Saved! File size: {file_size_mb:.1f} MB")

print("\n" + "=" * 60)
print("SAMPLE DATASET READY!")
print("=" * 60)
print(f"\n🎯 Next steps:")
print(f"   python scripts/train.py --data_path {output_path} --sample_size 50000")
print()
