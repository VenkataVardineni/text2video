"""
Dataset preparation script for sentiment analysis.

Downloads IMDB reviews dataset, cleans text, and splits into train/val/test sets.
Saves processed data as CSV files.
"""

import os
import re
import csv
import argparse
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlretrieve
import tarfile
import shutil

import pandas as pd
from sklearn.model_selection import train_test_split


def download_imdb_dataset(data_dir: str = "data/raw") -> str:
    """Download IMDB dataset from Stanford AI Lab."""
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    tar_path = os.path.join(data_dir, "aclImdb_v1.tar.gz")
    
    if not os.path.exists(tar_path):
        print("Downloading IMDB dataset...")
        urlretrieve(url, tar_path)
        print("Download complete.")
    
    # Extract if not already extracted
    extract_dir = os.path.join(data_dir, "aclImdb")
    if not os.path.exists(extract_dir):
        print("Extracting dataset...")
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(data_dir)
        print("Extraction complete.")
    
    return extract_dir


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def load_reviews(data_dir: str) -> Tuple[List[str], List[int]]:
    """Load reviews from IMDB directory structure."""
    texts = []
    labels = []
    
    # Load positive reviews
    pos_dir = os.path.join(data_dir, "train", "pos")
    for filename in os.listdir(pos_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(pos_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                texts.append(text)
                labels.append(1)  # 1 = positive
    
    # Load negative reviews
    neg_dir = os.path.join(data_dir, "train", "neg")
    for filename in os.listdir(neg_dir):
        if filename.endswith(".txt"):
            filepath = os.path.join(neg_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                text = clean_text(f.read())
                texts.append(text)
                labels.append(0)  # 0 = negative
    
    return texts, labels


def prepare_dataset(
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42
):
    """Main function to prepare dataset."""
    print("Preparing dataset...")
    
    # Download and extract dataset
    raw_data_dir = download_imdb_dataset()
    
    # Load reviews
    print("Loading reviews...")
    texts, labels = load_reviews(raw_data_dir)
    print(f"Loaded {len(texts)} reviews")
    
    # Create DataFrame
    df = pd.DataFrame({"text": texts, "label": labels})
    
    # Split into train, validation, and test
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    
    # Second split: train vs val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_ratio, random_state=random_state, stratify=train_val_df["label"]
    )
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"\nDataset prepared successfully!")
    print(f"Train: {len(train_df)} samples ({train_df['label'].sum()} positive, {len(train_df) - train_df['label'].sum()} negative)")
    print(f"Validation: {len(val_df)} samples ({val_df['label'].sum()} positive, {len(val_df) - val_df['label'].sum()} negative)")
    print(f"Test: {len(test_df)} samples ({test_df['label'].sum()} positive, {len(test_df) - test_df['label'].sum()} negative)")
    print(f"\nSaved to:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare sentiment analysis dataset")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--test-size", type=float, default=0.2,
                       help="Proportion of data for test set")
    parser.add_argument("--val-size", type=float, default=0.1,
                       help="Proportion of data for validation set")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    prepare_dataset(
        output_dir=args.output_dir,
        test_size=args.test_size,
        val_size=args.val_size,
        random_state=args.random_state
    )

