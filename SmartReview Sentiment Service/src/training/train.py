"""
Training script for sentiment analysis model.

Trains the text classifier model with logging of accuracy and F1 scores
to both stdout and CSV.
"""

import os
import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

# Add parent directory to path for imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.text_classifier import TextClassifier


class SentimentDataset(Dataset):
    """Dataset class for sentiment analysis."""
    
    def __init__(self, texts: list, labels: list, vocab: dict, max_length: int = 256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        tokens = text.lower().split()
        indices = [self.vocab.get(token, self.vocab.get('<UNK>', 1)) for token in tokens]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices = indices + [0] * (self.max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def build_vocab(texts: list, min_freq: int = 2) -> dict:
    """Build vocabulary from texts."""
    word_freq = {}
    for text in texts:
        for word in text.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Create vocab dict
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in word_freq.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab


def load_data(data_path: str) -> Tuple[list, list]:
    """Load data from CSV file."""
    df = pd.read_csv(data_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    return texts, labels


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for texts, labels in tqdm(dataloader, desc="Training"):
        texts = texts.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for texts, labels in tqdm(dataloader, desc="Validating"):
            texts = texts.to(device)
            labels = labels.to(device)
            
            outputs = model(texts)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1
    }


def train(
    train_data_path: str,
    val_data_path: str,
    model_dir: str = "models",
    num_epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    embedding_dim: int = 128,
    hidden_dim: int = 256,
    num_layers: int = 2,
    max_length: int = 256,
    device: str = None
):
    """Main training function."""
    
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_texts, train_labels = load_data(train_data_path)
    val_texts, val_labels = load_data(val_data_path)
    
    # Build vocabulary
    print("Building vocabulary...")
    vocab = build_vocab(train_texts)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create datasets
    train_dataset = SentimentDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = SentimentDataset(val_texts, val_labels, vocab, max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = TextClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_classes=2,
        dropout=0.3
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Save vocabulary
    vocab_path = os.path.join(model_dir, "vocab.txt")
    with open(vocab_path, 'w') as f:
        for word, idx in vocab.items():
            f.write(f"{word}\t{idx}\n")
    print(f"Saved vocabulary to {vocab_path}")
    
    # Training metrics log
    metrics_log_path = os.path.join(model_dir, "training_metrics.csv")
    with open(metrics_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'train_accuracy', 'train_f1',
                        'val_loss', 'val_accuracy', 'val_f1'])
    
    # Training loop
    best_val_f1 = 0.0
    
    print("\nStarting training...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Print metrics
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Accuracy: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Accuracy: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}")
        
        # Log to CSV
        with open(metrics_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                train_metrics['loss'],
                train_metrics['accuracy'],
                train_metrics['f1'],
                val_metrics['loss'],
                val_metrics['accuracy'],
                val_metrics['f1']
            ])
        
        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            model_path = os.path.join(model_dir, "best_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'embedding_dim': embedding_dim,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'max_length': max_length,
            }, model_path)
            print(f"Saved best model (F1: {best_val_f1:.4f}) to {model_path}")
    
    print(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")
    print(f"Metrics logged to: {metrics_log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train sentiment analysis model")
    parser.add_argument("--train-data", type=str, default="data/processed/train.csv",
                       help="Path to training data CSV")
    parser.add_argument("--val-data", type=str, default="data/processed/val.csv",
                       help="Path to validation data CSV")
    parser.add_argument("--model-dir", type=str, default="models",
                       help="Directory to save model and vocabulary")
    parser.add_argument("--num-epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                       help="Learning rate")
    parser.add_argument("--embedding-dim", type=int, default=128,
                       help="Embedding dimension")
    parser.add_argument("--hidden-dim", type=int, default=256,
                       help="Hidden dimension")
    parser.add_argument("--num-layers", type=int, default=2,
                       help="Number of LSTM layers")
    parser.add_argument("--max-length", type=int, default=256,
                       help="Maximum sequence length")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    
    train(
        train_data_path=args.train_data,
        val_data_path=args.val_data,
        model_dir=args.model_dir,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        max_length=args.max_length,
        device=args.device
    )

