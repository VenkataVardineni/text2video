"""
Text classification model for sentiment analysis.

Implements an LSTM-based classifier for binary sentiment classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextClassifier(nn.Module):
    """
    LSTM-based text classifier for sentiment analysis.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        hidden_dim: Dimension of LSTM hidden state
        num_layers: Number of LSTM layers
        num_classes: Number of output classes (default: 2 for binary classification)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(TextClassifier, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state from both directions
        # hidden shape: (num_layers * 2, batch_size, hidden_dim) for bidirectional
        # Take the last layer's forward and backward hidden states
        forward_hidden = hidden[-2, :, :]  # Last forward hidden state
        backward_hidden = hidden[-1, :, :]  # Last backward hidden state
        combined_hidden = torch.cat([forward_hidden, backward_hidden], dim=1)
        
        # Fully connected layers
        out = self.fc1(combined_hidden)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class SimpleCNNClassifier(nn.Module):
    """
    Simple CNN-based text classifier as an alternative architecture.
    
    Args:
        vocab_size: Size of vocabulary
        embedding_dim: Dimension of word embeddings
        num_filters: Number of filters for each convolution
        filter_sizes: List of filter sizes
        num_classes: Number of output classes
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: list = [3, 4, 5],
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super(SimpleCNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length)
        
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_length)
        
        # Convolutions
        conv_outs = [F.relu(conv(embedded)) for conv in self.convs]
        pooled_outs = [F.max_pool1d(conv_out, kernel_size=conv_out.size(2)).squeeze(2)
                      for conv_out in conv_outs]
        
        # Concatenate and classify
        cat = torch.cat(pooled_outs, dim=1)
        cat = self.dropout(cat)
        out = self.fc(cat)
        
        return out

