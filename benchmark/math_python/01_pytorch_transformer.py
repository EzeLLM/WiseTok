"""
PyTorch Transformer Block Implementation

A minimal transformer encoder block with multi-head attention,
position embeddings, feed-forward layers, and layer normalization.

Mathematical notation: $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_seq_length: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_seq_length, d_model)  # [max_seq_length, d_model]
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)  # [max_seq_length, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            -(math.log(10000.0) / d_model)
        )  # [d_model // 2]

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_seq_length, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [batch_size, seq_length, d_model]"""
        x = x + self.pe[:, :x.size(1), :]  # broadcast batch
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism.

    Divides d_model into num_heads parallel attention heads.
    Each head has dimension d_k = d_model // num_heads.
    """

    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # [d_model, d_model]
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: [batch_size, seq_len_q, d_model]
            k: [batch_size, seq_len_k, d_model]
            v: [batch_size, seq_len_v, d_model]
            mask: [batch_size, 1, seq_len_q, seq_len_k] or None

        Returns:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = q.size(0)

        # Linear projections: [batch_size, seq_len, d_model]
        Q = self.W_q(q)  # [batch_size, seq_len_q, d_model]
        K = self.W_k(k)  # [batch_size, seq_len_k, d_model]
        V = self.W_v(v)  # [batch_size, seq_len_v, d_model]

        # Reshape for multi-head: [batch_size, seq_len, num_heads, d_k] -> [batch_size, num_heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Attention scores: [batch_size, num_heads, seq_len_q, seq_len_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax over key dimension
        attn_weights = F.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values: [batch_size, num_heads, seq_len_q, d_k]
        context = torch.matmul(attn_weights, V)

        # Concatenate heads: [batch_size, seq_len_q, num_heads, d_k] -> [batch_size, seq_len_q, d_model]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # Final linear projection
        output = self.W_o(context)  # [batch_size, seq_len_q, d_model]

        return output, attn_weights


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network (FFN).

    FFN(x) = max(0, xW1 + b1)W2 + b2
    Typically d_ff = 4 * d_model for expansion.
    """

    def __init__(self, d_model: int, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x shape: [batch_size, seq_length, d_model]"""
        x = self.linear1(x)  # [batch_size, seq_length, d_ff]
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)  # [batch_size, seq_length, d_model]
        return x


class TransformerBlock(nn.Module):
    """Single transformer encoder block.

    Structure:
    1. MultiHeadAttention with residual + layer norm
    2. FeedForward with residual + layer norm
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)  # normalization after attention
        self.norm2 = nn.LayerNorm(d_model)  # normalization after FFN

        self.dropout = nn.Dropout(p=dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length, d_model]
            mask: optional attention mask

        Returns:
            output: [batch_size, seq_length, d_model]
        """
        # Self-attention with residual
        attn_output, _ = self.attn(x, x, x, mask)  # [batch_size, seq_length, d_model]
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward with residual
        ffn_output = self.ffn(x)  # [batch_size, seq_length, d_model]
        x = self.norm2(x + self.dropout(ffn_output))

        return x


class SimpleTransformer(nn.Module):
    """Full transformer model with embedding and positional encoding."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_length] token indices
            mask: optional attention mask

        Returns:
            logits: [batch_size, seq_length, vocab_size]
        """
        # Embedding: [batch_size, seq_length, d_model]
        x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        x = self.pos_encoding(x)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Output projection
        x = self.fc_out(x)  # [batch_size, seq_length, vocab_size]
        return x


if __name__ == "__main__":
    batch_size, seq_length, vocab_size = 32, 128, 1024
    d_model = 512

    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    logits = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
