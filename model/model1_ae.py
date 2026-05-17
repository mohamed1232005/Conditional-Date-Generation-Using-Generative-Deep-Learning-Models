"""
model1_ae.py
------------
Model 1 (IN-COURSE): Conditional Autoencoder (cAE).

Autoencoders were studied in the course as deterministic encoder-decoder
architectures that learn a compressed latent representation.

Architecture:
  Encoder:
    Input  : 4 condition token embeddings → flattened
    Layers : Linear(128→256) → ReLU → Dropout
             Linear(256→128) → ReLU → Dropout
             Linear(128→latent_dim)
    Output : deterministic latent vector z  (no sampling)

  Decoder:
    Input  : z concatenated with condition embeddings (skip connection)
             This skip connection is critical — it gives the decoder
             direct access to the original conditions, preventing
             information loss through the bottleneck.
    Layers : Linear(latent+cond→512) → ReLU → Dropout
             Linear(512→256)         → ReLU → Dropout
             Linear(256→seq_len * digit_vocab_size)
    Output : logits (B, 8, digit_vocab_size) for 8 digit positions

Loss function:
  CrossEntropyLoss over all 8 digit positions:
  L = (1/8) * Σ_t CE(logits_t, target_t)

  No KL term — this is a plain AE. The latent space is deterministic,
  so the same condition always maps to the same z, and therefore to
  the same (most frequent) valid date in the training set.

Key limitation vs GAN/VAE (discussed in report):
  A plain AE cannot model one-to-many mappings. For a given condition,
  it will always output the same date. This is the core motivation for
  the GAN and VAE models which introduce stochasticity.

Why include it:
  - Directly demonstrates the AE concept from the course.
  - Provides a clean deterministic baseline for comparison.
  - Comparison with Model 4 (VAE) clearly shows the benefit of
    the probabilistic latent space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class AEEncoder(nn.Module):
    """
    Deterministic encoder: conditions → latent vector z.

    Args:
        cond_vocab_size : size of the condition vocabulary (62)
        embed_dim       : per-token embedding dimension
        hidden_dim      : hidden layer width
        latent_dim      : bottleneck dimension
        dropout         : dropout probability
    """

    def __init__(
        self,
        cond_vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.embed = nn.Embedding(cond_vocab_size, embed_dim)
        cond_flat  = 4 * embed_dim   # 4 tokens × embed_dim = 128

        self.net = nn.Sequential(
            nn.Linear(cond_flat, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            cond : LongTensor (B, 4) — condition token indices

        Returns:
            z    : FloatTensor (B, latent_dim)   — latent vector
            e    : FloatTensor (B, 4*embed_dim)  — flattened embeddings
                   returned for the decoder skip connection
        """
        e = self.embed(cond).view(cond.size(0), -1)   # (B, 4*embed_dim)
        z = self.net(e)                                # (B, latent_dim)
        return z, e


class AEDecoder(nn.Module):
    """
    Decoder: (z + condition embeddings) → digit logits.

    Concatenating condition embeddings with z (skip connection) gives
    the decoder direct access to the original conditions, compensating
    for any information lost through the bottleneck.

    Args:
        digit_vocab_size : size of the digit vocabulary (13)
        seq_len          : number of digit positions to predict (8)
        latent_dim       : dimension of the latent vector
        cond_embed_flat  : size of flattened condition embeddings (4 * embed_dim)
        hidden_dim       : hidden layer width
        dropout          : dropout probability
    """

    def __init__(
        self,
        digit_vocab_size: int,
        seq_len: int = 8,
        latent_dim: int = 64,
        cond_embed_flat: int = 128,   # 4 * 32
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.seq_len         = seq_len
        self.digit_vocab_size = digit_vocab_size

        inp_dim = latent_dim + cond_embed_flat   # skip connection

        self.net = nn.Sequential(
            nn.Linear(inp_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, seq_len * digit_vocab_size),
        )

    def forward(self, z: Tensor, cond_embed: Tensor) -> Tensor:
        """
        Args:
            z          : (B, latent_dim)       — encoder output
            cond_embed : (B, 4*embed_dim)      — raw condition embeddings

        Returns:
            logits : (B, seq_len, digit_vocab_size)
        """
        h   = torch.cat([z, cond_embed], dim=-1)          # skip connection
        out = self.net(h)                                  # (B, seq*vocab)
        return out.view(-1, self.seq_len, self.digit_vocab_size)


class ConditionalAE(nn.Module):
    """
    Full Conditional Autoencoder.

    Combines AEEncoder + AEDecoder. During training, the encoder
    compresses the conditions to z, and the decoder reconstructs
    the target date from z + conditions.

    During inference, the forward pass is used directly — no sampling.

    Args:
        cond_vocab_size  : size of condition vocabulary (62)
        digit_vocab_size : size of digit vocabulary (13)
        embed_dim        : per-token condition embedding size
        hidden_dim       : hidden layer width
        latent_dim       : bottleneck dimension
        dropout          : dropout probability
    """

    def __init__(
        self,
        cond_vocab_size: int,
        digit_vocab_size: int,
        embed_dim: int = 32,
        hidden_dim: int = 256,
        latent_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.encoder = AEEncoder(
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        )
        self.decoder = AEDecoder(
            digit_vocab_size=digit_vocab_size,
            seq_len=8,
            latent_dim=latent_dim,
            cond_embed_flat=4 * embed_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming uniform init for all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, cond: Tensor) -> Tensor:
        """
        Full encode → decode pass.

        Args:
            cond : (B, 4) condition indices

        Returns:
            logits : (B, 8, digit_vocab_size)
        """
        z, e   = self.encoder(cond)
        logits = self.decoder(z, e)
        return logits

    @torch.no_grad()
    def generate(self, cond: Tensor) -> Tensor:
        """
        Generate digit indices for a batch of conditions.
        Deterministic — always returns the same output for the same input.

        Args:
            cond : (B, 4) condition indices

        Returns:
            digit_indices : (B, 8) — argmax over logits
        """
        self.eval()
        logits = self.forward(cond)          # (B, 8, vocab)
        return logits.argmax(dim=-1)         # (B, 8)


# ── Loss function ─────────────────────────────────────────────────────────────

def ae_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """
    CrossEntropyLoss over all 8 digit positions.

    Args:
        logits  : (B, 8, digit_vocab_size)
        targets : (B, 8) — ground-truth digit indices

    Returns:
        Scalar loss.
    """
    B, S, V = logits.shape
    return F.cross_entropy(
        logits.reshape(B * S, V),
        targets.reshape(B * S),
    )


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tokenizer import Tokenizer

    tok   = Tokenizer()
    model = ConditionalAE(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    )
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Dummy forward pass
    cond   = torch.randint(0, tok.cond_vocab_size, (4, 4))
    logits = model(cond)
    print(f"Input  shape: {cond.shape}")
    print(f"Output shape: {logits.shape}")   # (4, 8, 13)

    # Dummy generate
    preds = model.generate(cond)
    print(f"Generate shape: {preds.shape}")  # (4, 8)