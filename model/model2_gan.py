"""
model2_gan.py
-------------
Model 2 (IN-COURSE, REQUIRED): Conditional GAN with WGAN-GP loss.

GANs were studied in the course. This implementation conditions both
the Generator and Discriminator on the input conditions.

Architecture:
  Generator:
    Input  : condition embeddings (4 × embed_dim = 128) + Gaussian noise (noise_dim)
    Layers : Linear → BatchNorm → LeakyReLU  ×3
    Output : logits (B, seq_len=8, digit_vocab_size=13)

  Discriminator:
    Input  : condition embeddings + flattened date one-hot (8 × 13 = 104)
    Layers : Linear → LeakyReLU  ×3  (NO BatchNorm — required by WGAN-GP)
    Output : scalar Wasserstein score (no sigmoid)

Loss function — WGAN-GP:
  D loss = E[D(fake)] − E[D(real)] + λ·GP
  G loss = −E[D(G(z, cond))]

  Gradient Penalty:
    GP = λ · E[(‖∇_x̂ D(x̂)‖₂ − 1)²]
    where x̂ = α·real + (1−α)·fake,  α ~ Uniform(0,1)

  Why WGAN-GP over vanilla GAN:
    1. Eliminates mode collapse — Wasserstein distance is more meaningful.
    2. Stable training without careful G/D lr balancing.
    3. D loss correlates with generation quality (useful for monitoring).
    4. No vanishing gradients from sigmoid saturation.

  Why a GAN for this problem:
    The one-to-many mapping (many valid dates per condition) is exactly
    what GANs model through the noise vector z. Each z sample can
    produce a different valid date for the same condition.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class GANGenerator(nn.Module):
    """
    Conditional Generator: (conditions, noise) → digit logits.

    Args:
        cond_vocab_size  : size of condition vocabulary (62)
        noise_dim        : Gaussian noise input dimension
        digit_vocab_size : size of digit vocabulary (13)
        seq_len          : number of digit positions (8)
        embed_dim        : per-token condition embedding size
        hidden_dim       : hidden layer width
    """

    def __init__(
        self,
        cond_vocab_size: int,
        noise_dim: int = 64,
        digit_vocab_size: int = 13,
        seq_len: int = 8,
        embed_dim: int = 32,
        hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.seq_len          = seq_len
        self.digit_vocab_size = digit_vocab_size
        self.noise_dim        = noise_dim

        self.embed = nn.Embedding(cond_vocab_size, embed_dim)
        cond_flat  = 4 * embed_dim   # 128

        self.net = nn.Sequential(
            nn.Linear(cond_flat + noise_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 2, seq_len * digit_vocab_size),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, cond: Tensor, noise: Tensor) -> Tensor:
        """
        Args:
            cond  : LongTensor  (B, 4)         — condition indices
            noise : FloatTensor (B, noise_dim)  — sampled from N(0,1)

        Returns:
            logits : FloatTensor (B, seq_len, digit_vocab_size)
        """
        e   = self.embed(cond).view(cond.size(0), -1)
        x   = torch.cat([e, noise], dim=-1)
        out = self.net(x)
        return out.view(-1, self.seq_len, self.digit_vocab_size)

    def sample_noise(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample standard Gaussian noise for a batch."""
        return torch.randn(batch_size, self.noise_dim, device=device)

    @torch.no_grad()
    def generate(self, cond: Tensor) -> Tensor:
        """
        Generate digit indices. Samples fresh noise each call so
        different calls produce different valid dates.

        Args:
            cond : (B, 4) condition indices

        Returns:
            digit_indices : (B, seq_len)
        """
        self.eval()
        noise  = self.sample_noise(cond.size(0), cond.device)
        logits = self.forward(cond, noise)
        return logits.argmax(dim=-1)


class GANDiscriminator(nn.Module):
    """
    Conditional Discriminator: (conditions, date) → Wasserstein score.

    No BatchNorm — required by WGAN-GP because BatchNorm changes the
    gradient structure and invalidates the gradient penalty.
    Uses spectral normalization instead for Lipschitz stability.

    Args:
        cond_vocab_size  : size of condition vocabulary (62)
        digit_vocab_size : size of digit vocabulary (13)
        seq_len          : number of digit positions (8)
        embed_dim        : per-token condition embedding size
        hidden_dim       : hidden layer width
    """

    def __init__(
        self,
        cond_vocab_size: int,
        digit_vocab_size: int = 13,
        seq_len: int = 8,
        embed_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.digit_vocab_size = digit_vocab_size
        self.seq_len          = seq_len

        self.embed  = nn.Embedding(cond_vocab_size, embed_dim)
        cond_flat   = 4 * embed_dim              # 128
        date_flat   = seq_len * digit_vocab_size  # 104

        SN = nn.utils.spectral_norm
        self.net = nn.Sequential(
            SN(nn.Linear(cond_flat + date_flat, hidden_dim)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.LeakyReLU(0.2),
            SN(nn.Linear(hidden_dim // 2, hidden_dim // 4)),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, cond: Tensor, date_onehot: Tensor) -> Tensor:
        """
        Args:
            cond        : LongTensor  (B, 4)
            date_onehot : FloatTensor (B, seq_len, digit_vocab_size)

        Returns:
            scores : FloatTensor (B, 1) — raw Wasserstein scores
        """
        e = self.embed(cond).view(cond.size(0), -1)
        d = date_onehot.view(date_onehot.size(0), -1)
        return self.net(torch.cat([e, d], dim=-1))


# ── WGAN-GP losses ────────────────────────────────────────────────────────────

def gradient_penalty(
    discriminator: GANDiscriminator,
    cond: Tensor,
    real: Tensor,
    fake: Tensor,
    device: torch.device,
    lambda_gp: float = 10.0,
) -> Tensor:
    """
    Compute WGAN-GP gradient penalty.

    Interpolates between real and fake samples, runs D on the
    interpolation, and penalises deviation from unit gradient norm.

    GP = λ · E[(‖∇_x̂ D(x̂)‖₂ − 1)²]

    Args:
        discriminator : Discriminator module
        cond          : (B, 4) condition indices
        real          : (B, seq_len, digit_vocab_size) one-hot real dates
        fake          : (B, seq_len, digit_vocab_size) softmax fake dates
        device        : torch device
        lambda_gp     : penalty coefficient (default 10)

    Returns:
        Scalar GP loss term.
    """
    B     = real.size(0)
    alpha = torch.rand(B, 1, 1, device=device)
    interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)

    d_interp  = discriminator(cond, interpolated)
    gradients = torch.autograd.grad(
        outputs=d_interp,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interp),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(B, -1)
    grad_norm = gradients.norm(2, dim=1)
    return lambda_gp * ((grad_norm - 1.0) ** 2).mean()


def discriminator_loss(d_real: Tensor, d_fake: Tensor, gp: Tensor) -> Tensor:
    """
    WGAN-GP discriminator loss.
    L_D = E[D(fake)] − E[D(real)] + GP
    """
    return d_fake.mean() - d_real.mean() + gp


def generator_loss(d_fake: Tensor) -> Tensor:
    """
    WGAN-GP generator loss.
    L_G = −E[D(G(z))]
    """
    return -d_fake.mean()


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tokenizer import Tokenizer

    tok = Tokenizer()
    G   = GANGenerator(cond_vocab_size=tok.cond_vocab_size)
    D   = GANDiscriminator(cond_vocab_size=tok.cond_vocab_size)

    print(f"Generator     params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"Discriminator params: {sum(p.numel() for p in D.parameters()):,}")

    cond        = torch.randint(0, tok.cond_vocab_size, (4, 4))
    noise       = G.sample_noise(4, torch.device("cpu"))
    fake_logits = G(cond, noise)
    fake_onehot = F.softmax(fake_logits, dim=-1)
    real_digits = torch.randint(0, 10, (4, 8))
    real_onehot = F.one_hot(real_digits, 13).float()

    score = D(cond, real_onehot)
    gp    = gradient_penalty(D, cond, real_onehot, fake_onehot, torch.device("cpu"))

    print(f"G output shape: {fake_logits.shape}")  # (4, 8, 13)
    print(f"D score  shape: {score.shape}")         # (4, 1)
    print(f"GP value      : {gp.item():.4f}")
    print(f"Generate shape: {G.generate(cond).shape}")  # (4, 8)