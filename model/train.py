"""
train.py
--------
Training script for all 4 Dates Generator models.

Models:
  1. model1_ae.py            — Conditional Autoencoder         (IN-COURSE)
  2. model2_gan.py           — Conditional GAN (WGAN-GP)       (IN-COURSE, required)
  3. model3_transformer.py   — Transformer Encoder-Decoder     (OUTSIDE-COURSE)
  4. model4_bilstm_attention — BiLSTM + Bahdanau Attention     (OUTSIDE-COURSE)

Usage:
    python train.py --model ae
    python train.py --model gan
    python train.py --model transformer
    python train.py --model bilstm
    python train.py --model all --epochs 40

Best practices applied:
  - torch.manual_seed for full reproducibility
  - DataLoader shuffle=True on training set
  - 90 / 5 / 5 train / val / test split  (in dataset.py)
  - CSR (Constraint Satisfaction Rate) logged every epoch alongside loss
  - Gradient clipping (max_norm=1.0) to prevent exploding gradients
  - ReduceLROnPlateau scheduler for AE / BiLSTM
  - Warmup + cosine schedule for Transformer
  - Best weights saved to weights/<model>_best.pt  (by val CSR)
  - Loss + CSR history saved to weights/<model>_history.pt for report plots
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from tokenizer import Tokenizer, START_IDX, END_IDX, PAD_IDX
from dataset   import get_dataloaders
from evaluate  import constraint_satisfaction_rate

from model1_ae                import ConditionalAE, ae_loss
from model2_gan               import (
    GANGenerator, GANDiscriminator,
    gradient_penalty, discriminator_loss, generator_loss,
)
from model3_transformer       import TransformerDateModel
from model4_bilstm_attention  import BiLSTMAttentionModel


# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = Path("../data/data.txt")
WEIGHTS_DIR = Path("weights")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE  = 256
SEED        = 42


# ── Utilities ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for full reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_weights(model: nn.Module, name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    path = WEIGHTS_DIR / f"{name}_best.pt"
    torch.save(model.state_dict(), path)
    print(f"    ✓ Saved best weights → {path}")


def save_history(history: Dict[str, List[float]], name: str) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(history, WEIGHTS_DIR / f"{name}_history.pt")


def log_epoch(
    epoch: int,
    total: int,
    train_loss: float,
    val_loss: float,
    val_csr: float,
    elapsed: float,
) -> None:
    print(
        f"  Ep {epoch:03d}/{total} | "
        f"train={train_loss:.4f} | "
        f"val={val_loss:.4f} | "
        f"CSR={val_csr:.3f} | "
        f"{elapsed:.1f}s"
    )


# ── Shared validation helper ──────────────────────────────────────────────────

def _val_pass(
    model: nn.Module,
    val_dl: DataLoader,
    tok: Tokenizer,
    model_name: str,
    criterion: Optional[nn.Module],
) -> Tuple[float, float]:
    """
    Compute val loss and CSR for one epoch.

    Returns:
        (val_loss, val_csr)
        val_loss is 0.0 for GAN (no single loss value).
    """
    model.eval()
    val_losses: List[float] = []
    all_preds:  List[str]   = []
    all_conds:  List[List[str]] = []

    with torch.no_grad():
        for cond, date in val_dl:
            cond, date = cond.to(DEVICE), date.to(DEVICE)
            B = cond.size(0)

            if model_name == "ae":
                logits = model(cond)                         # (B,8,vocab)
                tgt    = date[:, 1:9]
                if criterion:
                    B2, S, V = logits.shape
                    val_losses.append(
                        criterion(logits.view(B2*S, V), tgt.reshape(-1)).item()
                    )
                preds = logits.argmax(dim=-1)                # (B,8)
                for i in range(B):
                    d = tok.decode_date([START_IDX] + preds[i].tolist() + [END_IDX])
                    all_preds.append(d or "")

            elif model_name == "transformer":
                logits = model(cond, date)                   # (B,seq-1,vocab)
                B2, S, V = logits.shape
                if criterion:
                    val_losses.append(
                        criterion(logits.view(B2*S, V), date[:, 1:].reshape(-1)).item()
                    )
                preds = logits.argmax(dim=-1)                # (B,seq-1)
                for i in range(B):
                    d = tok.decode_date([START_IDX] + preds[i].tolist())
                    all_preds.append(d or "")

            elif model_name == "bilstm":
                logits = model(cond, date, teacher_forcing_ratio=0.0)
                B2, S, V = logits.shape
                if criterion:
                    val_losses.append(
                        criterion(logits.view(B2*S, V), date[:, 1:].reshape(-1)).item()
                    )
                preds = logits.argmax(dim=-1)
                for i in range(B):
                    d = tok.decode_date([START_IDX] + preds[i].tolist())
                    all_preds.append(d or "")

            elif model_name == "gan":
                noise  = model.sample_noise(B, DEVICE)
                logits = model(cond, noise)                  # (B,8,vocab)
                preds  = logits.argmax(dim=-1)
                for i in range(B):
                    d = tok.decode_date([START_IDX] + preds[i].tolist() + [END_IDX])
                    all_preds.append(d or "")

            for i in range(B):
                all_conds.append(tok.decode_conditions(cond[i].tolist()))

    val_loss = sum(val_losses) / max(len(val_losses), 1)
    val_csr  = constraint_satisfaction_rate(all_preds, all_conds)
    return val_loss, val_csr


# ── Model 1: Conditional AE ───────────────────────────────────────────────────

def train_ae(
    train_dl: DataLoader,
    val_dl: DataLoader,
    tok: Tokenizer,
    epochs: int = 40,
) -> ConditionalAE:
    print("\n=== Training Model 1: Conditional AE (in-course) ===")
    model = ConditionalAE(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    ).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss()

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_csr": []
    }
    best_csr = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses: List[float] = []

        for cond, date in train_dl:
            cond, date = cond.to(DEVICE), date.to(DEVICE)
            tgt    = date[:, 1:9]          # 8 digit positions (skip <S>)
            logits = model(cond)           # (B, 8, vocab)
            loss   = ae_loss(logits, tgt)

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_losses.append(loss.item())

        val_loss, val_csr = _val_pass(model, val_dl, tok, "ae", criterion)
        scheduler.step(val_loss)

        avg = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg)
        history["val_loss"].append(val_loss)
        history["val_csr"].append(val_csr)
        log_epoch(epoch, epochs, avg, val_loss, val_csr, time.time() - t0)

        if val_csr > best_csr:
            best_csr = val_csr
            save_weights(model, "ae")

    save_history(history, "ae")
    print(f"  Best val CSR: {best_csr:.3f}")
    return model


# ── Model 2: cGAN (WGAN-GP) ───────────────────────────────────────────────────

def train_gan(
    train_dl: DataLoader,
    val_dl: DataLoader,
    tok: Tokenizer,
    epochs: int = 40,
    n_critic: int = 5,
) -> GANGenerator:
    print("\n=== Training Model 2: cGAN WGAN-GP (in-course, required) ===")

    G = GANGenerator(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
        seq_len=8,
    ).to(DEVICE)

    D = GANDiscriminator(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
        seq_len=8,
    ).to(DEVICE)

    # Adam with WGAN-recommended betas
    opt_G = torch.optim.Adam(G.parameters(), lr=1e-4, betas=(0.0, 0.9))
    opt_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.0, 0.9))

    history: Dict[str, List[float]] = {
        "g_loss": [], "d_loss": [], "val_csr": []
    }
    best_csr = -1.0
    step     = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        G.train(); D.train()
        g_losses: List[float] = []
        d_losses: List[float] = []

        for cond, date in train_dl:
            cond, date = cond.to(DEVICE), date.to(DEVICE)
            B = cond.size(0)

            # Real dates as one-hot over 8 digit positions
            real_digits = date[:, 1:9]
            real_onehot = F.one_hot(real_digits, tok.digit_vocab_size).float()

            # ── Discriminator step ────────────────────────────────────────
            noise       = G.sample_noise(B, DEVICE)
            fake_logits = G(cond, noise).detach()
            fake_onehot = F.softmax(fake_logits, dim=-1)

            gp     = gradient_penalty(D, cond, real_onehot, fake_onehot, DEVICE)
            d_real = D(cond, real_onehot)
            d_fake = D(cond, fake_onehot)
            d_loss = discriminator_loss(d_real, d_fake, gp)

            opt_D.zero_grad()
            d_loss.backward()
            opt_D.step()
            d_losses.append(d_loss.item())
            step += 1

            # ── Generator step (every n_critic D steps) ───────────────────
            if step % n_critic == 0:
                noise    = G.sample_noise(B, DEVICE)
                fake_out = G(cond, noise)
                fake_oh  = F.softmax(fake_out, dim=-1)
                g_loss   = generator_loss(D(cond, fake_oh))

                opt_G.zero_grad()
                g_loss.backward()
                opt_G.step()
                g_losses.append(g_loss.item())

        _, val_csr = _val_pass(G, val_dl, tok, "gan", criterion=None)
        avg_g = sum(g_losses) / max(len(g_losses), 1)
        avg_d = sum(d_losses) / len(d_losses)

        history["g_loss"].append(avg_g)
        history["d_loss"].append(avg_d)
        history["val_csr"].append(val_csr)

        print(
            f"  Ep {epoch:03d}/{epochs} | "
            f"G={avg_g:.4f} | D={avg_d:.4f} | "
            f"CSR={val_csr:.3f} | {time.time()-t0:.1f}s"
        )

        if val_csr > best_csr:
            best_csr = val_csr
            save_weights(G, "gan_generator")
            save_weights(D, "gan_discriminator")

    save_history(history, "gan")
    print(f"  Best val CSR: {best_csr:.3f}")
    return G


# ── Model 3: Transformer ──────────────────────────────────────────────────────

def train_transformer(
    train_dl: DataLoader,
    val_dl: DataLoader,
    tok: Tokenizer,
    epochs: int = 40,
) -> TransformerDateModel:
    print("\n=== Training Model 3: Transformer Encoder-Decoder (outside-course) ===")

    model = TransformerDateModel(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    ).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.98), eps=1e-9)

    # Linear warmup → cosine decay
    warmup_steps = 500
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.1, 0.5 * (1 + torch.cos(torch.tensor(
            (step - warmup_steps) / max(1, (epochs * len(train_dl) - warmup_steps)) * 3.14159
        )).item()))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_csr": []
    }
    best_csr = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses: List[float] = []

        for cond, date in train_dl:
            cond, date = cond.to(DEVICE), date.to(DEVICE)
            logits = model(cond, date)               # (B, seq-1, vocab)
            B, S, V = logits.shape
            loss = criterion(logits.view(B*S, V), date[:, 1:].reshape(-1))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            scheduler.step()
            train_losses.append(loss.item())

        val_loss, val_csr = _val_pass(model, val_dl, tok, "transformer", criterion)
        avg = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg)
        history["val_loss"].append(val_loss)
        history["val_csr"].append(val_csr)
        log_epoch(epoch, epochs, avg, val_loss, val_csr, time.time() - t0)

        if val_csr > best_csr:
            best_csr = val_csr
            save_weights(model, "transformer")

    save_history(history, "transformer")
    print(f"  Best val CSR: {best_csr:.3f}")
    return model


# ── Model 4: BiLSTM + Attention ───────────────────────────────────────────────

def train_bilstm(
    train_dl: DataLoader,
    val_dl: DataLoader,
    tok: Tokenizer,
    epochs: int = 40,
) -> BiLSTMAttentionModel:
    print("\n=== Training Model 4: BiLSTM + Attention (outside-course) ===")

    model = BiLSTMAttentionModel(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    ).to(DEVICE)

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [], "val_csr": []
    }
    best_csr = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_losses: List[float] = []

        # Teacher forcing: decay from 0.8 → 0.2 over training
        tf_ratio = max(0.2, 0.8 - (epoch / epochs) * 0.6)

        for cond, date in train_dl:
            cond, date = cond.to(DEVICE), date.to(DEVICE)
            logits = model(cond, date, teacher_forcing_ratio=tf_ratio)
            B, S, V = logits.shape
            loss = criterion(logits.view(B*S, V), date[:, 1:].reshape(-1))

            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimiser.step()
            train_losses.append(loss.item())

        val_loss, val_csr = _val_pass(model, val_dl, tok, "bilstm", criterion)
        scheduler.step(val_loss)

        avg = sum(train_losses) / len(train_losses)
        history["train_loss"].append(avg)
        history["val_loss"].append(val_loss)
        history["val_csr"].append(val_csr)
        log_epoch(epoch, epochs, avg, val_loss, val_csr, time.time() - t0)

        if val_csr > best_csr:
            best_csr = val_csr
            save_weights(model, "bilstm")

    save_history(history, "bilstm")
    print(f"  Best val CSR: {best_csr:.3f}")
    return model


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Dates Generator models")
    parser.add_argument(
        "--model",
        choices=["ae", "gan", "transformer", "bilstm", "all"],
        default="all",
        help="Which model to train",
    )
    parser.add_argument("--epochs", type=int,  default=40,         help="Training epochs")
    parser.add_argument("--batch",  type=int,  default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed",   type=int,  default=SEED,       help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)
    print(f"Device : {DEVICE}")
    print(f"Seed   : {args.seed}")
    print(f"Epochs : {args.epochs}")
    print(f"Batch  : {args.batch}")

    tok = Tokenizer()
    train_dl, val_dl, test_dl = get_dataloaders(DATA_PATH, tok, args.batch, args.seed)

    targets = (
        ["ae", "gan", "transformer", "bilstm"]
        if args.model == "all"
        else [args.model]
    )

    for target in targets:
        if   target == "ae":          train_ae(train_dl, val_dl, tok, args.epochs)
        elif target == "gan":         train_gan(train_dl, val_dl, tok, args.epochs)
        elif target == "transformer": train_transformer(train_dl, val_dl, tok, args.epochs)
        elif target == "bilstm":      train_bilstm(train_dl, val_dl, tok, args.epochs)


if __name__ == "__main__":
    main()