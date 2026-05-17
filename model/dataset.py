"""
dataset.py
----------
PyTorch Dataset for the Dates Generator problem.

Reads data.txt, encodes every line via the Tokenizer, and provides
train / validation / test splits (90 / 5 / 5).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List
from pathlib import Path

from tokenizer import Tokenizer, DATE_SEQ_LEN


class DatesDataset(Dataset):
    """
    Each item is a tuple of:
        cond  : LongTensor of shape (4,)  — encoded condition tokens
        date  : LongTensor of shape (10,) — encoded date digits incl. <S>/<E>
    """

    def __init__(self, filepath: str | Path, tokenizer: Tokenizer) -> None:
        """
        Args:
            filepath  : path to data.txt
            tokenizer : Tokenizer instance
        """
        self.tokenizer = tokenizer
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self._load(Path(filepath))

    def _load(self, path: Path) -> None:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue  # skip malformed lines

                date_str = parts[4]
                cond_indices = self.tokenizer.encode_conditions(line)
                date_indices = self.tokenizer.encode_date(date_str)

                self.samples.append((
                    torch.tensor(cond_indices, dtype=torch.long),
                    torch.tensor(date_indices, dtype=torch.long),
                ))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


def get_splits(
    filepath: str | Path,
    tokenizer: Tokenizer,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train (90%), val (5%), test (5%).

    Args:
        filepath  : path to data.txt
        tokenizer : Tokenizer instance
        seed      : random seed for reproducibility

    Returns:
        (train_dataset, val_dataset, test_dataset)
    """
    full_ds = DatesDataset(filepath, tokenizer)
    n = len(full_ds)
    n_val  = int(0.05 * n)
    n_test = int(0.05 * n)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_ds, [n_train, n_val, n_test], generator=generator
    )
    print(f"Dataset split — train: {n_train}, val: {n_val}, test: {n_test}")
    return train_ds, val_ds, test_ds


def get_dataloaders(
    filepath: str | Path,
    tokenizer: Tokenizer,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Return train / val / test DataLoaders.

    Train loader shuffles every epoch (data imbalance mitigation baseline).
    Val and test loaders do not shuffle.
    """
    train_ds, val_ds, test_ds = get_splits(filepath, tokenizer, seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    return train_loader, val_loader, test_loader


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = Tokenizer()
    train_dl, val_dl, test_dl = get_dataloaders("../data/data.txt", tok, batch_size=4)
    cond_batch, date_batch = next(iter(train_dl))
    print("Cond batch shape:", cond_batch.shape)   # (4, 4)
    print("Date batch shape:", date_batch.shape)   # (4, 10)
    print("Sample cond decoded:", tok.decode_conditions(cond_batch[0].tolist()))
    print("Sample date decoded:", tok.decode_date(date_batch[0].tolist()))