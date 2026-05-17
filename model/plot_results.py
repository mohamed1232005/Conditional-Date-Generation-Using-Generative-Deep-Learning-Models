import torch
import matplotlib.pyplot as plt
from pathlib import Path

for name in ["ae", "transformer", "bilstm"]:
    path = Path(f"weights/{name}_history.pt")
    if not path.exists():
        print(f"Skipping {name} — not found")
        continue

    h = torch.load(path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h["train_loss"], label="train loss")
    ax1.plot(h["val_loss"], label="val loss")
    ax1.set_title(f"{name.upper()} — Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(h["val_csr"], label="val CSR")
    ax2.set_title(f"{name.upper()} — CSR")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig(f"weights/{name}_curves.png", dpi=150)
    plt.close()

    print(f"Saved {name}_curves.png")

gan_path = Path("weights/gan_history.pt")

if gan_path.exists():
    h = torch.load(gan_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(h["g_loss"], label="G loss")
    ax1.plot(h["d_loss"], label="D loss")
    ax1.set_title("GAN — Generator & Discriminator Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(h["val_csr"], label="val CSR")
    ax2.set_title("GAN — CSR")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("weights/gan_curves.png", dpi=150)
    plt.close()

    print("Saved gan_curves.png")