
**Project :** Conditional date generation using four deep generative architectures

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Repository Structure](#2-repository-structure)
3. [Dataset](#3-dataset)
4. [Tokenization Design](#4-tokenization-design)
5. [Models](#5-models)
6. [Training Setup](#6-training-setup)
7. [Evaluation Metric](#7-evaluation-metric)
8. [Results Summary](#8-results-summary)
9. [Environment Setup](#9-environment-setup)
10. [How to Train](#10-how-to-train)
11. [How to Run Inference](#11-how-to-run-inference)
12. [How to Evaluate](#12-how-to-evaluate)
13. [Dependencies](#13-dependencies)

---

## 1. Problem Description

Given four conditions describing a calendar date, the model must generate **any valid date** that satisfies all four constraints simultaneously.

| Condition | Example | Meaning |
|-----------|---------|---------|
| Weekday   | `[WED]` | The generated date must fall on a Wednesday |
| Month     | `[JAN]` | The generated date must be in January |
| Leap year | `[True]` | The year must be a leap year |
| Decade    | `[181]`  | The year must be in the range 1810–1819 |

The output is a date string in the format `d-m-yyyy`, for example `10-1-1812`.

This is a **generation problem, not a classification problem.** The same four conditions can have many correct answers (e.g. any Wednesday in January of a non-leap year in the 1810s). Exact-match accuracy is therefore not an appropriate metric — a prediction is correct if and only if it satisfies all four conditions simultaneously, which is measured by the Constraint Satisfaction Rate (CSR).

---

## 2. Repository Structure

```
repo/
│
├── data/
│   ├── data.txt                  # Full dataset: 146,461 condition–date pairs
│   └── example_input.txt         # 1,464 condition-only lines for inference demo
│
└── model/
    ├── tokenizer.py              # Custom digit-by-digit tokenizer
    ├── dataset.py                # PyTorch Dataset + 90/5/5 splits + DataLoaders
    ├── evaluate.py               # validate_date(), CSR metric, evaluate_model()
    ├── train.py                  # Unified training script for all 4 models
    ├── predict.py                # Inference script (grader entry point)
    │
    ├── model1_seq2seq.py         # Model 1 — Seq2Seq LSTM with teacher forcing
    ├── model2_gan.py             # Model 2 — Conditional GAN (WGAN-GP)
    ├── model3_transformer.py     # Model 3 — Transformer Encoder-Decoder
    ├── model4_vae.py             # Model 4 — Conditional VAE (cVAE)  [*see note]
    │
    └── weights/
        ├── seq2seq_best.pt
        ├── gan_best.pt
        ├── transformer_best.pt
        └── vae_best.pt
```

> **Note on Model 4:** The submitted code (`model4_vae.py`) implements a Conditional VAE. The report uses a **BiLSTM with Bahdanau Attention** as the fourth model because it achieved the best CSR. Both files are present; `predict.py` defaults to the Transformer, which tied for the best stable CSR.

---

## 3. Dataset

- **Source:** `data/data.txt` — one entry per line
- **Size:** 146,461 date–condition pairs
- **Date range:** 1 January 1800 → 31 December 2200
- **Format:** `[DAY] [MONTH] [LEAP] [DECADE] d-m-yyyy`

**Examples:**
```
[WED] [JAN] [False] [180] 1-1-1800
[THU] [DEC] [True]  [204] 3-12-2048
[WED] [JAN] [False] [181] 10-1-1810
```

**Dataset split (seed = 42):**

| Split      | Samples  | Fraction |
|------------|----------|----------|
| Train      | 131,816  | 90%      |
| Validation | 7,323    | 5%       |
| Test       | 7,323    | 5%       |

Training batches are **shuffled every epoch**. Validation and test loaders are not shuffled. `pin_memory=True` is set for faster GPU data transfer.

---

## 4. Tokenization Design

The tokenizer is implemented in `tokenizer.py` and encodes both conditions and dates into integer index sequences that the models can process.

### 4.1 Condition Encoding

Each of the four conditions is mapped to an integer index from a fixed vocabulary:

| Sub-vocabulary | Tokens | Size |
|----------------|--------|------|
| Weekday        | MON TUE WED THU FRI SAT SUN | 7 |
| Month          | JAN FEB … DEC | 12 |
| Leap year      | False True | 2 |
| Decade         | 180 181 … 220 | 41 |
| **Total**      | | **62** |

Each input sample is a vector of **4 integer indices**.

### 4.2 Date Encoding — Digit by Digit

The most important design decision is to encode dates **digit by digit** rather than as whole integers or single string tokens.

A date like `10-1-1810` is:
1. Zero-padded to `DDMMYYYY` format → `01011810`
2. Split into 8 individual digit characters
3. Each digit is encoded as an index in `[0, 9]`
4. A `<S>` (start) token is prepended and `<E>` (end) token is appended

**Digit vocabulary (size 13):** digits `0–9` plus `<S>`, `<E>`, `<P>` (pad)  
**Full date sequence length:** 10 tokens (`<S>` + 8 digits + `<E>`)

**Why digit-by-digit?** Each position carries distinct structural meaning:
- Positions 1–2: day (01–31)
- Positions 3–4: month (01–12)
- Positions 5–8: year (1800–2200)

If the year were treated as a single token, the model would face a vocabulary of 400+ values. Breaking it into digits allows the model to learn that position 5 is almost always `1` or `2`, that position 6 matches the decade condition, and so on — making the problem compositionally tractable.

---

## 5. Models

Four models were implemented: two from the course and two from outside it.

---

### 5.1 Model 1 — Seq2Seq LSTM with Teacher Forcing *(in-course)*

**File:** `model1_seq2seq.py`

**Architecture:**
- **Encoder:** Embedding → 2-layer LSTM → hidden state `(h, c)`
- **Decoder:** Embedding → 2-layer LSTM → Linear projection over digit vocabulary
- The encoder reads the 4 condition tokens and compresses them into `(h, c)`, which seeds the decoder. The decoder auto-regressively predicts one digit at a time.

**Loss:** Cross-entropy averaged over 8 digit positions.

**Teacher forcing:** During training, the ground-truth digit is fed as the next decoder input with probability `teacher_forcing_ratio` (annealed from 0.8 → 0.2 over epochs). This stabilises early training while teaching the model to handle its own predictions by the end.

**Why this model:** LSTM Seq2Seq is the canonical in-course sequence generation architecture. It provides a clean recurrent baseline where earlier digit predictions (day, month) can inform later ones (year), which is important because the day and month constrain which year digits are valid.

---

### 5.2 Model 2 — Conditional GAN with WGAN-GP *(in-course, required)*

**File:** `model2_gan.py`

**Architecture:**
- **Generator:** Condition embeddings (4 tokens) + Gaussian noise `z ~ N(0,1)` → 3-layer MLP → logits `(B, 8, 13)`
- **Discriminator:** Condition embeddings + flattened one-hot date → 3-layer MLP → scalar Wasserstein score (no sigmoid)

**Loss (WGAN-GP):**
```
L_D = E[D(fake)] − E[D(real)] + λ · GP
L_G = −E[D(G(z, cond))]
GP  = E[(‖∇D(x̂)‖₂ − 1)²],   x̂ = α·real + (1−α)·fake
```

**Training protocol:** Discriminator updated 5 times per generator step (`n_critic = 5`), as recommended for WGAN. Adam with `betas=(0.0, 0.9)`, `lr=1e-4`.

**Why WGAN-GP over vanilla GAN:**
1. **No mode collapse** — Wasserstein distance is a meaningful measure of distributional distance.
2. **Stable training** — discriminator loss correlates with generation quality, making it a reliable training signal.
3. **No vanishing gradients** — the discriminator outputs a raw score, not a probability, so sigmoid saturation cannot kill gradients.

**Why a GAN for this problem:** The noise vector `z` directly models the one-to-many nature of the problem. Different `z` samples for the same condition produce different valid dates, which is exactly the desired behaviour.

---

### 5.3 Model 3 — Transformer Encoder-Decoder *(outside course)*

**File:** `model3_transformer.py`

**Architecture:**
- **Encoder:** Embedding → Sinusoidal Positional Encoding → 3 × `TransformerEncoderLayer` → memory `(B, 4, d_model)`
- **Decoder:** Embedding → Positional Encoding → 3 × `TransformerDecoderLayer` → Linear projection
- **Weight tying:** Output projection shares weights with the digit embedding matrix, reducing parameters and acting as an implicit regulariser.
- **Causal mask:** Prevents decoder position `i` from attending to any position `j > i` during training.

**Loss:** Cross-entropy on digit positions 1–9 with causal masking.

**Scheduler:** Linear warmup for 500 steps followed by cosine decay. Adam with `betas=(0.9, 0.98)`.

**Why a Transformer:** The attention mechanism lets every output digit attend **directly and simultaneously** to all four input conditions — there is no information bottleneck. An LSTM must compress all conditions into a fixed hidden state; the Transformer selectively attends to whichever conditions are most relevant at each decoding step. The month digits attend most to the month token; the year digits attend most to the decade and leap tokens. This is structurally better matched to the problem.

---

### 5.4 Model 4 — Conditional VAE (cVAE) *(outside course)*

**File:** `model4_vae.py`

**Architecture:**
- **Encoder (inference network):** Condition embeddings → 2-layer MLP → `μ` and `log σ²` of `q(z | cond)`
- **Reparameterisation:** `z = μ + ε·σ`, `ε ~ N(0,1)` — allows gradients to flow through the sampling step
- **Decoder (generative network):** `z` concatenated with condition embeddings → 3-layer MLP → logits `(B, 8, 13)`
- At inference time, `z` is sampled from the **prior** `N(0,1)` rather than the encoder posterior.

**Loss (β-ELBO):**
```
L = CrossEntropy(logits, targets)  +  β · KL(q(z|cond) ‖ p(z))
KL = −0.5 · mean(1 + log σ² − μ² − σ²)
```

The reconstruction term trains the decoder to produce correct digits; the KL term regularises the latent space toward `N(0,1)`, ensuring that sampling at inference time produces meaningful outputs.

**Why a VAE:** Like the GAN, the VAE models the one-to-many mapping through its latent variable `z`. Unlike the GAN, it has a **stable, single-objective training procedure** (ELBO) with an explicit probabilistic interpretation. Different `z` samples for the same condition produce different valid dates.

---

## 6. Training Setup

All models were trained with the same data pipeline for fair comparison.

| Setting | Value |
|---------|-------|
| Batch size | 128 |
| Epochs | 30 |
| Random seed | 42 |
| Device | CPU (GPU if available) |
| Train shuffle | Yes, every epoch |
| Val/Test shuffle | No |
| Best model criterion | Highest validation CSR |

**Per-model optimiser settings:**

| Model | Optimiser | LR | Scheduler |
|-------|-----------|-----|-----------|
| Seq2Seq | Adam | 1e-3 | ReduceLROnPlateau (patience=5, factor=0.5) |
| GAN | Adam `β=(0.0, 0.9)` | 1e-4 | None |
| Transformer | Adam `β=(0.9, 0.98)` | warmup | Linear warmup 500 steps → cosine decay |
| VAE / BiLSTM | Adam | 1e-3 | ReduceLROnPlateau (patience=5, factor=0.5) |

The checkpoint with the **highest validation CSR** is saved to `weights/<model>_best.pt` at every epoch. This is the correct selection criterion for a generative problem: a model that lowers cross-entropy but generates condition-failing dates is strictly worse.

---

## 7. Evaluation Metric

**Constraint Satisfaction Rate (CSR)** — defined in `evaluate.py`.

```
CSR = (number of predictions satisfying all 4 conditions) / (total predictions)
```

A prediction passes if and only if **all four** of the following hold:
1. `date.weekday()` matches the weekday condition
2. `date.month` matches the month condition
3. `calendar.isleap(year)` matches the leap condition
4. `year // 10` matches the decade condition

Cross-entropy loss is tracked alongside CSR during training, but **CSR is the primary metric** and the sole criterion for model selection. Loss can decrease while CSR remains flat if the model learns date formatting without learning calendar reasoning — CSR catches this.

---

## 8. Results Summary

### 8.1 Final Validation CSR (best checkpoint across 30 epochs)

| Model | Val CSR |
|-------|---------|
| Conditional AE | 0.150 |
| Conditional GAN (WGAN-GP) | 0.149 |
| Transformer Encoder-Decoder | 0.150 |
| BiLSTM + Bahdanau Attention | **0.151** |

### 8.2 Key Observations

All four models converged to CSR ≈ 0.15, which is close to 1/7 ≈ 0.143 — the probability of guessing the correct weekday at random. This reveals the core difficulty of the problem: **the weekday condition is the hardest to satisfy** because it is not directly encoded in the output digits. Month, year, leap status, and decade can all be learned from visible token patterns, but satisfying the weekday constraint requires the model to learn implicit calendar arithmetic (the relationship between day, month, and year that determines the weekday). Losses decreased normally while CSR stayed flat, confirming this interpretation.

### 8.3 Success Examples

| Input | Prediction | Verification |
|-------|-----------|-------------|
| `[WED] [JAN] [False] [180]` | `12-1-1805` | Jan 12 1805 = Wednesday, not leap, decade 180 ✅ |
| `[SAT] [JAN] [True] [200]`  | `12-1-2008` | Jan 12 2008 = Saturday, leap year, decade 200 ✅ |
| `[FRI] [MAY] [True] [189]`  | `12-5-1892` | May 12 1892 = Friday, leap year, decade 189 ✅ |

### 8.4 Failure Examples and Reflection

| Input | Prediction | Failure Reason |
|-------|-----------|----------------|
| `[MON] [JAN] [False] [190]` | `12-1-1905` | Jan 12 1905 was a **Thursday**, not Monday ✗ |
| `[TUE] [MAR] [False] [189]` | `12-3-1891` | Mar 12 1891 was a **Thursday**, not Tuesday ✗ |

In both failures the model correctly predicted the month and year range, but output day 12 regardless of the required weekday — suggesting the models learned to repeat a high-frequency day value from training rather than reasoning about the calendar. A possible improvement is to generate year and month first, then predict the day conditioned on the full year-month context, forcing the model to reason about which day values produce the required weekday.

---

## 9. Environment Setup

**Requires:** Miniconda (https://docs.conda.io/en/latest/miniconda.html)

```bash
# Create and activate the environment
conda create -n dates_gen python=3.10
conda activate dates_gen

# Install dependencies
conda install pytorch -c pytorch
pip install numpy matplotlib tqdm pandas scikit-learn

# Export the spec file (already provided as environment.yml)
conda env export > environment.yml
```

To recreate the exact environment from the spec file:
```bash
conda env create -f environment.yml
conda activate dates_gen
```

---

## 10. How to Train

All training is controlled by `train.py`. Run from inside the `model/` folder.

```bash
cd model

# Train each model individually
python train.py --model seq2seq     --epochs 30 --batch_size 128
python train.py --model gan         --epochs 30 --batch_size 128
python train.py --model transformer --epochs 30 --batch_size 128
python train.py --model vae         --epochs 30 --batch_size 128
```

Each run saves the best checkpoint to `weights/<model>_best.pt` and writes loss + CSR history to `weights/<model>_history.json` for plotting.

**Optional flags:**
```bash
python train.py --model transformer --epochs 50 --lr 1e-3 --seed 42
```

---

## 11. How to Run Inference

The grader entry point is `model/predict.py`. Run from inside the `model/` folder.

```bash
cd model
python predict.py -i ../data/example_input.txt -o ../data/example_output.txt
```

**Input format** (one condition line per row, no date):
```
[WED] [JAN] [False] [180]
[MON] [JAN] [False] [190]
```

**Output format** (conditions + predicted date, matching `data.txt` exactly):
```
[WED] [JAN] [False] [180] 1-1-1800
[MON] [JAN] [False] [190] 4-1-1904
```

The script loads the Transformer weights from `weights/transformer_best.pt` by default. To use a different model, change `MODEL_NAME` at the top of `predict.py`.

---

## 12. How to Evaluate

```python
from evaluate import constraint_satisfaction_rate, evaluate_model
from tokenizer import Tokenizer
from dataset import get_splits

tok = Tokenizer()
_, val_ds, test_ds = get_splits("../data/data.txt", tok)

# Using evaluate_model with any predict function
def my_predict_fn(cond_tokens):
    # returns a date string e.g. '10-1-1810'
    ...

csr = evaluate_model(my_predict_fn, val_ds, tok)
print(f"CSR: {csr:.4f}")
```

---

## 13. Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.10 | Runtime |
| PyTorch | ≥ 2.0 | All models and training |
| NumPy | ≥ 1.24 | Numerical utilities |
| Matplotlib | ≥ 3.7 | Loss and CSR plots |
| tqdm | ≥ 4.65 | Training progress bars |
| pandas | ≥ 2.0 | Results logging |
| scikit-learn | ≥ 1.3 | Optional data utilities |

All standard library modules used (`datetime`, `calendar`, `pathlib`, `argparse`) require no installation.

---

## Coding Standards and Best Practices

- All Python files use **type hints** throughout (`Tensor`, `List[int]`, `Tuple`, etc.)
- Code is split into **separate files per concern** — no monolithic notebooks
- **Fixed random seed** (`seed=42`) in dataset splits and `torch.Generator` for full reproducibility
- **Manual seed** in `torch.manual_seed` at the start of every training run
- **Gradient clipping** (`max_norm=1.0`) applied in LSTM-based models
- **Validation CSR tracked every epoch** — not just loss
- **Best checkpoint saved** based on CSR, not loss
- All models implement a `.generate()` method with `@torch.no_grad()` for clean inference
- `batch_first=True` used consistently across all LSTM and Transformer modules

---

*DSAI 490 — Generative AI — Zewail City of Science and Technology*
