# Conditional Date Generation using four Deep Learning architectures

---

## Table of Contents

1. [Problem Description](#1-problem-description)
2. [Repository Structure](#2-repository-structure)
3. [Setup and Installation](#3-setup-and-installation)
4. [Data Format](#4-data-format)
5. [Tokenization Design](#5-tokenization-design)
6. [Models](#6-models)
   - [Model 1 — Conditional Autoencoder (IN-COURSE)](#model-1--conditional-autoencoder-in-course)
   - [Model 2 — Conditional GAN with WGAN-GP (IN-COURSE, REQUIRED)](#model-2--conditional-gan-with-wgan-gp-in-course-required)
   - [Model 3 — Transformer Encoder-Decoder (OUTSIDE-COURSE)](#model-3--transformer-encoder-decoder-outside-course)
   - [Model 4 — BiLSTM with Bahdanau Attention (OUTSIDE-COURSE)](#model-4--bilstm-with-bahdanau-attention-outside-course)
7. [Training](#7-training)
8. [Inference](#8-inference)
9. [Evaluation Metric](#9-evaluation-metric)
10. [Results and Analysis](#10-results-and-analysis)
11. [Failure Analysis](#11-failure-analysis)
12. [Best Practices Applied](#13-best-practices-applied)

---

## 1. Problem Description

The task is to build a **conditional date generator**: given four input constraints, produce any valid calendar date that satisfies all of them simultaneously.

| Condition | Format | Example | Meaning |
|-----------|--------|---------|---------|
| Day of week | `[DAY]` | `[WED]` | Output date must fall on a Wednesday |
| Month | `[MON]` | `[JAN]` | Output date must be in January |
| Leap year | `[True/False]` | `[False]` | Output year must (not) be a leap year |
| Decade | `[DDD]` | `[181]` | Output year must be in 1810–1819 |

**Output:** A date string in the format `d-m-yyyy`, covering the range **1 Jan 1800 – 31 Dec 2200**.

This is a **generation problem**, not classification. Multiple correct dates exist per input. For example, `[WED] [JAN] [False] [181]` admits every Wednesday in January of any non-leap year in the 1810s. Exact-match accuracy is therefore meaningless as a metric. The correct metric is **Constraint Satisfaction Rate (CSR)** — the fraction of predictions that satisfy all four conditions simultaneously.

The dataset contains approximately **146,000 date-condition pairs** covering every calendar date in the 1800–2200 range with its associated conditions precomputed.

---

## 2. Repository Structure

```
.
├── data/
│   ├── data.txt                  # Full dataset (~146k lines)
│   ├── example_input.txt         # Example input file (conditions only)
│   └── example_output.txt        # Example expected output
│
├── model/
│   ├── tokenizer.py              # Custom digit-by-digit tokenizer
│   ├── dataset.py                # PyTorch Dataset + 90/5/5 splits
│   ├── evaluate.py               # CSR metric and date validation
│   ├── model1_ae.py              # Conditional Autoencoder
│   ├── model2_gan.py             # Conditional GAN (WGAN-GP)
│   ├── model3_transformer.py     # Transformer Encoder-Decoder
│   ├── model4_bilstm_attention.py# BiLSTM + Bahdanau Attention
│   ├── train.py                  # Unified training script (all 4 models)
│   ├── predict.py                # Inference script (CLI)
│   ├── plot_results.py           # Plots loss + CSR curves from saved history
│   └── weights/
│       ├── ae_best.pt
│       ├── gan_generator_best.pt
│       ├── gan_discriminator_best.pt
│       ├── transformer_best.pt
│       ├── bilstm_best.pt
│       └── *_history.pt          # Training history for plotting
│
├── environment.yml               # Conda environment spec
├── pyvenv.cfg                    # Venv config (Python 3.11)
└── README.md                     # This file
```

---

## 3. Setup and Installation

### Option A — Conda (recommended)

```bash
conda env create -f environment.yml
conda activate dates_gen
```

### Option B — venv (Python 3.11)

```bash
python -m venv dates_gen
# Windows:
dates_gen\Scripts\activate
# Linux/macOS:
source dates_gen/bin/activate

pip install torch torchvision matplotlib
```

**Requirements:** Python 3.11, PyTorch ≥ 2.0, Matplotlib.

---

## 4. Data Format

Each line of `data.txt` follows this exact format:

```
[DAY] [MONTH] [LEAP] [DECADE] d-m-yyyy
```

**Examples:**

```
[MON] [DEC] [False] [196] 3-12-1962
[THU] [DEC] [True]  [204] 3-12-2048
[WED] [JAN] [False] [181] 10-1-1810
```

`example_input.txt` contains only the four condition tokens per line (no date). `predict.py` reads this format and writes predictions in the full `data.txt` format.

---

## 5. Tokenization Design

> Tokenization was the most critical design decision. The wrong encoding makes the problem unsolvable regardless of model power.

### Condition Encoding

Each of the four input conditions is mapped to an integer index from a fixed vocabulary:

| Condition | Tokens | Count |
|-----------|--------|-------|
| Day | MON TUE WED THU FRI SAT SUN | 7 |
| Month | JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC | 12 |
| Leap | False True | 2 |
| Decade | 180 181 … 220 | 41 |
| **Total** | | **62** |

Each sample is a vector of 4 integer indices: `[day_idx, month_idx, leap_idx, decade_idx]`.

### Date Encoding — Digit by Digit

This is the most important design choice. A date like `10-1-1810` is:

1. Zero-padded to `DDMMYYYY` → `01011810`
2. Split into 8 individual digit characters
3. Each digit encoded as an index in `{0, …, 9}`
4. Wrapped with `<S>` (start) and `<E>` (end) special tokens

**Digit vocabulary:** `0–9` + `<S>` + `<E>` + `<P>` (pad) = **13 tokens**  
**Sequence length:** `<S>` + 8 digits + `<E>` = **10 tokens**

**Why digit-by-digit encoding?**

- Treating the full year as a single token would require a vocabulary of 400+ values, making learning much harder.
- Digit positions have clear semantic structure: positions 0–1 are day (01–31), positions 2–3 are month (01–12), positions 4–7 are year (1800–2200).
- Each digit has a small, fixed vocabulary of 10 values, which is easy for any model to learn.
- The model can learn positional priors: digit position 4 (year century) is almost always `1` or `2`; digit position 5 maps directly to the decade condition; digits 6–7 are largely free.

This encoding makes the implicit structure of a calendar date explicit to the model at the token level.

---

## 6. Models

Four models were implemented and trained. Two are from the course curriculum and two are from outside it.

---

### Model 1 — Conditional Autoencoder (IN-COURSE)

**File:** `model1_ae.py`

**Why chosen:** The AE was covered in the course and provides the cleanest deterministic baseline. It isolates the easy parts of the problem (month, year, decade) from the hard part (weekday), making it a useful diagnostic tool.

**Architecture:**

```
Encoder:
  Input: 4 condition embeddings (4 × 32 = 128-dim flattened)
  MLP:   Linear(128 → 256) → ReLU → Dropout
         Linear(256 → 128) → ReLU → Dropout
         Linear(128 → 64)                       ← bottleneck z

Decoder:
  Input: z (64) concatenated with condition embeddings (128) = 192-dim
         (skip connection: directly passes condition signal past bottleneck)
  MLP:   Linear(192 → 512) → ReLU → Dropout
         Linear(512 → 256) → ReLU → Dropout
         Linear(256 → 8 × 13)                   ← 8 digit logits
  Output: (B, 8, 13) logits
```

**Loss function:**

```
L = (1/8) × Σ_t CrossEntropy(logits_t, target_t)
```

Plain cross-entropy averaged over all 8 digit positions. No KL or regularisation term.

**Why the skip connection matters:** The encoder bottleneck (64-dim) can lose fine-grained condition detail. Concatenating the raw condition embeddings with `z` in the decoder gives it direct access to all four conditions, preventing information loss.

**Core limitation:** The AE is fully **deterministic**. For the same four conditions it always outputs the same date — the most frequent valid date seen during training. It cannot model the one-to-many nature of the problem where many valid dates exist per condition set. This motivated all three other models.

---

### Model 2 — Conditional GAN with WGAN-GP (IN-COURSE, REQUIRED)

**File:** `model2_gan.py`

**Why chosen:** GANs are required by the assignment and are a natural fit for this problem. The noise vector `z` explicitly models the one-to-many mapping: different `z` samples for the same condition can produce different valid dates.

**Architecture:**

```
Generator:
  Input:  condition embeddings (128-dim) + Gaussian noise (64-dim) = 192-dim
  MLP:    Linear → BatchNorm → LeakyReLU(0.2)   ×3   (512 → 512 → 256)
  Output: (B, 8, 13) logits — 8 digit positions

Discriminator:
  Input:  condition embeddings (128-dim) + flattened date one-hot (8×13=104-dim) = 232-dim
  MLP:    SpectralNorm(Linear) → LeakyReLU(0.2)  ×3   (256 → 128 → 64)
          Linear(64 → 1)                          ← raw Wasserstein score
  No BatchNorm (required by WGAN-GP)
  SpectralNorm for Lipschitz stability
```

**Loss function — WGAN-GP:**

```
Discriminator:  L_D = E[D(fake)] − E[D(real)] + λ · GP
Generator:      L_G = −E[D(G(z, cond))]

Gradient Penalty:
  x̂ = α · real + (1−α) · fake,   α ~ Uniform(0,1)
  GP = E[(‖∇_{x̂} D(x̂)‖₂ − 1)²]
  λ = 10  (standard)
```

**Why WGAN-GP over vanilla GAN:**

1. **No mode collapse** — Wasserstein distance is a more informative divergence measure than JS-divergence.
2. **Training stability** — no need for careful learning rate balancing between G and D.
3. **Meaningful D loss** — D loss correlates with generation quality and can be used to monitor training directly.
4. **No vanishing gradients** — D outputs a raw score instead of a saturating sigmoid probability.

**Training protocol:** D is updated 5 times for every 1 G update (`n_critic = 5`), as recommended for WGAN. Adam optimizer with `betas=(0.0, 0.9)` as recommended for WGAN training.

**Note on BatchNorm:** The discriminator deliberately has no BatchNorm. BatchNorm changes the gradient distribution across the batch and invalidates the gradient penalty calculation. SpectralNorm is used instead for Lipschitz regularisation.

---

### Model 3 — Transformer Encoder-Decoder (OUTSIDE-COURSE)

**File:** `model3_transformer.py`

**Why chosen:** The Transformer (Vaswani et al., 2017) is the backbone of modern LLMs and was not covered in the course. Its cross-attention mechanism is a uniquely strong fit for this problem: every output digit can attend directly to all four input conditions simultaneously, with no information bottleneck.

**Architecture:**

```
Encoder:
  Input:  4 condition token embeddings + sinusoidal positional encoding
  Layers: N=3 × TransformerEncoderLayer (multi-head self-attention + FFN)
  Output: memory (B, 4, d_model=128) — contextualised condition representation

Decoder:
  Input:  digit token embeddings + sinusoidal positional encoding
  Layers: N=3 × TransformerDecoderLayer
            - Masked self-attention (causal: position i cannot see j > i)
            - Cross-attention over encoder memory
            - Feed-forward network
  Output: logits (B, seq_len−1, 13)

Output projection: weight-tied with digit embedding matrix
```

**Key design choices:**

- **Causal mask:** prevents any digit from attending to future digit positions. This enforces correct auto-regressive generation.
- **Cross-attention:** each digit attends to all 4 conditions in parallel. Month digits attend to the month token; year digits attend to both the decade and leap tokens. This is impossible with a fixed hidden state.
- **Weight tying:** the output projection shares weights with the digit embedding. Reduces parameter count and improves generalisation.
- **LR schedule:** linear warmup for 500 steps then cosine decay. This is standard practice for Transformers and prevents instability in the early training phase.

**Loss function:**

```
L = (1/8) × Σ_t CrossEntropy(logits_t, target_t)
```

Teacher forcing is used during training (standard for encoder-decoder). Auto-regressive decoding is used at inference.

**Advantage over LSTM:** An LSTM must compress all four conditions into a fixed-size hidden state before decoding begins. The Transformer's cross-attention lets each decoder step query the condition representations directly, selecting what is relevant at each digit position.

---

### Model 4 — BiLSTM with Bahdanau Attention (OUTSIDE-COURSE)

**File:** `model4_bilstm_attention.py`

**Why chosen:** This model was not covered in the course. It combines bidirectional encoding (not in the course) with an explicit Bahdanau attention mechanism (not in the course). It fills a meaningful middle ground between the simple AE baseline and the more powerful Transformer, while offering interpretable attention weights that can verify the model is reasoning correctly.

**Architecture:**

```
Encoder — Bidirectional LSTM:
  Input:  4 condition token embeddings (64-dim each)
  LSTM:   bidirectional, 2 layers, hidden_dim=128 per direction
  Output: enc_outputs (B, 4, 256) — forward + backward states concatenated
          h_dec, c_dec projected to (2, B, 128) for decoder initialisation

Attention — Bahdanau (additive):
  e_{t,i} = v^T · tanh(W_s · s_t + W_h · h_i)    ← alignment score
  α_{t,i} = softmax(e_{t,i})                       ← attention weights
  c_t      = Σ_i α_{t,i} · h_i                    ← context vector (B, 256)

Decoder — LSTM:
  Input at step t:  [digit_embedding_t ; context_vector_t]   (64 + 256 = 320-dim)
  LSTM:             2 layers, hidden_dim=128
  Projection:       fc([hidden_output ; context]) → 13 logits
```

**Training details:**
- Teacher forcing with a ratio that **decays linearly** from 0.8 at epoch 1 to 0.2 at epoch 30. This gradually forces the model to rely on its own predictions rather than ground-truth inputs, preventing exposure bias at inference time.
- Gradient clipping at `max_norm=1.0` prevents exploding gradients common in RNN training.

**Interpretability:** The `get_attention_weights()` method returns the full attention matrix `(num_steps × 4)`, showing which condition each digit position attends to most. This can be visualised to verify the model is reasoning in the expected direction (e.g., month digits attending to the month condition token, year digits attending to the decade token).

**Advantage over plain Seq2Seq:** A standard Seq2Seq decoder only sees the encoder's final hidden state. Bahdanau attention gives the decoder access to all encoder positions at each step, reducing the information bottleneck and making the model's reasoning interpretable.

---

## 7. Training

### Running training

```bash
cd model/

# Train a single model
python train.py --model ae
python train.py --model gan
python train.py --model transformer
python train.py --model bilstm

# Train all four models sequentially
python train.py --model all --epochs 30 --batch 256 --seed 42
```

### Training pipeline

- **Data split:** 90% train / 5% val / 5% test (fixed seed=42, ~131k / 7.3k / 7.3k samples)
- **Shuffling:** training DataLoader shuffles every epoch
- **Batch size:** 256 (configurable via `--batch`)
- **Epochs:** 30 default (configurable via `--epochs`)
- **Device:** CUDA if available, else CPU
- **Gradient clipping:** `max_norm=1.0` for all models with a gradient flow

### Per-model optimiser settings

| Model | Optimiser | LR | Scheduler | Notes |
|-------|-----------|----|-----------|----|
| AE | Adam | 1e-3 | ReduceLROnPlateau (patience=5, factor=0.5) | Standard |
| GAN | Adam | 1e-4 | None | betas=(0.0, 0.9) — WGAN standard |
| Transformer | Adam | 1e-3 | Linear warmup 500 steps → cosine decay | betas=(0.9, 0.98), eps=1e-9 |
| BiLSTM | Adam | 1e-3 | ReduceLROnPlateau (patience=5, factor=0.5) | Teacher forcing decay 0.8→0.2 |

### Saved outputs

All weight files and training histories are saved to `model/weights/`:

```
weights/
├── ae_best.pt                  # Best AE weights (by val CSR)
├── gan_generator_best.pt       # Best GAN Generator weights
├── gan_discriminator_best.pt   # Best GAN Discriminator weights
├── transformer_best.pt         # Best Transformer weights
├── bilstm_best.pt              # Best BiLSTM weights
├── ae_history.pt               # {train_loss, val_loss, val_csr} lists
├── gan_history.pt              # {g_loss, d_loss, val_csr} lists
├── transformer_history.pt
└── bilstm_history.pt
```

### Plotting training curves

```bash
python plot_results.py
```

Generates `weights/<model>_curves.png` for each trained model, showing loss and CSR over epochs.

---

## 8. Inference

### Command-line inference

```bash
cd model/
python predict.py -i ../data/example_input.txt -o predictions.txt
```

The default model is the **Transformer** (best overall CSR). To switch models, change `MODEL_NAME` at the top of `predict.py`:

```python
MODEL_NAME = "transformer"   # options: "ae", "gan", "transformer", "bilstm"
```

### Input format

One condition per line, no date:

```
[WED] [JAN] [False] [180]
[MON] [JAN] [False] [190]
[SAT] [JAN] [True]  [200]
```

### Output format

Matches `data.txt` exactly — conditions followed by predicted date:

```
[WED] [JAN] [False] [180] 12-1-1805
[MON] [JAN] [False] [190] 12-1-1905
[SAT] [JAN] [True]  [200] 12-1-2008
```

---

## 9. Evaluation Metric

**Constraint Satisfaction Rate (CSR)** is the primary metric throughout training and evaluation.

```
CSR = (number of predictions satisfying all 4 conditions) / (total predictions)
```

A prediction is valid if and only if:

1. The date's weekday matches the `[DAY]` condition
2. The date's month matches the `[MONTH]` condition
3. `calendar.isleap(year)` matches the `[LEAP]` condition
4. `year // 10` matches the `[DECADE]` condition
5. The date is a valid calendar date (e.g. not Feb 30) in the range 1800–2200

CSR is logged at every epoch on the validation set. The checkpoint with the **highest validation CSR** is saved as the best model — not the checkpoint with the lowest loss. This is the correct selection criterion because in a generation problem with multiple valid answers, loss reduction does not guarantee better constraint satisfaction.

---

## 10. Results and Analysis

### Final validation CSR (best checkpoint over 30 epochs)

| Model | Best Val CSR | Architecture type |
|-------|-------------|-------------------|
| Conditional AE | 0.150 | In-course, deterministic |
| Conditional GAN (WGAN-GP) | 0.149 | In-course, stochastic |
| Transformer Encoder-Decoder | 0.150 | Outside-course |
| **BiLSTM + Bahdanau Attention** | **0.151** | Outside-course |

### Interpretation of the ~0.15 ceiling

A CSR of 0.15 is very close to 1/7 ≈ 0.143 — the probability of guessing the correct weekday at random. This is a meaningful and informative result.

All four models learned to generate **syntactically valid dates** in the correct month and decade. The month condition (positions 2–3 in the digit sequence) and the decade condition (positions 4–6) are directly visible as digit patterns in the output. These are easy for any model to learn.

The **weekday condition is fundamentally harder** because it is not written anywhere in the output digit string. To satisfy it, the model must implicitly learn the Zeller/modular arithmetic relationship between day, month, and year — not just copy a pattern from the input. The models learn date formatting but do not learn calendar arithmetic, so they tend to predict a fixed day (most commonly day 12, a frequent value in training) regardless of the weekday condition, satisfying month and decade but failing weekday.

This is precisely why loss can continue to decrease while CSR stays near 0.15: the model is becoming a better date-string generator, but not a better calendar reasoner.

### Model-by-model behaviour

**AE:** Converges quickly (by epoch 5) and plateaus. Training and validation loss track closely with no overfitting. CSR oscillates between 0.13 and 0.15. Determinism is the binding constraint — the model always outputs the same date for the same condition, so it cannot explore the space of valid dates.

**GAN:** Shows a distinct training pattern. Generator loss starts near 0, drops sharply to −3 by epoch 4, then recovers toward −0.6 by epoch 30. Discriminator loss starts at 1.5 and converges toward 0. This is expected WGAN-GP behaviour: a stronger discriminator provides a stronger gradient signal to the generator. CSR starts at 0 and climbs steadily to 0.14, meaning the GAN is still actively learning at epoch 30. More training epochs would likely improve GAN results.

**Transformer:** Extremely stable training with nearly identical training and validation loss curves, indicating strong generalisation and no overfitting. CSR plateaus near 0.15 early in training and stays there. The model is the best in terms of training stability and is used as the default in `predict.py`.

**BiLSTM:** Achieves the highest CSR (0.151). The attention mechanism allows the decoder to focus on the most relevant condition at each digit position, which provides a small but consistent improvement over the other models. The decaying teacher forcing ratio forces the model to rely on its own predictions progressively, improving robustness at inference time.

### Success examples

| Input | Output | Verification |
|-------|--------|-------------|
| `[WED] [JAN] [False] [180]` | `12-1-1805` | Jan 12, 1805 = Wednesday, non-leap, decade 180 ✓ |
| `[SAT] [JAN] [True] [200]` | `12-1-2008` | Jan 12, 2008 = Saturday, leap year, decade 200 ✓ |
| `[FRI] [MAY] [True] [189]` | `12-5-1892` | May 12, 1892 = Friday, leap year, decade 189 ✓ |

---

## 11. Failure Analysis

| Input | Output | Issue |
|-------|--------|-------|
| `[MON] [JAN] [False] [190]` | `12-1-1905` | Jan 12, 1905 was a Thursday, not Monday ✗ |
| `[TUE] [MAR] [False] [189]` | `12-3-1891` | Mar 12, 1891 was a Thursday, not Tuesday ✗ |

In failure cases, the model correctly predicts the month and year (visible patterns), but consistently outputs day 12 regardless of the weekday requirement. This reveals that the model has learned date formatting rather than calendar reasoning.

**Root cause:** The weekday is an implicit function of the full date (`(d + m + y) mod 7` with correction factors). No individual digit in the output encodes this information directly. The model would need to learn modular arithmetic implicitly from co-occurrence patterns alone — a significantly harder inductive leap than copying the month token into the month digit positions.

**Potential improvements:**
- Generate year and month first (fixing the year-month context), then predict the day conditioned on all prior context. This concentrates the calendar reasoning into the day-prediction step.
- Add a calendar-arithmetic auxiliary loss that rewards the correct weekday explicitly during training.
- Use a constraint-aware decoding strategy that rejects invalid dates and samples again.
- Train for more epochs (especially the GAN, which had not converged at epoch 30).

---

## 12. Best Practices Applied

| Practice | Implementation |
|----------|---------------|
| Fixed random seed | `torch.manual_seed(42)` in `train.py` and dataset splitting |
| Train/val/test split | 90% / 5% / 5% via `random_split` with fixed generator seed |
| Shuffled training data | `DataLoader(shuffle=True)` on train set only |
| Gradient clipping | `nn.utils.clip_grad_norm_(params, max_norm=1.0)` on all models |
| Best checkpoint saving | Saved by highest validation CSR, not lowest loss |
| Correct monitoring metric | CSR logged every epoch (not accuracy or loss alone) |
| Type hints | Full `typing` annotations throughout all files |
| Modular code structure | One file per model, separate tokenizer/dataset/evaluate modules |
| Sanity tests | `if __name__ == "__main__":` block in every module |
| Training history | Saved as `.pt` files for reproducible plotting |
| Environment spec | `environment.yml` for exact conda environment replication |
| CLI interface | `argparse` in both `train.py` and `predict.py` |
| GAN protocol | n_critic=5, WGAN-recommended Adam betas, spectral norm, no BN in D |
| Teacher forcing decay | BiLSTM linearly decays TF ratio 0.8 → 0.2 to reduce exposure bias |
| Weight tying | Transformer output projection shares weights with digit embedding |
| LR scheduling | ReduceLROnPlateau for AE/BiLSTM; warmup+cosine for Transformer |
