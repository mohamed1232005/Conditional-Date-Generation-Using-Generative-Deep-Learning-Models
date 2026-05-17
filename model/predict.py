"""
predict.py
----------
Inference script for the Dates Generator.

Usage:
    python predict.py -i path/to/example_input.txt -o path/to/output.txt

Input file format (one condition per line, no date):
    [WED] [JAN] [False] [180]

Output file format (conditions + predicted date, matching data.txt exactly):
    [WED] [JAN] [False] [180] 1-1-1800

Default model: Transformer (best CSR).
To switch models, change MODEL_NAME to 'ae', 'gan', or 'bilstm'.

Models and their weight files:
    ae          → weights/ae_best.pt
    gan         → weights/gan_generator_best.pt
    transformer → weights/transformer_best.pt
    bilstm      → weights/bilstm_best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tokenizer import Tokenizer, START_IDX, END_IDX
from model3_transformer import TransformerDateModel

# ── Config — change MODEL_NAME to switch models ───────────────────────────────

MODEL_NAME   = "transformer"
WEIGHTS_PATH = Path("weights") / f"{MODEL_NAME}_best.pt"
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Load tokenizer and model ──────────────────────────────────────────────────

tok = Tokenizer()

model = TransformerDateModel(
    cond_vocab_size=tok.cond_vocab_size,
    digit_vocab_size=tok.digit_vocab_size,
)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


# ── Prediction function ───────────────────────────────────────────────────────

def predict_date(line: str) -> str:
    """
    Predict a date string for a given condition line.

    Args:
        line : e.g. '[WED] [JAN] [False] [180]'
                 or '[WED] [JAN] [False] [180] 1-1-1800' (date ignored)

    Returns:
        Date string 'd-m-yyyy', or '1-1-1800' as a safe fallback.
    """
    cond_indices = tok.encode_conditions(line)
    cond_tensor  = torch.tensor([cond_indices], dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        digit_indices = model.generate(
            cond_tensor,
            start_idx=START_IDX,
            end_idx=END_IDX,
        )

    date_str = tok.decode_date([START_IDX] + digit_indices + [END_IDX])
    return date_str if date_str else "1-1-1800"


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Dates Generator — inference")
    parser.add_argument("-i", required=True, help="Path to input conditions file")
    parser.add_argument("-o", required=True, help="Path to output predictions file")
    args = parser.parse_args()

    input_path  = Path(args.i)
    output_path = Path(args.o)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with input_path.open("r") as fin, output_path.open("w") as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue

            # Always take only the 4 condition tokens (ignore any trailing date)
            parts     = line.split()
            cond_part = " ".join(parts[:4])
            pred_date = predict_date(cond_part)

            # Output format matches data.txt exactly
            fout.write(f"{cond_part} {pred_date}\n")
            written += 1

    print(f"Done — wrote {written} predictions to {output_path}")


if __name__ == "__main__":
    main()