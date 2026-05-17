"""
tokenizer.py
------------
Custom tokenizer for the Dates Generator problem.

Design decisions (explained in report):
- Conditions are encoded as integer indices into a fixed vocabulary.
- Dates are encoded DIGIT-BY-DIGIT (not as a whole number) so the model
  can learn positional digit structure (e.g. year[0] is always '1' or '2').
- Date format is zero-padded: dd mm yyyy → 8 digits total.
- Special tokens: <S> (start), <E> (end), <P> (pad).
"""

from __future__ import annotations
from typing import List, Optional


# ── Condition vocabularies ────────────────────────────────────────────────────

DAY_TOKENS: List[str] = ["MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN"]

MONTH_TOKENS: List[str] = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
]

LEAP_TOKENS: List[str] = ["False", "True"]

# Decades: 1800-2200 → first 3 digits of year → 180, 181, ..., 220
DECADE_TOKENS: List[str] = [str(d) for d in range(180, 221)]

# ── Digit vocabulary ──────────────────────────────────────────────────────────

DIGITS: List[str] = list("0123456789")
SPECIAL_TOKENS: List[str] = ["<S>", "<E>", "<P>"]
DIGIT_VOCAB: List[str] = DIGITS + SPECIAL_TOKENS

START_IDX: int = DIGIT_VOCAB.index("<S>")
END_IDX:   int = DIGIT_VOCAB.index("<E>")
PAD_IDX:   int = DIGIT_VOCAB.index("<P>")

# Date sequence length: <S> + 8 digits + <E> = 10
DATE_SEQ_LEN: int = 10


class Tokenizer:
    """
    Encodes condition tokens and date strings into integer index sequences,
    and decodes integer sequences back into date strings.

    Condition vocab size : len(DAY+MONTH+LEAP+DECADE) = 7+12+2+41 = 62
    Digit vocab size     : 10 digits + 3 special = 13
    """

    def __init__(self) -> None:
        self.cond_vocab: List[str] = DAY_TOKENS + MONTH_TOKENS + LEAP_TOKENS + DECADE_TOKENS
        self.cond2idx: dict[str, int] = {t: i for i, t in enumerate(self.cond_vocab)}
        self.digit2idx: dict[str, int] = {t: i for i, t in enumerate(DIGIT_VOCAB)}
        self.idx2digit: dict[int, str] = {i: t for t, i in self.digit2idx.items()}

        self.cond_vocab_size: int = len(self.cond_vocab)
        self.digit_vocab_size: int = len(DIGIT_VOCAB)

    # ── Encoding ──────────────────────────────────────────────────────────────

    def encode_conditions(self, line: str) -> List[int]:
        """
        Parse a data.txt line and return a list of 4 condition indices.

        Args:
            line: e.g. '[WED] [JAN] [False] [181] 10-1-1810'
                  or   '[WED] [JAN] [False] [181]'  (no date)

        Returns:
            List of 4 integer indices: [day_idx, month_idx, leap_idx, decade_idx]
        """
        parts = line.strip().split()
        tokens = [p.strip("[]") for p in parts[:4]]
        return [self.cond2idx[t] for t in tokens]

    def decode_conditions(self, indices: List[int]) -> List[str]:
        """Decode 4 condition indices back to string tokens."""
        return [self.cond_vocab[i] for i in indices]

    def encode_date(self, date_str: str) -> List[int]:
        """
        Encode a date string to a padded digit-index sequence with start/end tokens.

        '10-1-1810' → zero-pad → '10011810' → [<S>, 1,0,0,1,1,8,1,0, <E>]

        Returns:
            List of length DATE_SEQ_LEN (10).
        """
        d_str, m_str, y_str = date_str.strip().split("-")
        padded = f"{int(d_str):02d}{int(m_str):02d}{int(y_str):04d}"
        indices = (
            [START_IDX]
            + [self.digit2idx[ch] for ch in padded]
            + [END_IDX]
        )
        return indices

    def decode_date(self, indices: List[int]) -> Optional[str]:
        """
        Decode a sequence of digit indices back into a date string.

        Returns:
            Date string 'd-m-yyyy', or None if the sequence is malformed.
        """
        digits = ""
        for idx in indices:
            tok = self.idx2digit.get(idx, "")
            if tok == "<S>" or tok == "<P>":
                continue
            if tok == "<E>":
                break
            digits += tok

        if len(digits) < 8:
            return None

        digits = digits[:8]  # guard against extra tokens
        dd   = int(digits[0:2])
        mm   = int(digits[2:4])
        yyyy = int(digits[4:8])
        return f"{dd}-{mm}-{yyyy}"

    # ── Utilities ─────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Tokenizer("
            f"cond_vocab_size={self.cond_vocab_size}, "
            f"digit_vocab_size={self.digit_vocab_size})"
        )


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    tok = Tokenizer()
    print(tok)

    line = "[WED] [JAN] [False] [181] 10-1-1810"
    cond = tok.encode_conditions(line)
    date = tok.encode_date("10-1-1810")
    print("Conditions encoded:", cond)
    print("Conditions decoded:", tok.decode_conditions(cond))
    print("Date encoded      :", date)
    print("Date decoded      :", tok.decode_date(date))