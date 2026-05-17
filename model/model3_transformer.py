"""
model3_transformer.py
---------------------
Model 3 (OUTSIDE-COURSE): Transformer Encoder-Decoder.

This architecture is outside the course curriculum. It applies the
Transformer (Vaswani et al., 2017) — the backbone of modern LLMs —
to the conditional date generation problem.

Architecture:
  Encoder:
    Input  : 4 condition token embeddings + positional encoding
    Layers : N × TransformerEncoderLayer (multi-head self-attention + FFN)
    Output : memory (B, 4, d_model)

  Decoder:
    Input  : digit token embeddings + positional encoding
    Layers : N × TransformerDecoderLayer
             (masked self-attention + cross-attention to encoder + FFN)
    Output : logits (B, seq_len, digit_vocab_size)

  Key design choices:
    - Causal mask in decoder self-attention: prevents looking at future digits.
    - Cross-attention: every digit position attends to ALL 4 conditions
      simultaneously — no information bottleneck unlike LSTM hidden state.
    - Weight tying: output projection shares weights with digit embedding
      (reduces params, improves generalisation).
    - Warmup + cosine LR schedule (set in train.py).

Loss function:
  CrossEntropyLoss over digit positions 1..9 of the target sequence.
  L = (1/8) Σ_t CE(logits_t, target_t)

Why a Transformer for this problem:
  Cross-attention directly relates each digit to ALL four conditions in
  parallel. Whether year-digit[2] is '8' or '9' depends jointly on the
  decade condition AND the leap condition — attention captures this
  directly, while an LSTM must compress everything into a fixed hidden state.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i/d))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d))

    Args:
        d_model : model dimension
        max_len : maximum sequence length supported
        dropout : dropout probability applied after adding PE
    """

    def __init__(self, d_model: int, max_len: int = 20, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe       = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, seq_len, d_model)
        Returns:
            x + PE with dropout, same shape
        """
        return self.dropout(x + self.pe[:, : x.size(1)])


class TransformerDateModel(nn.Module):
    """
    Encoder-Decoder Transformer for conditional date generation.

    Args:
        cond_vocab_size    : size of condition vocabulary (62)
        digit_vocab_size   : size of digit vocabulary (13)
        d_model            : transformer model dimension
        nhead              : number of attention heads (must divide d_model)
        num_encoder_layers : encoder depth
        num_decoder_layers : decoder depth
        dim_feedforward    : FFN inner dimension
        dropout            : dropout probability
    """

    def __init__(
        self,
        cond_vocab_size: int,
        digit_vocab_size: int,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.digit_vocab_size = digit_vocab_size
        self.d_model          = d_model

        # ── Source (condition) side ───────────────────────────────────────
        self.src_embed = nn.Embedding(cond_vocab_size, d_model)
        self.src_pos   = PositionalEncoding(d_model, max_len=10,  dropout=dropout)

        # ── Target (digit) side ───────────────────────────────────────────
        self.tgt_embed = nn.Embedding(digit_vocab_size, d_model)
        self.tgt_pos   = PositionalEncoding(d_model, max_len=15, dropout=dropout)

        # ── Transformer core ──────────────────────────────────────────────
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        # ── Output projection (weight-tied with tgt_embed) ────────────────
        self.output_proj        = nn.Linear(d_model, digit_vocab_size, bias=False)
        self.output_proj.weight = self.tgt_embed.weight   # weight tying

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init for all parameters except weight-tied output."""
        for name, p in self.named_parameters():
            if "output_proj" in name:
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, size: int, device: torch.device) -> Tensor:
        """Upper-triangular mask so position i cannot attend to j > i."""
        return nn.Transformer.generate_square_subsequent_mask(size, device=device)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            src : (B, 4)        — condition indices
            tgt : (B, seq_len)  — digit sequence incl. <S> and <E>
                  Decoder input  = tgt[:, :-1]
                  Prediction target = tgt[:, 1:]

        Returns:
            logits : (B, seq_len-1, digit_vocab_size)
        """
        tgt_input = tgt[:, :-1]
        tgt_mask  = self._causal_mask(tgt_input.size(1), src.device)

        memory = self.src_pos(self.src_embed(src))
        out    = self.transformer(
            src=memory,
            tgt=self.tgt_pos(self.tgt_embed(tgt_input)),
            tgt_mask=tgt_mask,
        )
        return self.output_proj(out)   # (B, seq-1, vocab)

    @torch.no_grad()
    def generate(
        self,
        src: Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 12,
    ) -> list[int]:
        """
        Auto-regressive inference — decodes one token at a time.

        Args:
            src       : (1, 4) condition tensor
            start_idx : <S> token index
            end_idx   : <E> token index
            max_len   : safety cap

        Returns:
            List of digit indices (excluding <S>, stopping before <E>).
        """
        self.eval()
        device  = src.device
        memory  = self.src_pos(self.src_embed(src))
        tgt_seq = torch.tensor([[start_idx]], dtype=torch.long, device=device)
        result: list[int] = []

        for _ in range(max_len):
            mask  = self._causal_mask(tgt_seq.size(1), device)
            out   = self.transformer(
                src=memory,
                tgt=self.tgt_pos(self.tgt_embed(tgt_seq)),
                tgt_mask=mask,
            )
            nxt = self.output_proj(out[:, -1, :]).argmax(dim=-1).item()
            if nxt == end_idx:
                break
            result.append(nxt)
            tgt_seq = torch.cat(
                [tgt_seq, torch.tensor([[nxt]], dtype=torch.long, device=device)],
                dim=1,
            )
        return result


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tokenizer import Tokenizer, START_IDX, END_IDX

    tok   = Tokenizer()
    model = TransformerDateModel(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    )
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    cond   = torch.randint(0, tok.cond_vocab_size, (4, 4))
    date   = torch.randint(0, tok.digit_vocab_size, (4, 10))
    logits = model(cond, date)
    print(f"Train output shape: {logits.shape}")   # (4, 9, 13)

    indices  = model.generate(cond[:1], START_IDX, END_IDX)
    date_str = tok.decode_date([START_IDX] + indices + [END_IDX])
    print(f"Generated date    : {date_str}")