"""
model4_bilstm_attention.py
--------------------------
Model 4 (OUTSIDE-COURSE): Bidirectional LSTM with Bahdanau Attention.

This architecture is outside the course curriculum. It combines two
advanced concepts not covered in the course:
  1. Bidirectional LSTM encoder — reads conditions in both directions.
  2. Bahdanau (additive) attention mechanism — lets the decoder focus
     on the most relevant condition tokens when predicting each digit.

Architecture:

  Encoder — Bidirectional LSTM:
    Input  : 4 condition token embeddings
    LSTM   : reads the sequence forward AND backward simultaneously
    Output : hidden states h = [h_forward; h_backward] for each position
             Shape: (B, 4, 2*hidden_dim)  — 4 positions, each with
             concatenated forward and backward hidden states.
    Why BiLSTM: even though we only have 4 tokens, bidirectionality
    ensures each token representation is informed by all other tokens.
    The month token can "see" the decade token and vice versa.

  Attention — Bahdanau (additive) mechanism:
    For each decoder step t, computes an alignment score between the
    decoder hidden state s_t and each encoder hidden state h_i:

      e_{t,i} = v^T · tanh(W_s · s_t + W_h · h_i)
      α_{t,i} = softmax(e_{t,i})           ← attention weights
      c_t     = Σ_i α_{t,i} · h_i         ← context vector

    The context vector c_t is a weighted sum of encoder states,
    focusing on the conditions most relevant to predicting digit t.

  Decoder — LSTM:
    Input at each step : [digit_embedding; context_vector]
                         (concatenation of current digit embed + attention context)
    State              : h, c from previous step
    Output             : logits over digit_vocab_size

Loss function:
  CrossEntropyLoss over the 8 date digit positions.
  L = (1/8) Σ_t CE(logits_t, target_t)
  Teacher forcing applied during training (decaying ratio).

Why BiLSTM+Attention for this problem:
  - The attention mechanism can learn WHICH conditions matter most for
    each digit. E.g. for predicting the month digits (positions 2-3),
    the model should attend strongly to the [MON] condition token.
    For year digits (positions 4-7), it should attend to [decade] and [leap].
  - This interpretability is a key advantage — attention weights can be
    visualised to verify the model is reasoning correctly.
  - Unlike the plain Seq2Seq (which would be in-course), the attention
    mechanism is a significant architectural addition that was developed
    after basic RNNs and before Transformers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple


class BiLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM encoder over 4 condition tokens.

    Reads the condition sequence in both forward and backward directions,
    concatenating the hidden states to get a richer representation.

    Args:
        cond_vocab_size : size of condition vocabulary (62)
        embed_dim       : per-token embedding dimension
        hidden_dim      : LSTM hidden state size (each direction)
        num_layers      : number of BiLSTM layers
        dropout         : dropout between layers
    """

    def __init__(
        self,
        cond_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embed = nn.Embedding(cond_vocab_size, embed_dim)
        self.lstm  = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project bidirectional hidden state (2*hidden) to hidden for decoder init
        self.hidden_proj = nn.Linear(2 * hidden_dim, hidden_dim)
        self.cell_proj   = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, cond: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            cond : LongTensor (B, 4) — condition indices

        Returns:
            enc_outputs : (B, 4, 2*hidden_dim) — all hidden states (for attention)
            h_dec       : (num_layers, B, hidden_dim) — decoder initial hidden
            c_dec       : (num_layers, B, hidden_dim) — decoder initial cell
        """
        embedded    = self.embed(cond)                         # (B, 4, embed)
        enc_outputs, (h, c) = self.lstm(embedded)              # enc: (B,4,2*hid)
                                                               # h,c: (2*layers,B,hid)

        # h shape: (num_layers*2, B, hidden_dim)
        # Separate forward/backward layers, concatenate, project
        h = h.view(self.num_layers, 2, -1, self.hidden_dim)   # (layers,2,B,hid)
        c = c.view(self.num_layers, 2, -1, self.hidden_dim)

        # Concatenate forward (0) and backward (1) for each layer
        h_cat = torch.cat([h[:, 0, :, :], h[:, 1, :, :]], dim=-1)  # (layers,B,2*hid)
        c_cat = torch.cat([c[:, 0, :, :], c[:, 1, :, :]], dim=-1)

        # Project to hidden_dim for decoder
        h_dec = torch.tanh(self.hidden_proj(h_cat))   # (layers, B, hidden_dim)
        c_dec = torch.tanh(self.cell_proj(c_cat))

        return enc_outputs, h_dec, c_dec


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.

    Computes alignment scores between decoder state and all encoder states:
      e_i   = v^T · tanh(W_s · s + W_h · h_i)
      α     = softmax(e)
      context = Σ α_i · h_i

    Args:
        encoder_dim : dimension of encoder hidden states (2*hidden_dim for BiLSTM)
        decoder_dim : dimension of decoder hidden state
        attn_dim    : internal attention dimension
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        attn_dim: int = 64,
    ) -> None:
        super().__init__()
        # W_s · s_t  (decoder state projection)
        self.W_s = nn.Linear(decoder_dim, attn_dim, bias=False)
        # W_h · h_i  (encoder state projection)
        self.W_h = nn.Linear(encoder_dim, attn_dim, bias=False)
        # v^T        (score vector)
        self.v   = nn.Linear(attn_dim, 1, bias=False)

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            decoder_hidden  : (B, decoder_dim) — current decoder hidden state
            encoder_outputs : (B, 4, encoder_dim) — all encoder positions

        Returns:
            context : (B, encoder_dim) — weighted sum of encoder states
            weights : (B, 4)           — attention weights (sum to 1, for viz)
        """
        # Broadcast decoder state over encoder positions
        s = self.W_s(decoder_hidden).unsqueeze(1)  # (B, 1, attn_dim)
        h = self.W_h(encoder_outputs)              # (B, 4, attn_dim)

        # Additive score
        scores = self.v(torch.tanh(s + h)).squeeze(-1)   # (B, 4)
        weights = F.softmax(scores, dim=-1)               # (B, 4)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)  # (B, enc_dim)
        return context, weights


class AttentionDecoder(nn.Module):
    """
    LSTM decoder with Bahdanau attention.

    At each step:
      1. Compute attention context c_t from current hidden state + encoder outputs.
      2. Concatenate [digit_embed_t ; c_t] as LSTM input.
      3. LSTM step → new hidden state.
      4. Project to digit logits.

    Args:
        digit_vocab_size : size of digit vocabulary (13)
        embed_dim        : digit token embedding dimension
        hidden_dim       : LSTM hidden state size
        encoder_dim      : encoder output dimension (2*encoder_hidden for BiLSTM)
        num_layers       : LSTM depth
        dropout          : dropout probability
    """

    def __init__(
        self,
        digit_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        encoder_dim: int = 256,   # 2 * BiLSTM hidden_dim
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.digit_vocab_size = digit_vocab_size

        self.embed     = nn.Embedding(digit_vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attn_dim=64)

        # LSTM input = digit_embed + context_vector
        self.lstm = nn.LSTM(
            embed_dim + encoder_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Final projection: [hidden; context] → logits
        self.fc = nn.Linear(hidden_dim + encoder_dim, digit_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward_step(
        self,
        x: Tensor,
        h: Tensor,
        c: Tensor,
        enc_outputs: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        Single decoder step.

        Args:
            x           : (B, 1) current digit token
            h           : (num_layers, B, hidden_dim) hidden state
            c           : (num_layers, B, hidden_dim) cell state
            enc_outputs : (B, 4, encoder_dim)

        Returns:
            logits  : (B, digit_vocab_size)
            h       : updated hidden state
            c       : updated cell state
            weights : (B, 4) attention weights
        """
        # Use top layer hidden state for attention
        top_h = h[-1]                                          # (B, hidden_dim)
        context, attn_weights = self.attention(top_h, enc_outputs)  # (B, enc_dim)

        emb  = self.dropout(self.embed(x.squeeze(1)))         # (B, embed)
        lstm_in = torch.cat([emb, context], dim=-1).unsqueeze(1)  # (B,1,emb+enc)

        out, (h, c) = self.lstm(lstm_in, (h, c))              # out: (B,1,hidden)
        out = out.squeeze(1)                                   # (B, hidden)

        # Concatenate hidden output with context for richer projection
        logits = self.fc(torch.cat([out, context], dim=-1))   # (B, vocab)
        return logits, h, c, attn_weights


class BiLSTMAttentionModel(nn.Module):
    """
    Full BiLSTM + Bahdanau Attention model.

    Combines BiLSTMEncoder + AttentionDecoder.

    Args:
        cond_vocab_size  : size of condition vocabulary (62)
        digit_vocab_size : size of digit vocabulary (13)
        embed_dim        : token embedding dimension (shared encoder/decoder)
        hidden_dim       : LSTM hidden state size
        num_layers       : LSTM depth
        dropout          : dropout probability
    """

    def __init__(
        self,
        cond_vocab_size: int,
        digit_vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        encoder_dim = 2 * hidden_dim   # BiLSTM concatenates fwd+bwd

        self.encoder = BiLSTMEncoder(
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.decoder = AttentionDecoder(
            digit_vocab_size=digit_vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            encoder_dim=encoder_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.digit_vocab_size = digit_vocab_size

    def forward(
        self,
        cond: Tensor,
        tgt: Tensor,
        teacher_forcing_ratio: float = 0.5,
    ) -> Tensor:
        """
        Training forward pass with teacher forcing.

        Args:
            cond                  : (B, 4) condition indices
            tgt                   : (B, seq_len) target digit sequence incl. <S>/<E>
            teacher_forcing_ratio : probability of using ground-truth as next input

        Returns:
            all_logits : (B, seq_len-1, digit_vocab_size)
        """
        enc_outputs, h, c = self.encoder(cond)

        inp        = tgt[:, 0:1]   # <S> token
        all_logits = []

        for t in range(1, tgt.size(1)):
            logits, h, c, _ = self.decoder.forward_step(inp, h, c, enc_outputs)
            all_logits.append(logits.unsqueeze(1))

            use_teacher = torch.rand(1).item() < teacher_forcing_ratio
            inp = tgt[:, t:t+1] if use_teacher else logits.argmax(dim=-1, keepdim=True)

        return torch.cat(all_logits, dim=1)   # (B, seq-1, vocab)

    @torch.no_grad()
    def generate(
        self,
        cond: Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 12,
    ) -> list[int]:
        """
        Auto-regressive generation for a single sample.

        Args:
            cond      : (1, 4) condition tensor
            start_idx : <S> token index
            end_idx   : <E> token index
            max_len   : safety cap

        Returns:
            List of digit indices (excluding <S>, stopping before <E>).
        """
        self.eval()
        enc_outputs, h, c = self.encoder(cond)
        inp       = torch.tensor([[start_idx]], dtype=torch.long, device=cond.device)
        generated: list[int] = []

        for _ in range(max_len):
            logits, h, c, _ = self.decoder.forward_step(inp, h, c, enc_outputs)
            nxt = logits.argmax(dim=-1).item()
            if nxt == end_idx:
                break
            generated.append(nxt)
            inp = torch.tensor([[nxt]], dtype=torch.long, device=cond.device)

        return generated

    @torch.no_grad()
    def get_attention_weights(
        self,
        cond: Tensor,
        start_idx: int,
        end_idx: int,
        max_len: int = 12,
    ) -> Tuple[list[int], Tensor]:
        """
        Generate a date AND return the attention weights at each step.
        Useful for visualisation and report analysis.

        Args:
            cond      : (1, 4) condition tensor
            start_idx : <S> token index
            end_idx   : <E> token index
            max_len   : safety cap

        Returns:
            (generated_indices, attention_matrix)
            attention_matrix : (num_steps, 4) — weight on each condition token
                               per generated digit step
        """
        self.eval()
        enc_outputs, h, c = self.encoder(cond)
        inp        = torch.tensor([[start_idx]], dtype=torch.long, device=cond.device)
        generated: list[int] = []
        all_weights: list[Tensor] = []

        for _ in range(max_len):
            logits, h, c, weights = self.decoder.forward_step(inp, h, c, enc_outputs)
            nxt = logits.argmax(dim=-1).item()
            if nxt == end_idx:
                break
            generated.append(nxt)
            all_weights.append(weights.squeeze(0))   # (4,)
            inp = torch.tensor([[nxt]], dtype=torch.long, device=cond.device)

        attn_matrix = torch.stack(all_weights, dim=0) if all_weights else torch.zeros(1, 4)
        return generated, attn_matrix


# ── Quick sanity test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from tokenizer import Tokenizer, START_IDX, END_IDX

    tok   = Tokenizer()
    model = BiLSTMAttentionModel(
        cond_vocab_size=tok.cond_vocab_size,
        digit_vocab_size=tok.digit_vocab_size,
    )
    print(model)
    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")

    # Training forward
    cond   = torch.randint(0, tok.cond_vocab_size, (4, 4))
    date   = torch.randint(0, tok.digit_vocab_size, (4, 10))
    logits = model(cond, date, teacher_forcing_ratio=0.5)
    print(f"Train output shape : {logits.shape}")   # (4, 9, 13)

    # Inference
    indices, attn = model.get_attention_weights(cond[:1], START_IDX, END_IDX)
    date_str = tok.decode_date([START_IDX] + indices + [END_IDX])
    print(f"Generated date     : {date_str}")
    print(f"Attention matrix   : {attn.shape}")     # (num_steps, 4)
    print(f"Attention weights  :\n{attn.numpy()}")