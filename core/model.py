"""Conditional SELFIES-VAE + latent property head.

* Encoder: bidirectional GRU -> (mu, logvar) over a `latent_dim` Gaussian.
* Decoder: GRU conditioned on [z ; condition] both at init and every step, so
  generation can be steered toward a property spec (conditional VAE).
* Property head: MLP mapping the latent mean -> normalized properties, used for
  gradient-based latent search (search.py) and as an auxiliary training signal.

Sized for ~6 GB VRAM (≈8M params at the defaults in config.py).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from batterygen.core import config



class SelfiesVAE(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_idx: int,
        n_props: int = config.N_PROPS,
        emb_dim: int = config.EMB_DIM,
        enc_hidden: int = config.ENC_HIDDEN,
        dec_hidden: int = config.DEC_HIDDEN,
        latent_dim: int = config.LATENT_DIM,
        enc_layers: int = config.ENC_LAYERS,
        dec_layers: int = config.DEC_LAYERS,
        dropout: float = config.DROPOUT,
        prop_hidden: int = config.PROP_HEAD_HIDDEN,
        unk_idx: int = 3,
        word_dropout: float = config.WORD_DROPOUT,
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.unk_idx = unk_idx
        self.word_dropout = word_dropout
        self.latent_dim = latent_dim
        self.n_props = n_props
        self.dec_hidden = dec_hidden
        self.dec_layers = dec_layers

        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)

        # encoder
        self.encoder = nn.GRU(
            emb_dim, enc_hidden, num_layers=enc_layers,
            batch_first=True, bidirectional=True,
            dropout=dropout if enc_layers > 1 else 0.0,
        )
        self.fc_mu = nn.Linear(2 * enc_hidden, latent_dim)
        self.fc_logvar = nn.Linear(2 * enc_hidden, latent_dim)

        # decoder (conditioned on z + properties)
        cond_dim = latent_dim + n_props
        self.z2h = nn.Linear(cond_dim, dec_layers * dec_hidden)
        self.decoder = nn.GRU(
            emb_dim + cond_dim, dec_hidden, num_layers=dec_layers,
            batch_first=True, dropout=dropout if dec_layers > 1 else 0.0,
        )
        self.out = nn.Linear(dec_hidden, vocab_size)

        # latent -> properties
        self.prop_head = nn.Sequential(
            nn.Linear(latent_dim, prop_hidden), nn.ReLU(),
            nn.Linear(prop_hidden, n_props),
        )

    # ----------------------------------------------------------------- encode
    def encode(self, tokens: torch.Tensor, lengths: torch.Tensor):
        emb = self.dropout(self.embed(tokens))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.encoder(packed)                      # [num_layers*2, B, enc_hidden]
        h = torch.cat([h[-2], h[-1]], dim=-1)            # last layer, both directions
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def _init_hidden(self, zc: torch.Tensor) -> torch.Tensor:
        B = zc.size(0)
        h0 = torch.tanh(self.z2h(zc))                    # [B, dec_layers*dec_hidden]
        return h0.view(B, self.dec_layers, self.dec_hidden).transpose(0, 1).contiguous()

    # ----------------------------------------------------------------- decode
    def decode(self, z, cond, dec_inputs):
        """Teacher-forced decode. dec_inputs: [B, T] (starts with BOS)."""
        zc = torch.cat([z, cond], dim=-1)
        h0 = self._init_hidden(zc)
        emb = self.dropout(self.embed(dec_inputs))       # [B, T, emb]
        zc_exp = zc.unsqueeze(1).expand(-1, emb.size(1), -1)
        out, _ = self.decoder(torch.cat([emb, zc_exp], dim=-1), h0)
        return self.out(out)                             # [B, T, vocab]

    def _word_dropout(self, dec_inputs):
        """Randomly replace input tokens with <unk> (never BOS/pad) during training."""
        if not self.training or self.word_dropout <= 0:
            return dec_inputs
        drop = (torch.rand_like(dec_inputs, dtype=torch.float) < self.word_dropout)
        drop &= dec_inputs != self.pad_idx
        drop[:, 0] = False  # keep the leading BOS
        return dec_inputs.masked_fill(drop, self.unk_idx)

    def forward(self, tokens, lengths, cond):
        mu, logvar = self.encode(tokens, lengths)
        z = self.reparameterize(mu, logvar)
        dec_inputs = self._word_dropout(tokens[:, :-1])
        logits = self.decode(z, cond, dec_inputs)        # predict tokens[:, 1:]
        prop_pred = self.prop_head(mu)
        return logits, mu, logvar, prop_pred

    # --------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n, cond, bos, eos, max_len=config.MAX_SELFIES_LEN,
               temperature=1.0, z=None, z_scale=1.0, device="cpu", greedy=False):
        """Autoregressively sample n sequences. cond: [n, n_props]; returns [n, L].

        z_scale < 1.0 draws the prior from N(0, z_scale^2 I) — a truncated/shrunk
        latent that stays in well-trained regions, raising sample validity at a
        small cost to diversity (only used when z is not supplied)."""
        self.eval()
        if z is None:
            z = torch.randn(n, self.latent_dim, device=device) * z_scale
        cond = cond.to(device)
        zc = torch.cat([z, cond], dim=-1)
        h = self._init_hidden(zc)
        tok = torch.full((n, 1), bos, dtype=torch.long, device=device)
        seqs = [tok]
        finished = torch.zeros(n, dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            emb = self.embed(tok)
            dec_in = torch.cat([emb, zc.unsqueeze(1)], dim=-1)
            out, h = self.decoder(dec_in, h)
            logits = self.out(out[:, -1]) / max(temperature, 1e-6)
            if greedy:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            seqs.append(nxt)
            tok = nxt
            finished |= nxt.squeeze(1) == eos
            if bool(finished.all()):
                break
        return torch.cat(seqs, dim=1)


# --------------------------------------------------------------------------- #
# Loss
# --------------------------------------------------------------------------- #
def vae_loss(logits, targets, mu, logvar, prop_pred, prop_true, pad_idx,
             beta=1.0, free_bits=config.FREE_BITS, prop_weight=config.PROP_LOSS_WEIGHT):
    """Returns (total, dict of components). Recon + KL are per-molecule (mean over batch)."""
    B = targets.size(0)
    recon = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1),
        ignore_index=pad_idx, reduction="sum",
    ) / B

    # KL with per-dimension free bits (anti collapse)
    kl_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())     # [B, latent]
    kl = torch.clamp(kl_dim, min=free_bits).sum(dim=1).mean()
    kl_raw = kl_dim.sum(dim=1).mean()

    prop = F.mse_loss(prop_pred, prop_true)

    total = recon + beta * kl + prop_weight * prop
    return total, {
        "recon": recon.item(),
        "kl": kl_raw.item(),
        "prop": prop.item(),
        "beta": beta,
    }


def build_model(vocab, device) -> SelfiesVAE:
    model = SelfiesVAE(vocab_size=len(vocab), pad_idx=vocab.pad, unk_idx=vocab.unk).to(device)
    return model


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
