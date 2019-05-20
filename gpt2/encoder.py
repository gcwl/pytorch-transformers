# https://twitter.com/thom_wolf/status/1129658539142766592
# http://nlp.seas.harvard.edu/2018/04/03/attention.html
# https://github.com/openai/gpt-2/blob/master/src/model.py
# https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/encoders/transformer.py

import torch
import torch.nn as nn


class Transformer(nn.Module):
    """GPT-2 transformer"""

    def __init__(self, embed_dim, hidden_dim, num_embed, num_pos, num_heads, num_layers, dropout):
        super().__init__()
        self.tok_embedding = nn.Embedding(num_embed, embed_dim)
        self.pos_embedding = nn.Embedding(num_pos, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.attn, self.ff = nn.ModuleList(), nn.ModuleList()
        self.ln_1, self.ln_2 = nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.attn.append(nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout))
            self.ff.append(
                nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embed_dim)
                )
            )
            self.ln_1.append(nn.LayerNorm(embed_dim, eps=1e-12))
            self.ln_2.append(nn.LayerNorm(embed_dim, eps=1e-12))

    def forward(self, x):
        nx = len(x)
        p = torch.arange(nx).to(x.device).unsqueeze(-1)

        t = self.tok_embedding(x)
        t = self.dropout(t + self.pos_embedding(p).expand_as(t))

        attn_mask = torch.full((nx, nx), -float("Inf"), device=t.device, dtype=t.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)
        for attention, layernorm_1, feed_forward, layernorm_2 in zip(
            self.attn, self.ln_1, self.ff, self.ln_2
        ):
            s = layernorm_1(t)
            s, _ = attention(s, s, s, attn_mask=attn_mask, need_weights=False)
            t += self.dropout(s)
            t += self.dropout(feed_forward(layernorm_2(t)))
