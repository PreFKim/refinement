import torch
import torch.nn as nn
from typing import Any, Callable, List, Optional, Union
from torch import Tensor
from torch.nn import functional as F
import numpy as np 
from .embedding import RotaryEmbedding, PositionalEncoding

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = RotaryEmbedding(dim=d_model)

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]
    ) -> Tensor:
        qk = self.rotary.rotate_queries_or_keys(x) 
        x = self.self_attn(
            qk,
            qk,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class RefineNet(nn.Module):
    def __init__(self, 
                 num_layers=8, 
                 latent_dim=512,
                 feedforward_dim=1024,
                 nhead=8,
                 dropout=0.1,
                 activation=F.gelu
                 ):
        super(RefineNet,self).__init__()

        self.projection = nn.Linear(312, latent_dim)

        self.pos_emb = PositionalEncoding(latent_dim, dropout=dropout, batch_first=True)
        self.blk = nn.Sequential()
        for _ in range(num_layers):
            self.blk.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=nhead,
                    dim_feedforward=feedforward_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True
                )
            )
        
        self.out_layer = nn.Linear(latent_dim, 3)

    def forward(self, x):

        x = self.projection(x)
        x = self.pos_emb(x)
        for i, blk in enumerate(self.blk):
            x = blk(x)
        
        x = self.out_layer(x)

        return x
