# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

# Code from https://github.com/facebookresearch/DiT/blob/main/models.py


import math

import clip
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
from transformers import AutoTokenizer, T5EncoderModel

from .helpers import *


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
class StackLinear(nn.Module): #( 128 *2, 4, 128)
    def __init__(self, quant_factor=2, unstack=False, seq_first=True):
        super().__init__()
        self.quant_factor = quant_factor
        self.latent_frame_size = 2**quant_factor
        self.unstack = unstack
        self.seq_first = seq_first

    def forward(self, x):
        if self.seq_first:
            B, T, F = x.shape # (BS,64,256)
        else:
            B, F, T = x.shape
            x = x.permute(0, 2, 1)

        if not self.unstack:# stack
            assert T % self.latent_frame_size == 0, "T must be divisible by latent_frame_size"
            T_latent = T // self.latent_frame_size
            F_stack = F * self.latent_frame_size
            x = x.reshape(B, T_latent, F_stack)
        else: #unstack
            F_stack = F // self.latent_frame_size
            x = x.reshape(B, T * self.latent_frame_size, F_stack)

        if not self.seq_first:
            x = x.permute(0, 2, 1)

        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                    These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class DiT1DBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) self-attn/MLP conditioning,
    plus a T5 token cross-attn inserted between self-attn and MLP.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)

        # T5 토큰 cross-attn: adaLN 없이 plain LN + zero-init gate (PixArt-alpha 방식).
        # 학습 초반엔 cross_gate=0이라 identity로 시작해서 self-attn/MLP 학습을 방해하지 않음.
        self.norm_cross = nn.LayerNorm(hidden_size, eps=1e-6)
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=0.1, batch_first=True)
        self.cross_gate = nn.Parameter(torch.zeros(1, 1, hidden_size))

        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, key_padding_mask=None, attn_mask=None, text_kv=None, text_key_padding_mask=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        scaled_attn = modulate(self.norm1(x), shift_msa, scale_msa) #정규화된 x를 c 기반으로 shift/scale
        attn_output, _ = self.attn(scaled_attn,scaled_attn,scaled_attn,key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = x + gate_msa.unsqueeze(1) * attn_output #attention 결과를 c 기반 gate로 조절 후 residual

        if text_kv is not None:
            cross_out, _ = self.cross_attn(self.norm_cross(x), text_kv, text_kv, key_padding_mask=text_key_padding_mask)
            x = x + self.cross_gate * cross_out

        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp)) #MLP 동일
        return x


class FinalLayer1D(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT1DforFlowDecoder(nn.Module):
    """ DiT Refiner for motion prior learning
    code adapted from https://github.com/facebookresearch/DiT/blob/main/models.py

    text_condition=True면:
    - CLIP pooled 문장 임베딩을 time embedding에 더해서 adaLN(전역 conditioning)으로 주입
    - T5 토큰 시퀀스를 각 블록의 cross-attn K/V(단어 단위 conditioning)로 주입
    """
    def __init__(self, out_dim=263, embed_dim=384, c_in_dim=512, c_proj_dim=512, num_heads=8, mlp_ratio=4, depth=6, max_seq_len=200,
                text_condition=False,
                t5_version="google/t5-v1_1-base",
                t5_max_text_len=20):
        super().__init__()
        self.in_dim = out_dim
        self.blocks = nn.ModuleList([
            DiT1DBlock(
                hidden_size=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
            ) for _ in range(depth)
        ])
        self.in_proj = nn.Linear(out_dim + c_proj_dim, embed_dim)
        self.out_proj = FinalLayer1D(embed_dim, out_dim)

        self.t_embedder = TimestepEmbedder(hidden_size=embed_dim)

        self.post_norm = nn.LayerNorm(embed_dim)
        self.text_condition = text_condition
        if text_condition:
            self.text_embed = nn.Linear(512, embed_dim, bias=True)
            self.clip_model = self.load_and_freeze_clip("ViT-B/32")

            self.t5_max_text_len = t5_max_text_len
            self.t5_tokenizer = AutoTokenizer.from_pretrained(t5_version, legacy=False)
            self.t5_model = T5EncoderModel.from_pretrained(t5_version).eval()
            for p in self.t5_model.parameters():
                p.requires_grad = False
            self.t5_proj = nn.Linear(self.t5_model.config.d_model, embed_dim)

        # alibi future temporal bias mask (프레임 간 self-attn에 거리 기반 bias)
        self.attn_mask = init_faceformer_biased_mask_future(num_heads, max_seq_len)

        # y(VQVAE latent) -> motion 프레임 수로 업샘플 + projection
        self.cin_proj = nn.Sequential(
            StackLinear(quant_factor=2, seq_first=True, unstack=False), # (B, dim, T) -> (B, dim, T//4, 4) -> (B, dim*4, T//4)
            nn.Linear(c_in_dim*4, c_in_dim*4, 1),
            StackLinear(quant_factor=2, seq_first=True, unstack=True), # (B, T//4, in_dim*4) -> (B, T, in_dim)
            nn.Linear(c_in_dim, c_proj_dim, 1),
        )

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device="cpu", jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def encode_text_t5(self, raw_text):
        """cross-attn용 T5 토큰별 hidden state. return: (hidden (B, L, d_model), key_padding_mask (B, L), True=pad)"""
        device = next(self.parameters()).device
        tok = self.t5_tokenizer(
            raw_text,
            max_length=self.t5_max_text_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = tok['input_ids'].to(device)
        attention_mask = tok['attention_mask'].to(device)
        with torch.no_grad():
            hidden = self.t5_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return hidden, ~attention_mask.bool()

    def forward(self, x, times, y, padding_mask=None, text_embedding=None, t5_embedding=None, t5_padding_mask=None):
        # x: (B, T, in_dim) t: (B, 1) c: (B, T, in_dim)  (x: noised_x, c: reconed_x)
        t = times
        t = self.t_embedder(t)  # (B, embed_dim)
        text_kv = None
        if self.text_condition:
            pooled_text = self.text_embed(text_embedding) # (BS, embed_dim)
            t = t + pooled_text
            if t5_embedding is not None:
                text_kv = self.t5_proj(t5_embedding) # (B, L_text, embed_dim)

        y = y.repeat_interleave(4, dim = 1)

        y = self.cin_proj(y) # (B, T, c_proj_dim)
        inputs = torch.cat([x, y], dim=-1) # (B, T, in_dim + c_proj_dim)
        inputs = self.in_proj(inputs) # (B, T, embed_dim)

        attn_mask = make_temporal_mask(inputs, self.attn_mask)

        for block in self.blocks:
            inputs = block(inputs, t, key_padding_mask=padding_mask, attn_mask=attn_mask,
                            text_kv=text_kv, text_key_padding_mask=t5_padding_mask)  # (B, T, embed_dim * 2)

        x = self.out_proj(inputs, t) # (B, T, out_dim )
        return x
