"""
Orginally Taken verbatim from xformers library
https://github.com/facebookresearch/xformers/blob/bcb707576c6a80eaf850aa80e8643d3497ec2bc4/xformers/components/positional_embedding/rotary.py

The difference is that xformers seems to assume the inputs to be
(bs, head, seq_len, dim) while we assume (bs, seq_len, head, dim)

"""
# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.


# CREDITS: This implementation is inspired by GPT-NeoX https://github.com/EleutherAI/gpt-neox
# NOTE: Almost the same right now, moving parts to Triton is the next step

import os
from typing import Optional, Tuple

import numpy as np
import torch
import math


def rotate_half(x):
    # x1, x2 = x.chunk(2, dim=-1)
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=x1.ndim - 1)


# def rotate_two(x):
#     x1, x2 = x[..., ::2], x[..., 1::2]
#     return torch.cat((-x2, x1), dim=x1.ndim - 1)


@torch.jit.script
def apply_rotary_pos_emb(x, cos, sin, seq_dimension: int):
    # NOTE: This could probably be moved to Triton

    if seq_dimension == 0:
        cos = cos[: x.shape[0], None, None, :]
        sin = sin[: x.shape[0], None, None, :]
    elif seq_dimension == 1:
        # Handle a possible sequence length mismatch in between q and k
        cos = cos[None, : x.shape[1], None, :]
        sin = sin[None, : x.shape[1], None, :]

    return (x * cos) + (rotate_half(x) * sin)


def get_cached_cosine_sine(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    inv_freq: torch.Tensor,
    seqlen_offset: int = 0,
    position_scale: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache the tables for the sinusoidal position encoding.
    The only update the cache if the sequence length is larger than the
    previous one, or if the device / dtype changes.
    Otherwise the cache is returned.

    The values get cached in the function itself, so that the cache is
    shared across all instances of the class (so shared across all layers
    of the Transformer).

    Args:
        seq_len (int):
            The sequence length of the input tensor
        device (torch.device):
            The device of the input tensor
        dtype (torch.dtype):
            The dtype of the input tensor
        inv_freq (torch.Tensor):
            The base inverse frequency of the sinusoidal encoding
        seqlen_offset (int):
            past sequence length
        

    Raises:
        ValueError:
            If seq_dimension is not 0 or 1

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            The cosine and sine tables

    """
    if (
        not hasattr(get_cached_cosine_sine, "_seq_len_cached")
        # new sequence length is larger than the cached value
        or get_cached_cosine_sine._seq_len_cached < seq_len
        # model dimension is different from the cached value
        or get_cached_cosine_sine._dim_model != inv_freq.shape[0]
        # different device
        or get_cached_cosine_sine._cos_cached.device != device
        # different dtype
        or get_cached_cosine_sine._cos_cached.dtype != dtype
        or get_cached_cosine_sine._position_scale != position_scale
    ):
        t = torch.arange(seq_len, device=device, dtype=torch.float32) * position_scale
        # shape: (seq_len, dim_model // 2)
        freqs = torch.einsum("i,j->ij", t, inv_freq.to(dtype))
        # shape: (seq_len, dim_model)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        # Cache the tables and metadata in the function closure
        get_cached_cosine_sine._seq_len_cached = seq_len
        get_cached_cosine_sine._dim_model = inv_freq.shape[0]
        get_cached_cosine_sine._cos_cached = emb.cos().to(dtype)
        get_cached_cosine_sine._sin_cached = emb.sin().to(dtype)
        get_cached_cosine_sine._position_scale = position_scale

    return (get_cached_cosine_sine._cos_cached[seqlen_offset:seq_len], 
            get_cached_cosine_sine._sin_cached[seqlen_offset:seq_len])


"""
def get_cached_cosine_sine(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    inv_freq: torch.Tensor,
    seqlen_offset: int = 0,
    position_scale: float = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
"""

# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
def _yarn_find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

def get_cached_cosine_sine_revised_yarn(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    dim: int,
    base=10000,
    scale=1,
    beta_fast=32,
    beta_slow=1,
    extrapolation_factor=1,
    attn_factor=1,
    original_max_position_embeddings=2048,
    seqlen_offset: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cache the tables for the sinusoidal position encoding using revised yarn logic.
    The cache is updated if the sequence length is larger than the previous one,
    or if the device / dtype changes. The cache is shared across all instances
    of the class (so shared across all layers of the Transformer).

    Args:
        seq_len (int): The sequence length of the input tensor
        device (torch.device): The device of the input tensor
        dtype (torch.dtype): The dtype of the input tensor
        dim (int): Dimension of the model
        Other arguments are specific to the revised yarn logic.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cached cosine and sine tables
    """

    # Check if cached values need to be updated
    if (not hasattr(get_cached_cosine_sine_revised_yarn, "_seq_len_cached") or
        get_cached_cosine_sine_revised_yarn._seq_len_cached < seq_len or
        get_cached_cosine_sine_revised_yarn._dim_model != dim or
        get_cached_cosine_sine_revised_yarn._cos_cached.device != device or
        get_cached_cosine_sine_revised_yarn._cos_cached.dtype != dtype):

        # Calculate inverse frequencies using revised yarn logic
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
        low, high = _yarn_find_correction_range(beta_fast, beta_slow, dim, base, original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, dim // 2).float().to(device)) * extrapolation_factor
        inv_freq = inv_freq / ((1 - inv_freq_mask) * scale + inv_freq_mask)

        # Scale adjustment
        mscale = float(_yarn_get_mscale(scale) * attn_factor)

        # Generate sinusoidal embeddings
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.einsum("i,j->ij", t, inv_freq.to(dtype))
        emb = torch.cat((freqs, freqs), dim=-1).to(device)

        # Cache the computed values
        get_cached_cosine_sine_revised_yarn._seq_len_cached = seq_len
        get_cached_cosine_sine_revised_yarn._dim_model = dim
        get_cached_cosine_sine_revised_yarn._cos_cached = (emb.cos() * mscale).to(dtype)
        get_cached_cosine_sine_revised_yarn._sin_cached = (emb.sin() * mscale).to(dtype)

    return (get_cached_cosine_sine_revised_yarn._cos_cached[seqlen_offset:seq_len], 
            get_cached_cosine_sine_revised_yarn._sin_cached[seqlen_offset:seq_len])


class RotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.
    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration
    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox
    .. warning: Please note that this embedding is not registered on purpose, as it is transformative
        (it does not create the embedding dimension) and will likely be picked up (imported) on a ad-hoc basis

    # Arguments
    :param dim_mode: head dimention
    :param max_seq_len:
    :param default_seq_dimension: which dim is the sequence length
    :param dtype: cos/sin dtype
    :param use_fused_kernel: if to use customized fused kernel.
        Note: if used, q, k will be modified inplace. Ok for both forward & backward.
    """

    def __init__(
        self,
        dim_model: int,
        *,
        max_seq_len: Optional[int] = None,
        dtype: Optional[torch.dtype] = None,
        use_fused_kernel=False,
        base=10000,
        position_scale=1,
    ):
        super().__init__()
        self.base = base
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim_model, 2).float() / dim_model))
        self.register_buffer("inv_freq", inv_freq)
        self.use_fused_kernel = use_fused_kernel
        self.position_scale = position_scale
        if (
            max_seq_len is not None
            and max_seq_len is not None
            and dtype is not None
        ):
            device = torch.cuda.current_device()
            # setup the cache tables
            get_cached_cosine_sine(
                seq_len=max_seq_len,
                device=device,
                dtype=dtype,
                inv_freq=self.inv_freq.to(device),
                position_scale=position_scale,
            )

    def forward(
        self, q: torch.Tensor,
        k: torch.Tensor,
        seq_dimension: int = 1,
        seqlen_offset: int = 0,
        position_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k does not include `seqlen_offset`
        q: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        k: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        """
        if seq_dimension < 0:
            seq_dimension = k.ndim + seq_dimension
        assert seq_dimension in (0, 1)
        seq_len = k.shape[seq_dimension] + seqlen_offset

        position_scale = position_scale or self.position_scale
        cos_cached, sin_cached = get_cached_cosine_sine(
            seq_len=seq_len,
            device=k.device,
            dtype=k.dtype,
            inv_freq=self.inv_freq,
            seqlen_offset=seqlen_offset,
            position_scale=position_scale,
        )


        return (
                apply_rotary_pos_emb(
                    q, cos_cached, sin_cached, seq_dimension=seq_dimension
                ),
                apply_rotary_pos_emb(
                    k, cos_cached, sin_cached, seq_dimension=seq_dimension
                ),
            )


class RevisedYaRNRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, beta_fast=32, beta_slow=1, extrapolation_factor=1, attn_factor=1, original_max_position_embeddings=2048, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.original_max_position_embeddings = original_max_position_embeddings
        self.device = device or torch.device('cpu')


    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_dimension: int = 1, seqlen_offset: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        k: Either (bs, seq_len, num_heads, head_dim) or (seq_len, bs, num_heads, head_dim)
        """
        if seq_dimension < 0:
            seq_dimension = k.ndim + seq_dimension
        assert seq_dimension in (0, 1), "seq_dimension must be 0 or 1"
        seq_len = k.shape[seq_dimension] + seqlen_offset

        cos_cached, sin_cached = get_cached_cosine_sine_revised_yarn(
            seq_len=seq_len,
            device=k.device,
            dtype=k.dtype,
            dim=self.dim,
            base=self.base,
            scale=self.scale,
            beta_fast=self.beta_fast,
            beta_slow=self.beta_slow,
            extrapolation_factor=self.extrapolation_factor,
            attn_factor=self.attn_factor,
            max_position_embeddings=self.original_max_position_embeddings,
            seqlen_offset=seqlen_offset,
        )

        return (
            apply_rotary_pos_emb(q, cos_cached, sin_cached, seq_dimension=seq_dimension),
            apply_rotary_pos_emb(k, cos_cached, sin_cached, seq_dimension=seq_dimension),
        )
