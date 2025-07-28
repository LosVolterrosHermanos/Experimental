"""
Lite version of flaxattention for GPT-2 integration.

This is a simplified version of the flaxattention library that includes only the
core functionality needed for GPT-2 attention implementation, specifically the
Pallas-based attention kernel.
"""

import math
import jax
import jax.numpy as jnp
from jax import lax
from typing import Callable, Optional
from jax import Array
from functools import partial

# Import the Pallas-based multi-head attention kernel
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False
    pl = None
    plgpu = None


DEFAULT_MASK_VALUE = -0.7 * float(jnp.finfo(jnp.dtype("float32")).max)


def _identity_score_mod(
    score: Array,
    batch: Array,
    head: Array,
    token_q: Array,
    token_kv: Array,
) -> Array:
    """Identity score modification function."""
    return score


def _causal_mask_mod(
    batch: Array,
    head: Array,
    token_q: Array,
    token_kv: Array,
) -> Array:
    """Causal mask modification function."""
    return token_q[:, None] >= token_kv[None, :]


def mha_forward_kernel(
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref,
    o_ref,
    *residual_refs,
    num_heads: int,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_d: int,
    block_k: int,
    score_mod=None,
    mask_mod=None,
):
    """Pallas kernel for multi-head attention forward pass."""
    seq_len = k_ref.shape[0]
    start_q = pl.program_id(0)
    start_b = pl.program_id(1)
    start_h = pl.program_id(2)

    # Initialize accumulators
    m_i = jnp.zeros(block_q, dtype=jnp.float32) - float("inf")
    l_i = jnp.zeros(block_q, dtype=jnp.float32)
    o = jnp.zeros((block_q, block_d), dtype=jnp.float32)

    # Load query block
    curr_q_slice = pl.dslice(start_q * block_q, block_q)
    q = q_ref[...]
    q_segment_ids = (
        None if segment_ids_ref is None else pl.load(segment_ids_ref, (curr_q_slice,))
    )

    RCP_LN2 = 1.44269504

    def body(start_k, carry):
        o_prev, m_prev, l_prev = carry
        curr_k_slice = pl.dslice(start_k * block_k, block_k)

        k = pl.load(k_ref, (curr_k_slice, slice(None)))
        qk = pl.dot(q, k.T)
        if sm_scale != 1.0:
            qk *= sm_scale

        # Apply score and mask modifications
        span_q = start_q * block_q + jnp.arange(block_q)
        span_k = start_k * block_k + jnp.arange(block_k)
        
        if score_mod is not None:
            qk = score_mod(qk, start_b, start_h, span_q, span_k)
        if mask_mod is not None:
            qk = jnp.where(
                mask_mod(start_b, start_h, span_q, span_k), qk, DEFAULT_MASK_VALUE
            )

        # Apply causal mask if needed
        if causal:
            causal_mask = span_q[:, None] >= span_k[None, :]
            qk = jnp.where(causal_mask, qk, DEFAULT_MASK_VALUE)

        qk *= RCP_LN2
        m_curr = qk.max(axis=-1)
        m_next = jnp.maximum(m_prev, m_curr)
        m_next = jnp.where(m_next == DEFAULT_MASK_VALUE, 0.0, m_next)
        
        correction = jnp.exp2(m_prev - m_next)
        l_prev_corr = correction * l_prev
        s_curr = jnp.exp2(qk - m_next[:, None])
        l_curr = s_curr.sum(axis=-1)
        l_next = l_prev_corr + l_curr
        
        o_prev_corr = correction[:, None] * o_prev
        v = pl.load(v_ref, (curr_k_slice, pl.dslice(block_d)))
        o_curr = pl.dot(s_curr.astype(v.dtype), v)
        
        o_next = o_prev_corr + o_curr
        return o_next, m_next, l_next

    if causal:
        upper_bound = lax.div(block_q * (start_q + 1) + block_k - 1, block_k)
    else:
        upper_bound = pl.cdiv(seq_len, block_k)
    
    o, m_i, l_i = lax.fori_loop(0, upper_bound, body, (o, m_i, l_i))

    # Scale output
    l_i = jnp.where(l_i == 0.0, 1, l_i)
    o /= l_i[:, None]

    # Store LSE if needed
    if residual_refs:
        lse_ref = residual_refs[0]
        lse_ref[...] = m_i + jnp.log2(l_i)
    
    o_ref[...] = o.astype(o_ref.dtype)


@partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "num_warps",
        "num_stages",
        "score_mod",
        "mask_mod",
    ],
)
def mha_pallas(
    q,
    k,
    v,
    segment_ids=None,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    score_mod=None,
    mask_mod=None,
):
    """Pallas-based multi-head attention implementation."""
    if not PALLAS_AVAILABLE:
        raise RuntimeError("Pallas is not available. Please install JAX with Pallas support.")
    
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    
    # Set default grid and warps
    grid = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    num_warps = num_warps or (4 if head_dim <= 64 else 8)
    
    # Set default score and mask mods
    if score_mod is None:
        score_mod = _identity_score_mod
    if mask_mod is None:
        mask_mod = _causal_mask_mod if causal else lambda *args: True

    kernel = partial(
        mha_forward_kernel,
        num_heads=num_heads,
        sm_scale=sm_scale,
        causal=causal,
        block_q=block_q,
        block_k=block_k,
        block_d=head_dim,
        score_mod=score_mod,
        mask_mod=mask_mod,
    )

    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    in_specs.append(
        None if segment_ids is None
        else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    
    out_shape = jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype)
    return pl.pallas_call(
        kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
        ),
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps, num_stages=num_stages
        ),
        out_shape=out_shape,
        name="mha_forward",
    )(q, k, v, segment_ids)


def flax_attention_pallas(
    query: Array,
    key: Array,
    value: Array,
    scale: Optional[float] = None,
    is_causal: bool = True,
    block_q: int = 64,
    block_k: int = 64,
) -> Array:
    """
    Simplified FlaxAttention Pallas interface for GPT-2.
    
    Args:
        query: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, n_heads, head_dim)
        scale: Attention scale factor. If None, uses 1/sqrt(head_dim)
        is_causal: Whether to apply causal masking
        block_q: Query block size for Pallas kernel
        block_k: Key block size for Pallas kernel
    
    Returns:
        Attention output of shape (batch, seq_len, n_heads, head_dim)
    """
    if not PALLAS_AVAILABLE:
        raise RuntimeError("Pallas is not available. Cannot use fa-pallas attention method.")
    
    # Validate input shapes
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("Query, key, and value must be 4D tensors")
    
    batch_size, seq_len, n_heads, head_dim = query.shape
    
    # Set default scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # The flaxattention pallas kernel expects BLHD format, but we have BHLD
    # We need to transpose to match the expected format
    q_transposed = jnp.transpose(query, (0, 1, 2, 3))  # Already BLHD
    k_transposed = jnp.transpose(key, (0, 1, 2, 3))    # Already BLHD  
    v_transposed = jnp.transpose(value, (0, 1, 2, 3))  # Already BLHD
    
    # Apply Pallas attention
    output = mha_pallas(
        q_transposed,
        k_transposed, 
        v_transposed,
        segment_ids=None,
        sm_scale=scale,
        causal=is_causal,
        block_q=block_q,
        block_k=block_k,
    )
    
    return output


def math_attention_fallback(
    query: Array,
    key: Array,
    value: Array,
    scale: Optional[float] = None,
    is_causal: bool = True,
) -> Array:
    """
    Fallback math-based attention when Pallas is not available.
    
    Args:
        query: Query tensor of shape (batch, seq_len, n_heads, head_dim)
        key: Key tensor of shape (batch, seq_len, n_heads, head_dim)
        value: Value tensor of shape (batch, seq_len, n_heads, head_dim)
        scale: Attention scale factor. If None, uses 1/sqrt(head_dim)
        is_causal: Whether to apply causal masking
    
    Returns:
        Attention output of shape (batch, seq_len, n_heads, head_dim)
    """
    batch_size, seq_len, n_heads, head_dim = query.shape
    
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    
    # Compute attention scores: (B, T, N, H) x (B, S, N, H) -> (B, N, T, S)
    scores = jnp.einsum('btnh,bsnh->bnts', query, key) * scale
    
    # Apply causal mask if needed
    if is_causal:
        mask = jnp.tril(jnp.ones((seq_len, seq_len), dtype=jnp.bool_))[None, None, :, :]
        scores = jnp.where(mask, scores, float('-inf'))
    
    # Apply softmax
    attn_weights = jax.nn.softmax(scores, axis=-1)
    
    # Apply to values: (B, N, T, S) x (B, S, N, H) -> (B, T, N, H)
    output = jnp.einsum('bnts,bsnh->btnh', attn_weights, value)
    
    return output