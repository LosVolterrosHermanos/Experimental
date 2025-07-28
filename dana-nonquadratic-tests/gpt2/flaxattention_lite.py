"""
Lite version of flaxattention for GPT-2 integration.

This is a simplified version of the flaxattention library that includes the
core functionality needed for GPT-2 attention implementation, specifically the
Pallas-based attention kernel with proper gradient support.
"""

import math
import functools
from typing import Callable, Optional, Any

import jax
import jax.numpy as jnp
from jax import lax
from jax import Array
import numpy as np

# Import the Pallas-based multi-head attention kernel
try:
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    PALLAS_AVAILABLE = True
except ImportError:
    PALLAS_AVAILABLE = False
    pl = None
    plgpu = None


DEFAULT_MASK_VALUE = -0.7 * float(np.finfo(np.dtype("float32")).max)


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


def segment_mask(
    q_segment_ids: jax.Array,
    kv_segment_ids: jax.Array,
):
    # [B, T, 1] or [T, 1]
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    # [B, 1, S] or [1, S]
    if kv_segment_ids.ndim == 1:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
    else:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)


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

        # Apply causal mask and segment mask if needed
        if causal or segment_ids_ref is not None:
            mask = None
            if segment_ids_ref is not None:
                kv_segment_ids = pl.load(segment_ids_ref, (curr_k_slice,))
                mask = segment_mask(q_segment_ids, kv_segment_ids)
            if causal:
                causal_mask = span_q[:, None] >= span_k[None, :]
                mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
            qk = jnp.where(mask, qk, DEFAULT_MASK_VALUE)

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


def _preprocess_backward_kernel(out_ref, dout_ref, delta_ref):
    # load
    o = out_ref[...].astype(jnp.float32)
    do = dout_ref[...].astype(jnp.float32)
    # compute
    delta = jnp.sum(o * do, axis=1)
    # write-back
    delta_ref[...] = delta.astype(delta_ref.dtype)


@jax.named_scope("preprocess_backward")
def _preprocess_backward(out, do, lse, block_q: int, debug: bool, interpret: bool):
    batch_size, seq_len, num_heads, head_dim = out.shape
    out_shape = jax.ShapeDtypeStruct(lse.shape, lse.dtype)
    delta = pl.pallas_call(
        _preprocess_backward_kernel,
        grid=(pl.cdiv(seq_len, block_q), batch_size, num_heads),
        in_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        ],
        out_specs=pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        compiler_params=dict(triton=dict(num_warps=4, num_stages=3)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_preprocess_backward",
    )(out, do)
    return delta


# Simplified backward pass kernel (for basic cases)
def mha_backward_kernel(
    # Inputs
    q_ref,
    k_ref,
    v_ref,
    segment_ids_ref,
    out_ref,
    do_scaled_ref,
    lse_ref,
    delta_ref,
    # Outputs
    dq_ref,
    dk_ref,
    dv_ref,
    *,
    sm_scale: float,
    causal: bool,
    block_q1: int,
    block_k1: int,
    block_q2: int,
    block_k2: int,
    block_d: int,
    score_mod=None,
    mask_mod=None,
    score_mod_grad=None,
):
    """Simplified backward kernel - in practice, would use XLA fallback for gradients."""
    # For simplicity, we'll just initialize zero gradients
    # In practice, the full implementation would compute proper gradients
    dq_ref[...] = jnp.zeros_like(dq_ref[...])
    dk_ref[...] = jnp.zeros_like(dk_ref[...])
    dv_ref[...] = jnp.zeros_like(dv_ref[...])


@functools.partial(
    jax.custom_vjp, nondiff_argnums=[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
)
@functools.partial(
    jax.jit,
    static_argnames=[
        "sm_scale",
        "causal",
        "block_q",
        "block_k",
        "backward_pass_impl",
        "num_warps",
        "num_stages",
        "grid",
        "interpret",
        "debug",
        "score_mod",
        "mask_mod",
        "score_mod_grad",
    ],
)
def mha_pallas(
    q,
    k,
    v,
    segment_ids,
    sm_scale: float = 1.0,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    backward_pass_impl: str = "xla",
    num_warps: Optional[int] = None,
    num_stages: int = 2,
    grid=None,
    interpret: bool = False,
    debug: bool = False,
    score_mod=None,
    mask_mod=None,
    score_mod_grad=None,
):
    """Pallas-based multi-head attention implementation with gradient support."""
    if not PALLAS_AVAILABLE:
        raise RuntimeError("Pallas is not available. Please install JAX with Pallas support.")
    
    del backward_pass_impl
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    
    # Set default grid and warps
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)
    
    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8
    
    # Set default score and mask mods
    if score_mod is None:
        score_mod = _identity_score_mod
    if mask_mod is None:
        mask_mod = _causal_mask_mod if causal else lambda *args: True

    kernel = functools.partial(
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
        grid=grid_,
        in_specs=in_specs,
        out_specs=pl.BlockSpec(
            (None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)
        ),
        compiler_params=plgpu.TritonCompilerParams(
            num_warps=num_warps_, num_stages=num_stages
        ),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, segment_ids)


def _mha_forward(
    q,
    k,
    v,
    segment_ids,
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    score_mod,
    mask_mod,
    score_mod_grad,
):
    del backward_pass_impl, score_mod_grad
    batch_size, q_seq_len, num_heads, head_dim = q.shape
    kv_seq_len = k.shape[1]
    block_q = min(block_q, q_seq_len)
    block_k = min(block_k, kv_seq_len)
    
    # Set default grid and warps
    grid_ = grid
    if grid_ is None:
        grid_ = (pl.cdiv(q_seq_len, block_q), batch_size, num_heads)

    num_warps_ = num_warps
    if num_warps_ is None:
        num_warps_ = 4 if head_dim <= 64 else 8

    # Set default score and mask mods
    if score_mod is None:
        score_mod = _identity_score_mod
    if mask_mod is None:
        mask_mod = _causal_mask_mod if causal else lambda *args: True

    kernel = functools.partial(
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
    out_shape = [
        jax.ShapeDtypeStruct(shape=q.shape, dtype=q.dtype),  # out
        jax.ShapeDtypeStruct(
            shape=(batch_size, num_heads, q_seq_len),
            dtype=jnp.float32,  # lse
        ),
    ]
    in_specs = [
        pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
        pl.BlockSpec((None, kv_seq_len, None, head_dim), lambda _, j, k: (j, 0, k, 0)),
    ]
    in_specs.append(
        None if segment_ids is None
        else pl.BlockSpec((None, kv_seq_len), lambda _, j, k: (j, 0))
    )
    out, lse = pl.pallas_call(
        kernel,
        grid=grid_,
        in_specs=in_specs,
        out_specs=[
            pl.BlockSpec((None, block_q, None, head_dim), lambda i, j, k: (j, i, k, 0)),
            pl.BlockSpec((None, None, block_q), lambda i, j, k: (j, k, i)),
        ],
        compiler_params=dict(triton=dict(num_warps=num_warps_, num_stages=num_stages)),
        out_shape=out_shape,
        debug=debug,
        interpret=interpret,
        name="mha_forward",
    )(q, k, v, segment_ids)
    return out, (q, k, v, segment_ids, out, lse)


@functools.partial(jax.jit, static_argnames=["sm_scale", "causal"])
def mha_reference(
    q,
    k,
    v,
    segment_ids,
    sm_scale=1.0,
    causal: bool = False,
):
    """Reference implementation using standard JAX operations."""
    q_seq_len = q.shape[1]
    kv_seq_len = k.shape[1]
    logits = jnp.einsum("bqhc,bkhc->bhqk", q, k).astype(jnp.float32)
    mask = None
    if segment_ids is not None:
        mask = jnp.expand_dims(segment_mask(segment_ids, segment_ids), 1)
        mask = jnp.broadcast_to(mask, logits.shape)
    if causal:
        causal_mask = jnp.tril(jnp.ones((1, 1, q_seq_len, kv_seq_len), dtype=bool))
        causal_mask = jnp.broadcast_to(causal_mask, logits.shape)
        mask = causal_mask if mask is None else jnp.logical_and(mask, causal_mask)
    logits = logits if mask is None else jnp.where(mask, logits, float("-inf"))
    weights = jax.nn.softmax(logits * sm_scale).astype(q.dtype)
    return jnp.einsum("bhqk,bkhc->bqhc", weights, v)


def _mha_backward(
    sm_scale: float,
    causal: bool,
    block_q: int,
    block_k: int,
    backward_pass_impl: str,
    num_warps: int | None,
    num_stages: int,
    grid: Any,
    interpret: bool,
    debug: bool,
    score_mod,
    mask_mod,
    score_mod_grad,
    res,
    do,
):
    """Backward pass implementation - falls back to XLA for gradients."""
    del num_warps, num_stages, grid, interpret, debug, score_mod, mask_mod, score_mod_grad
    q, k, v, segment_ids, out, lse = res

    # Always use XLA for backward pass in this lite version
    return jax.vjp(
        functools.partial(mha_reference, sm_scale=sm_scale, causal=causal),
        q,
        k,
        v,
        segment_ids,
    )[1](do)


# Define custom VJP for the mha_pallas function
mha_pallas.defvjp(_mha_forward, _mha_backward)


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
    
    # The flaxattention pallas kernel expects BLHD format, but we have BTNH
    # We need to transpose to match the expected format
    q_transposed = jnp.transpose(query, (0, 1, 2, 3))  # Already BTNH -> BLHD
    k_transposed = jnp.transpose(key, (0, 1, 2, 3))    # Already BTNH -> BLHD  
    v_transposed = jnp.transpose(value, (0, 1, 2, 3))  # Already BTNH -> BLHD
    
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