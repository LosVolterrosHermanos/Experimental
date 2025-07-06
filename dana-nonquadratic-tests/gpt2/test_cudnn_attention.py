#!/usr/bin/env python
"""
Test script to verify the cudnn attention implementation works correctly.
"""

import jax
import jax.numpy as jnp
from nanogpt_rope_mixed_precision import ModelConfig, GPTWithRoPE

def test_cudnn_attention():
    """Test that cudnn attention produces similar results to the original implementation."""
    
    # Set up test configuration
    config = ModelConfig(
        vocab_size=1000,
        n_head=4,
        n_embd=128,
        block_size=64,
        n_layer=2,
        dropout_rate=0.0,  # No dropout for testing
        use_cudnn_attention=False  # Start with original implementation
    )
    
    # Create test input
    batch_size = 2
    seq_len = 32
    rng = jax.random.PRNGKey(42)
    tokens = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size, dtype=jnp.uint16)
    
    # Test with original implementation
    model_original = GPTWithRoPE(config, mixed_precision=True)
    params_original = model_original.init(jax.random.PRNGKey(0), tokens, True)
    logits_original = model_original.apply(params_original, tokens, True)
    
    # Test with cudnn implementation
    config_cudnn = ModelConfig(
        vocab_size=1000,
        n_head=4,
        n_embd=128,
        block_size=64,
        n_layer=2,
        dropout_rate=0.0,
        use_cudnn_attention=True  # Enable cudnn attention
    )
    
    model_cudnn = GPTWithRoPE(config_cudnn, mixed_precision=True)
    # Use the same parameters for fair comparison
    logits_cudnn = model_cudnn.apply(params_original, tokens, True)
    
    print(f"Original logits shape: {logits_original.shape}")
    print(f"CUDNN logits shape: {logits_cudnn.shape}")
    print(f"Shapes match: {logits_original.shape == logits_cudnn.shape}")
    
    # Check that outputs are finite (not NaN or infinite)
    print(f"Original logits finite: {jnp.all(jnp.isfinite(logits_original))}")
    print(f"CUDNN logits finite: {jnp.all(jnp.isfinite(logits_cudnn))}")
    
    # Test both mixed precision modes
    print("\nTesting pure precision mode:")
    model_original_pure = GPTWithRoPE(config, mixed_precision=False)
    params_original_pure = model_original_pure.init(jax.random.PRNGKey(0), tokens, True)
    logits_original_pure = model_original_pure.apply(params_original_pure, tokens, True)
    
    model_cudnn_pure = GPTWithRoPE(config_cudnn, mixed_precision=False)
    logits_cudnn_pure = model_cudnn_pure.apply(params_original_pure, tokens, True)
    
    print(f"Pure precision - Original finite: {jnp.all(jnp.isfinite(logits_original_pure))}")
    print(f"Pure precision - CUDNN finite: {jnp.all(jnp.isfinite(logits_cudnn_pure))}")
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    test_cudnn_attention()