#!/usr/bin/env python
"""
Test script to verify the cudnn attention implementation works correctly.
"""

import jax
import jax.numpy as jnp
from nanogpt_rope_mixed_precision import ModelConfig, GPTWithRoPE

def test_attention_implementations():
    """Test all three attention implementations: naive, xla, and cudnn."""
    
    # Create test input
    batch_size = 2
    seq_len = 32
    rng = jax.random.PRNGKey(42)
    
    # Test configurations for each implementation
    implementations = ['naive', 'xla', 'cudnn']
    results = {}
    
    for impl in implementations:
        print(f"\nTesting {impl} implementation:")
        
        config = ModelConfig(
            vocab_size=1000,
            n_head=4,
            n_embd=128,
            block_size=64,
            n_layer=2,
            dropout_rate=0.0,  # No dropout for testing
            attention_implementation=impl
        )
        
        tokens = jax.random.randint(rng, (batch_size, seq_len), 0, config.vocab_size, dtype=jnp.uint16)
        
        try:
            # Test with mixed precision
            model = GPTWithRoPE(config, mixed_precision=True)
            params = model.init(jax.random.PRNGKey(0))
            logits = model.apply(params, tokens, True)
            
            print(f"  Mixed precision - Shape: {logits.shape}")
            print(f"  Mixed precision - Finite: {jnp.all(jnp.isfinite(logits))}")
            
            # Test with pure precision
            model_pure = GPTWithRoPE(config, mixed_precision=False)
            params_pure = model_pure.init(jax.random.PRNGKey(0))
            logits_pure = model_pure.apply(params_pure, tokens, True)
            
            print(f"  Pure precision - Shape: {logits_pure.shape}")
            print(f"  Pure precision - Finite: {jnp.all(jnp.isfinite(logits_pure))}")
            
            results[impl] = {'success': True, 'mixed': logits, 'pure': logits_pure}
            print(f"  {impl} implementation: SUCCESS")
            
        except Exception as e:
            print(f"  {impl} implementation: FAILED - {str(e)}")
            results[impl] = {'success': False, 'error': str(e)}
    
    # Compare results between implementations
    print("\nComparison between implementations:")
    successful_impls = [k for k, v in results.items() if v['success']]
    
    if len(successful_impls) > 1:
        ref_impl = successful_impls[0]
        for impl in successful_impls[1:]:
            # Compare shapes
            shape_match = (results[ref_impl]['mixed'].shape == results[impl]['mixed'].shape)
            print(f"  {ref_impl} vs {impl} - Shape match: {shape_match}")
    
    print(f"\nSuccessful implementations: {successful_impls}")
    print("Test completed!")

if __name__ == "__main__":
    test_attention_implementations()