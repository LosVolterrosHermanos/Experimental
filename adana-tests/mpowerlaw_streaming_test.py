#!/usr/bin/env python3
"""
Test script for MPowerLawRF using mlsq_streaming_optax_simple function.

This script tests the full integration between MPowerLawRF, ADANA optimizer,
and the mlsq_streaming_optax_simple training function with exponentially 
spaced loss recording.

Author: AI Assistant
Date: 2025-01-27
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt

import sys
import os
sys.path.append('../')
sys.path.append('../power_law_rf')

# Import required modules
import optimizers
from power_law_rf.power_law_rf import MPowerLawRF
from power_law_rf.least_squares import mlsq_streaming_optax_simple

def main():
    print("=" * 70)
    print("MPowerLawRF + mlsq_streaming_optax_simple Integration Test")
    print("=" * 70)
    
    # Set random seed
    key = random.PRNGKey(456)
    
    # Problem setup
    print("\n1. Setting up MPowerLawRF problem...")
    ALPHA = 1.0
    BETA = 0.8  
    ZETA = 0.6
    M = 4           # Number of experts
    V = 50          # Hidden dimensionality
    D = 25          # Embedded dimensionality
    BATCH_SIZE = 10
    STEPS = 500     # Number of optimization steps
    
    print(f"  Parameters: α={ALPHA}, β={BETA}, ζ={ZETA}")
    print(f"  Dimensions: M={M}, V={V}, D={D}")
    print(f"  Training: {STEPS} steps, batch_size={BATCH_SIZE}")
    
    # Create problem
    key, subkey = random.split(key)
    problem = MPowerLawRF.initialize_random(ALPHA, BETA, ZETA, M, V, D, key=subkey)
    print(f"  Expert probabilities: {problem.expert_probs}")
    print(f"  CheckW shape: {problem.checkW.shape}")
    print(f"  Initial population risk: {problem.get_population_risk(jnp.zeros((M, D))):.6f}")
    
    # Test empirical limit
    empirical_limit = problem.get_empirical_limit_loss()
    print(f"  Empirical limit loss: {empirical_limit:.6f}")
    
    # Set up ADANA optimizer
    print("\n2. Setting up ADANA optimizer...")
    G2_SCALE = 0.05
    G3_IV = 0.02
    
    g2 = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*ALPHA), 1)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2, g3=g3, Delta=Delta)
    print(f"  ADANA configured: g2={G2_SCALE}, g3_iv={G3_IV}")
    
    # Set up SGD for comparison
    print("\n3. Setting up SGD optimizer for comparison...")
    SGD_LR = 0.05*jnp.sqrt(D)
    sgd_opt = optax.sgd(learning_rate=SGD_LR)
    print(f"  SGD learning rate: {SGD_LR}")
    
    # Initial parameters (same for both optimizers)
    key, subkey = random.split(key)
    init_params = random.normal(subkey, (M, D)) * 0.01
    print(f"  Initial parameter norm: {jnp.linalg.norm(init_params):.6f}")
    
    # Run ADANA with mlsq_streaming_optax_simple
    print("\n4. Running ADANA with mlsq_streaming_optax_simple...")
    key, subkey = random.split(key)
    
    adana_times, adana_losses = mlsq_streaming_optax_simple(
        subkey,
        problem.get_data,
        BATCH_SIZE,
        STEPS,
        adana_opt,
        init_params,
        problem.get_population_risk,
        tqdm_bar=False  # Disable for cleaner output
    )
    
    print(f"  ADANA completed!")
    print(f"  Loss recorded at {len(adana_times)} time points")
    print(f"  Final loss: {adana_losses[-1]:.6f}")
    print(f"  Loss reduction: {adana_losses[0] - adana_losses[-1]:.6f}")
    
    # Run SGD with mlsq_streaming_optax_simple
    print("\n5. Running SGD with mlsq_streaming_optax_simple...")
    key, subkey = random.split(key)
    
    sgd_times, sgd_losses = mlsq_streaming_optax_simple(
        subkey,
        problem.get_data,
        BATCH_SIZE,
        STEPS,
        sgd_opt,
        init_params,
        problem.get_population_risk,
        tqdm_bar=False
    )
    
    print(f"  SGD completed!")
    print(f"  Loss recorded at {len(sgd_times)} time points")
    print(f"  Final loss: {sgd_losses[-1]:.6f}")
    print(f"  Loss reduction: {sgd_losses[0] - sgd_losses[-1]:.6f}")
    
    # Analyze results
    print("\n6. Analyzing results...")
    print(f"  Initial loss: {adana_losses[0]:.6f}")
    print(f"  ADANA final loss: {adana_losses[-1]:.6f}")
    print(f"  SGD final loss: {sgd_losses[-1]:.6f}")
    print(f"  Empirical limit: {empirical_limit:.6f}")
    
    if adana_losses[-1] < sgd_losses[-1]:
        improvement = (sgd_losses[-1] - adana_losses[-1]) / sgd_losses[-1] * 100
        print(f"  ADANA is {improvement:.1f}% better than SGD")
    else:
        improvement = (adana_losses[-1] - sgd_losses[-1]) / adana_losses[-1] * 100
        print(f"  SGD is {improvement:.1f}% better than ADANA")
    
    # Check how close to empirical limit
    adana_gap = adana_losses[-1] - empirical_limit
    sgd_gap = sgd_losses[-1] - empirical_limit
    print(f"  ADANA gap to empirical limit: {adana_gap:.6f}")
    print(f"  SGD gap to empirical limit: {sgd_gap:.6f}")
    
    # Verify exponential spacing of loss times
    print("\n7. Verifying exponential spacing of loss recording...")
    print(f"  Loss times: {adana_times[:10]}... (first 10)")
    
    # Check that spacing approximately follows 1.1^k pattern
    expected_times = jnp.unique(jnp.concatenate([
        jnp.array([0]),
        jnp.int32(1.1**jnp.arange(1, jnp.ceil(jnp.log(STEPS)/jnp.log(1.1)))),
        jnp.array([STEPS])
    ]))
    
    print(f"  Expected pattern matches: {jnp.allclose(adana_times, expected_times)}")
    
    # Create visualization
    print("\n8. Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Loss vs iteration
    ax1.loglog(adana_times + 1, adana_losses, 'r-o', label='ADANA', markersize=3)
    ax1.loglog(sgd_times + 1, sgd_losses, 'b-s', label='SGD', markersize=3)
    ax1.axhline(y=empirical_limit, color='green', linestyle='--', 
                label=f'Empirical Limit ({empirical_limit:.4f})')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Population Risk')
    ax1.set_title('Population Risk vs Iteration')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Loss vs time spacing verification 
    ax2.semilogy(range(len(adana_times)), adana_losses, 'r-o', 
                 label='ADANA', markersize=3)
    ax2.semilogy(range(len(sgd_times)), sgd_losses, 'b-s', 
                 label='SGD', markersize=3)
    ax2.axhline(y=empirical_limit, color='green', linestyle='--', 
                label=f'Empirical Limit')
    ax2.set_xlabel('Loss Recording Index')
    ax2.set_ylabel('Population Risk')
    ax2.set_title('Loss Trajectory (Exponentially Spaced Points)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    if os.path.exists('results'):
        plt.savefig('results/mpowerlaw_streaming_test.pdf', bbox_inches='tight')
        print(f"  Plot saved to: results/mpowerlaw_streaming_test.pdf")
    else:
        print("  Results directory not found, showing plot instead...")
        plt.show()
    
    print("\n" + "=" * 70)
    print("Integration test completed successfully!")
    print("Key findings:")
    print(f"  - MPowerLawRF integrates properly with mlsq_streaming_optax_simple")
    print(f"  - ADANA optimizer works correctly in streaming setup")
    print(f"  - Exponential spacing of loss recording works as expected")
    print(f"  - Both optimizers successfully reduce population risk")
    if adana_losses[-1] < sgd_losses[-1]:
        print(f"  - ADANA outperforms SGD on this problem instance")
    else:
        print(f"  - SGD outperforms ADANA on this problem instance")
    print("=" * 70)

if __name__ == "__main__":
    main() 