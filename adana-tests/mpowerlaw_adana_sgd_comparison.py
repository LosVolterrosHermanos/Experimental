#!/usr/bin/env python3
"""
Comparison script for MPowerLawRF with ADANA vs SGD optimizers.

This script:
1. Creates a small instance of MPowerLawRF
2. Runs both ADANA and SGD optimizers for a few steps
3. Compares their behavior and convergence

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

def run_optimizer_steps(problem, optimizer, init_params, key, num_steps=10, batch_size=5):
    """Run optimizer for a few steps and return loss trajectory."""
    params = init_params
    opt_state = optimizer.init(params)
    losses = []
    pop_risks = []
    
    # Batch MSE loss function
    def batch_mse(params, embeddings, targets, expert_indices):
        y_pred = jnp.sum(embeddings * params[expert_indices], axis=1)
        if targets.ndim > 1:
            targets = targets.squeeze()
        return jnp.mean(optax.l2_loss(y_pred, targets))
    
    for step in range(num_steps):
        # Generate batch
        key, subkey = random.split(key)
        embeddings, targets, expert_indices = problem.get_data(subkey, batch_size)
        
        # Compute gradients and apply update
        loss_fn = lambda p: batch_mse(p, embeddings, targets, expert_indices)
        loss_val, grads = jax.value_and_grad(loss_fn)(params)
        
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        # Track metrics
        pop_risk = problem.get_population_risk(params)
        losses.append(float(loss_val))
        pop_risks.append(float(pop_risk))
    
    return losses, pop_risks, params

def main():
    print("=" * 70)
    print("MPowerLawRF: ADANA vs SGD Comparison")
    print("=" * 70)
    
    # Set random seed
    key = random.PRNGKey(123)
    
    # Problem setup
    print("\n1. Setting up problem...")
    ALPHA = 1.0
    BETA = 0.7  
    ZETA = 0.5
    M = 3
    V = 30
    D = 15
    BATCH_SIZE = 8
    NUM_STEPS = 15
    
    print(f"  Parameters: α={ALPHA}, β={BETA}, ζ={ZETA}")
    print(f"  Dimensions: M={M}, V={V}, D={D}")
    print(f"  Training: {NUM_STEPS} steps, batch_size={BATCH_SIZE}")
    
    # Create problem
    key, subkey = random.split(key)
    problem = MPowerLawRF.initialize_random(ALPHA, BETA, ZETA, M, V, D, key=subkey)
    print(f"  Expert probabilities: {problem.expert_probs}")
    
    # Initial parameters
    key, subkey = random.split(key)
    init_params = random.normal(subkey, (M, D)) * 0.01
    initial_pop_risk = problem.get_population_risk(init_params)
    print(f"  Initial population risk: {initial_pop_risk:.6f}")
    
    # Set up optimizers
    print("\n2. Setting up optimizers...")
    
    # ADANA optimizer
    G2_SCALE = 0.1
    G3_IV = 0.05
    g2 = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*ALPHA), 1)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    adana_opt = optimizers.adana_optimizer(g2=g2, g3=g3, Delta=Delta)
    
    # SGD optimizer
    SGD_LR = 0.01
    sgd_opt = optax.sgd(learning_rate=SGD_LR)
    
    print(f"  ADANA: g2={G2_SCALE}, g3_iv={G3_IV}")
    print(f"  SGD: lr={SGD_LR}")
    
    # Run ADANA
    print("\n3. Running ADANA optimizer...")
    key, subkey = random.split(key)
    adana_losses, adana_risks, adana_final_params = run_optimizer_steps(
        problem, adana_opt, init_params, subkey, NUM_STEPS, BATCH_SIZE)
    
    # Run SGD
    print("4. Running SGD optimizer...")
    key, subkey = random.split(key)  
    sgd_losses, sgd_risks, sgd_final_params = run_optimizer_steps(
        problem, sgd_opt, init_params, subkey, NUM_STEPS, BATCH_SIZE)
    
    # Compare results
    print("\n5. Comparing results...")
    print(f"  ADANA final population risk: {adana_risks[-1]:.6f}")
    print(f"  SGD final population risk: {sgd_risks[-1]:.6f}")
    print(f"  ADANA improvement: {initial_pop_risk - adana_risks[-1]:.6f}")
    print(f"  SGD improvement: {initial_pop_risk - sgd_risks[-1]:.6f}")
    
    # Parameter change analysis
    adana_param_change = jnp.linalg.norm(adana_final_params - init_params)
    sgd_param_change = jnp.linalg.norm(sgd_final_params - init_params)
    print(f"  ADANA parameter change norm: {adana_param_change:.6f}")
    print(f"  SGD parameter change norm: {sgd_param_change:.6f}")
    
    # Print step-by-step comparison
    print("\n6. Step-by-step comparison:")
    print("  Step | ADANA Pop Risk | SGD Pop Risk | ADANA Better?")
    print("  -----|----------------|--------------|---------------")
    for i in range(NUM_STEPS):
        adana_better = "✓" if adana_risks[i] < sgd_risks[i] else "✗"
        print(f"  {i+1:4d} |     {adana_risks[i]:8.6f} |   {sgd_risks[i]:8.6f} |       {adana_better}")
    
    # Create visualization
    print("\n7. Creating visualization...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    steps = range(1, NUM_STEPS + 1)
    
    # Plot 1: Population Risk
    ax1.plot(steps, adana_risks, 'r-o', label='ADANA', markersize=4)
    ax1.plot(steps, sgd_risks, 'b-s', label='SGD', markersize=4)
    ax1.set_xlabel('Optimization Step')
    ax1.set_ylabel('Population Risk')
    ax1.set_title('Population Risk Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Batch Loss
    ax2.plot(steps, adana_losses, 'r-o', label='ADANA', markersize=4)
    ax2.plot(steps, sgd_losses, 'b-s', label='SGD', markersize=4) 
    ax2.set_xlabel('Optimization Step')
    ax2.set_ylabel('Batch Loss')
    ax2.set_title('Batch Loss Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot if results directory exists
    if os.path.exists('results'):
        plt.savefig('results/mpowerlaw_adana_sgd_comparison.pdf', bbox_inches='tight')
        print(f"  Plot saved to: results/mpowerlaw_adana_sgd_comparison.pdf")
    else:
        print("  Results directory not found, showing plot instead...")
        plt.show()
    
    print("\n" + "=" * 70)
    print("Comparison completed!")
    
    # Summary statistics
    adana_wins = sum(1 for i in range(NUM_STEPS) if adana_risks[i] < sgd_risks[i])
    print(f"Summary: ADANA achieved lower population risk in {adana_wins}/{NUM_STEPS} steps")
    
    if adana_risks[-1] < sgd_risks[-1]:
        improvement = (sgd_risks[-1] - adana_risks[-1]) / sgd_risks[-1] * 100
        print(f"ADANA final risk is {improvement:.1f}% better than SGD")
    else:
        improvement = (adana_risks[-1] - sgd_risks[-1]) / adana_risks[-1] * 100
        print(f"SGD final risk is {improvement:.1f}% better than ADANA")
    
    print("=" * 70)

if __name__ == "__main__":
    main() 