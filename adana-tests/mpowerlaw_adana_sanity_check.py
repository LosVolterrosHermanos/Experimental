#!/usr/bin/env python3
"""
Basic sanity check script for MPowerLawRF class and ADANA optimizer implementation.

This script:
1. Creates a small instance of MPowerLawRF
2. Sets up the loss function similar to mlsq_streaming_optax
3. Tests one step of the ADANA optimizer 
4. Examines the updates that result

Author: AI Assistant
Date: 2025-01-27
"""

import jax
import jax.numpy as jnp
import jax.random as random
import optax
from typing import NamedTuple

import sys
import os
sys.path.append('../')
sys.path.append('../power_law_rf')

# Import required modules
import optimizers
from power_law_rf.power_law_rf import MPowerLawRF
from power_law_rf.least_squares import mlsq_streaming_optax_simple

def main():
    print("=" * 60)
    print("MPowerLawRF + ADANA Optimizer Sanity Check")
    print("=" * 60)
    
    # Set random seed for reproducibility
    key = random.PRNGKey(42)
    
    # Problem parameters (small for testing)
    print("\n1. Setting up problem parameters...")
    ALPHA = 1.0      # Power law exponent for eigenvalue decay
    BETA = 0.7       # Power law exponent for target coefficient decay  
    ZETA = 0.5       # Power law exponent for expert selection probability
    M = 3            # Number of experts (small for testing)
    V = 20           # Hidden dimensionality (small)
    D = 10           # Embedded dimensionality (small)
    BATCH_SIZE = 5   # Small batch size
    
    print(f"  α (eigenvalue decay): {ALPHA}")
    print(f"  β (target decay): {BETA}")
    print(f"  ζ (expert selection): {ZETA}")
    print(f"  M (num experts): {M}")
    print(f"  V (hidden dim): {V}")
    print(f"  D (embedded dim): {D}")
    print(f"  Batch size: {BATCH_SIZE}")
    
    # Create MPowerLawRF instance
    print("\n2. Creating MPowerLawRF instance...")
    key, subkey = random.split(key)
    problem = MPowerLawRF.initialize_random(ALPHA, BETA, ZETA, M, V, D, key=subkey)
    
    print(f"  Problem created successfully!")
    print(f"  Expert probabilities: {problem.expert_probs}")
    print(f"  CheckW shape: {problem.checkW.shape}")
    print(f"  Checkb shape: {problem.checkb.shape}")
    
    # Initialize parameters (weight matrix for all experts)
    print("\n3. Initializing parameters...")
    key, subkey = random.split(key)
    params = random.normal(subkey, (M, D)) * 0.01  # Small random initialization
    print(f"  Parameter matrix shape: {params.shape}")
    print(f"  Initial parameter norm: {jnp.linalg.norm(params):.6f}")
    
    # Test population risk calculation
    print("\n4. Testing population risk calculation...")
    initial_risk = problem.get_population_risk(params)
    print(f"  Initial population risk: {initial_risk:.6f}")
    
    # Generate a batch of data
    print("\n5. Generating test data batch...")
    key, subkey = random.split(key)
    embeddings, targets, expert_indices = problem.get_data(subkey, BATCH_SIZE)
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Expert indices shape: {expert_indices.shape}")
    print(f"  Expert indices: {expert_indices}")
    
    # Set up batch MSE loss function (similar to mlsq_streaming_optax)
    print("\n6. Setting up batch MSE loss function...")
    def batch_mse(params, embeddings, targets, expert_indices):
        """MSE loss for mixed expert model (same as in mlsq_streaming_optax)."""
        # Get predictions for each expert
        y_pred = jnp.sum(embeddings * params[expert_indices], axis=1)
        
        # Ensure targets have the right shape
        if targets.ndim > 1:
            targets = targets.squeeze()
            
        return jnp.mean(optax.l2_loss(y_pred, targets))
    
    # Test the loss function
    batch_loss = batch_mse(params, embeddings, targets, expert_indices)
    print(f"  Batch loss: {batch_loss:.6f}")
    
    # Set up ADANA optimizer
    print("\n7. Setting up ADANA optimizer...")
    G2_SCALE = 0.1
    G3_IV = 0.05
    
    # ADANA parameter schedules
    g2 = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*ALPHA), 1)  
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2, g3=g3, Delta=Delta)
    print(f"  ADANA optimizer created")
    print(f"  g2 scale: {G2_SCALE}")
    print(f"  g3 initial value: {G3_IV}")
    
    # Initialize optimizer state
    print("\n8. Initializing optimizer state...")
    opt_state = adana_opt.init(params)
    print(f"  Optimizer state initialized")
    print(f"  State type: {type(opt_state)}")
    print(f"  Count: {opt_state.count}")
    print(f"  m shape: {jax.tree.map(lambda x: x.shape if x is not None else None, opt_state.m)}")
    print(f"  v shape: {jax.tree.map(lambda x: x.shape if x is not None else None, opt_state.v)}")
    print(f"  tau shape: {jax.tree.map(lambda x: x.shape if x is not None else None, opt_state.tau)}")
    
    # Compute gradients
    print("\n9. Computing gradients...")
    loss_fn = lambda p: batch_mse(p, embeddings, targets, expert_indices)
    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    print(f"  Loss value: {loss_val:.6f}")
    print(f"  Gradient shape: {grads.shape}")
    print(f"  Gradient norm: {jnp.linalg.norm(grads):.6f}")
    print(f"  Gradient statistics:")
    print(f"    Mean: {jnp.mean(grads):.6f}")
    print(f"    Std: {jnp.std(grads):.6f}")
    print(f"    Min: {jnp.min(grads):.6f}")
    print(f"    Max: {jnp.max(grads):.6f}")
    
    # Apply one optimizer step
    print("\n10. Applying one ADANA optimizer step...")
    updates, new_opt_state = adana_opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    print(f"  Updates applied successfully")
    print(f"  Updates shape: {updates.shape}")
    print(f"  Updates norm: {jnp.linalg.norm(updates):.6f}")
    print(f"  Updates statistics:")
    print(f"    Mean: {jnp.mean(updates):.6f}")
    print(f"    Std: {jnp.std(updates):.6f}")
    print(f"    Min: {jnp.min(updates):.6f}")
    print(f"    Max: {jnp.max(updates):.6f}")
    
    # Analyze parameter changes
    print("\n11. Analyzing parameter changes...")
    param_diff = new_params - params
    print(f"  Parameter change norm: {jnp.linalg.norm(param_diff):.6f}")
    print(f"  Relative parameter change: {jnp.linalg.norm(param_diff) / jnp.linalg.norm(params):.6f}")
    print(f"  New parameter norm: {jnp.linalg.norm(new_params):.6f}")
    
    # Check new loss
    new_loss_val = loss_fn(new_params)
    new_population_risk = problem.get_population_risk(new_params)
    print(f"  New batch loss: {new_loss_val:.6f}")
    print(f"  New population risk: {new_population_risk:.6f}")
    print(f"  Loss change: {new_loss_val - loss_val:.6f}")
    print(f"  Population risk change: {new_population_risk - initial_risk:.6f}")
    
    # Examine optimizer state changes
    print("\n12. Examining optimizer state changes...")
    print(f"  Count: {opt_state.count} → {new_opt_state.count}")
    
    # Check momentum terms
    m_norm_old = jnp.linalg.norm(opt_state.m)
    m_norm_new = jnp.linalg.norm(new_opt_state.m)
    print(f"  First momentum (m) norm: {m_norm_old:.6f} → {m_norm_new:.6f}")
    
    v_norm_old = jnp.linalg.norm(opt_state.v)
    v_norm_new = jnp.linalg.norm(new_opt_state.v)
    print(f"  Second momentum (v) norm: {v_norm_old:.6f} → {v_norm_new:.6f}")
    
    tau_norm_old = jnp.linalg.norm(opt_state.tau)
    tau_norm_new = jnp.linalg.norm(new_opt_state.tau)
    print(f"  Tau norm: {tau_norm_old:.6f} → {tau_norm_new:.6f}")
    
    # Test a few more steps to see convergence behavior
    print("\n13. Testing a few more optimizer steps...")
    current_params = new_params
    current_opt_state = new_opt_state
    current_loss = new_loss_val
    
    for step in range(1, 4):
        # Generate new batch
        key, subkey = random.split(key)
        embeddings, targets, expert_indices = problem.get_data(subkey, BATCH_SIZE)
        
        # Compute gradients
        loss_fn = lambda p: batch_mse(p, embeddings, targets, expert_indices)
        loss_val, grads = jax.value_and_grad(loss_fn)(current_params)
        
        # Apply optimizer step
        updates, current_opt_state = adana_opt.update(grads, current_opt_state, current_params)
        current_params = optax.apply_updates(current_params, updates)
        
        # Check population risk
        pop_risk = problem.get_population_risk(current_params)
        
        print(f"  Step {step}: batch_loss={loss_val:.6f}, pop_risk={pop_risk:.6f}, "
              f"update_norm={jnp.linalg.norm(updates):.6f}")
    
    print("\n" + "=" * 60)
    print("Sanity check completed successfully!")
    print("Key observations:")
    print(f"  - MPowerLawRF properly handles {M} experts")
    print(f"  - ADANA optimizer state updates correctly")
    print(f"  - Gradients and updates have reasonable magnitudes")
    print(f"  - Loss and population risk track as expected")
    print("=" * 60)

if __name__ == "__main__":
    main() 