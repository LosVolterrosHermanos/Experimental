import jax
import jax.numpy as jnp
import jax.random as random
import optax
import scipy as sp

from tqdm import tqdm
import matplotlib.pyplot as plt

from typing import NamedTuple, Callable

import sys
import os

sys.path.append('../')
import optimizers

# Import individual modules to avoid relative import issues
from power_law_rf.least_squares import mlsq_streaming_optax_simple
#from power_law_rf.deterministic_equivalent import deterministic_equivalent as theory
from power_law_rf.power_law_rf import MPowerLawRF

import matplotlib.cm as cm
import numpy as np

key = random.PRNGKey(0)

# Fixed hyperparameters
ALPHA = 1.0
BETA = 0.3 # Decay
ZETA = 0.5 # Expert selection power law exponent
D_VALUES = [32, 64, 128, 256]  # Smaller set for debugging
SGDBATCH = 32
STEPS = 10**3  # Reduced for debugging
DECAY_STEPS = STEPS  # Decay over the entire training

G2_SCALE = 0.25
G3_IV = 0.05

# Generate data for each (D, V) pair
results_by_dimension = []

for D in tqdm(D_VALUES, desc="Processing dimensions"):
    V = 4 * D  # V = 4*D as specified
    M = D
    INITIAL_LR = 1.0/jnp.sqrt(D)
    
    key, newkey = random.split(key)
    # Create problem instance for this (D, V) pair
    problem = MPowerLawRF.initialize_random(ALPHA, BETA, ZETA, M, V, D, key=newkey)
    
    # Run ADANA optimizer with cosine decay
    key, newkey = random.split(key)
    #(2α+2β−1)/(2α) 
    #MAGIC = ((2*problem.alpha+2*problem.beta-1)/(2*problem.alpha))*(2.0-1.0/(2*problem.alpha))
    #adam_powerlaw_schedule = optimizers.powerlaw_schedule(INITIAL_LR, 0.0, -MAGIC/2.0, 1)
    g2_adana = optimizers.powerlaw_schedule(INITIAL_LR*G2_SCALE, 0.0, 0.0, 1)
    g3_adana = optimizers.powerlaw_schedule(INITIAL_LR*G3_IV, 0.0, -1.0/(2*problem.alpha), 1)
    Delta_adana = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2_adana, g3=g3_adana, Delta=Delta_adana)
    # scaled_adana_opt = optax.chain(
    #     adana_opt,
    #     #optax.scale_by_schedule(adam_powerlaw_schedule)
    #     #optax.scale(INITIAL_LR)
    #     #optax.scale_by_schedule(optax.cosine_decay_schedule(INITIAL_LR, DECAY_STEPS))
    # )
    adanatimes, adanalosses = mlsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        adana_opt,
        jnp.zeros((problem.m, problem.d)),  # Changed to (m, d) shape
        problem.get_population_risk
    )
    
    # Run regular DANA optimizer
    key, newkey = random.split(key)
    g1_dana = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2_dana = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3_dana = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*problem.alpha), 1)
    Delta_dana = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    dana_opt = optimizers.dana_optimizer(g1=g1_dana, g2=g2_dana, g3=g3_dana, Delta=Delta_dana)
    danatimes, danalosses = mlsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        dana_opt,
        jnp.zeros((problem.m, problem.d)),  # Changed to (m, d) shape
        problem.get_population_risk
    )
    
    # Run simple SGD with step size G2_SCALE
    key, newkey = random.split(key)
    sgd_opt = optax.sgd(learning_rate=G2_SCALE)
    sgdtimes, sgdlosses = mlsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        sgd_opt,
        jnp.zeros((problem.m, problem.d)),  # Changed to (m, d) shape
        problem.get_population_risk
    )
    
    results_by_dimension.append({
        'D': D,
        'V': V,
        'adana': (adanatimes, adanalosses),
        'dana': (danatimes, danalosses),
        'sgd': (sgdtimes, sgdlosses),
        'empirical_limit': problem.get_empirical_limit_loss()
    })
    
    print(f"Completed D={D}, V={V}, M={M}, Empirical limit: {problem.get_empirical_limit_loss():.6f}")
    del problem

# Create plots - single plot with all curves overlaid

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for idx, result in enumerate(results_by_dimension):
    D = result['D']
    V = result['V']
    adanatimes, adanalosses = result['adana']
    danatimes, danalosses = result['dana']
    sgdtimes, sgdlosses = result['sgd']
    
    # Calculate FLOPS (D * B * iterations for mixed model)
    adana_flops = D * SGDBATCH * (adanatimes + 1)
    dana_flops = D * SGDBATCH * (danatimes + 1)  
    sgd_flops = D * SGDBATCH * (sgdtimes + 1)
    
    # Use consistent colors for each algorithm, varying transparency/line width by dimension
    alpha = 0.6 + 0.4 * (idx / (len(D_VALUES) - 1))  # Alpha varies from 0.6 to 1.0
    
    # Plot with consistent colors for each optimizer
    ax.loglog(adana_flops, adanalosses, '-', color='red', alpha=alpha, 
              label='ADANA (cosine decay LR schedule)' if idx == 0 else None)
    ax.loglog(dana_flops, danalosses, '--', color='green', alpha=alpha,
              label='DANA (regular)' if idx == 0 else None)  
    ax.loglog(sgd_flops, sgdlosses, ':', color='blue', alpha=alpha,
              label='SGD' if idx == 0 else None)

ax.set_xlabel('FLOPS (D × B × Iterations)')
ax.set_ylabel('Population Risk')
ax.set_title(f'Mixed Experts: ADANA vs DANA vs SGD (β={BETA}, ζ={ZETA}, M={M})')
ax.grid(True)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='ADANA (cosine decay LR schedule)'),
    Line2D([0], [0], color='green', linestyle='--', label='DANA (regular)'),
    Line2D([0], [0], color='blue', linestyle=':', label='SGD')
]

# Add text annotation for dimension range and expert info
ax.text(0.02, 0.98, f'D ∈ [{min(D_VALUES)}, {max(D_VALUES)}]\nM = {M}, ζ = {ZETA}', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig('results/madana_dana_sgd_varying_dimensions.pdf', bbox_inches='tight')
plt.show()

# Also create a summary plot showing final losses vs dimension
final_adana_losses = []
final_dana_losses = []
final_sgd_losses = []
empirical_limits = []
dimensions = []

for result in results_by_dimension:
    D = result['D']
    adanatimes, adanalosses = result['adana']
    danatimes, danalosses = result['dana']
    sgdtimes, sgdlosses = result['sgd']
    
    final_adana_losses.append(adanalosses[-1])
    final_dana_losses.append(danalosses[-1])
    final_sgd_losses.append(sgdlosses[-1])
    empirical_limits.append(result['empirical_limit'])
    dimensions.append(D)

plt.figure(figsize=(10, 6))
plt.loglog(dimensions, final_adana_losses, 'o-', label='ADANA (optimized power LR schedule)', color='red')
plt.loglog(dimensions, final_dana_losses, 's-', label='DANA (regular)', color='green')
plt.loglog(dimensions, final_sgd_losses, '^-', label='SGD', color='blue')
plt.loglog(dimensions, empirical_limits, 'x-', label='Empirical Limit', color='black', linestyle='--')
plt.xlabel('Dimension D')
plt.ylabel('Final Population Risk')
plt.title(f'Mixed Experts: Final Risk vs Dimension (β={BETA}, ζ={ZETA}, M={M})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('results/mixed_final_risk_vs_dimension_with_sgd.pdf', bbox_inches='tight')
plt.show()

# Print summary statistics
print("\nSummary:")
print("="*50)
for i, result in enumerate(results_by_dimension):
    D = result['D']
    _, adanalosses = result['adana']
    _, danalosses = result['dana']
    _, sgdlosses = result['sgd']
    emp_limit = result['empirical_limit']
    
    print(f"D={D:4d}: ADANA={adanalosses[-1]:.2e}, DANA={danalosses[-1]:.2e}, "
          f"SGD={sgdlosses[-1]:.2e}, Limit={emp_limit:.2e}") 