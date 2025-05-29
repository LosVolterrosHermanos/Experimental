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
sys.path.append('../power_law_rf')
import optimizers
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.ode import ode_resolvent_log_implicit
from power_law_rf.ode import DanaHparams
from power_law_rf.ode import ODEInputs
from power_law_rf.least_squares import lsq_streaming_optax_simple
import power_law_rf.deterministic_equivalent as theory

import matplotlib.cm as cm
import numpy as np

key = random.PRNGKey(0)

# Fixed hyperparameters
ALPHA = 1.0
BETA = 0.7 # Decay
D_VALUES = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SGDBATCH = 32
STEPS = 10**5
DECAY_STEPS = STEPS  # Decay over the entire training

G2_SCALE = 0.25
G3_IV = 0.05

# Generate data for each (D, V) pair
results_by_dimension = []

for D in tqdm(D_VALUES, desc="Processing dimensions"):
    V = 4 * D  # V = 4*D as specified
    INITIAL_LR = 1.0/jnp.sqrt(D)
    
    key, newkey = random.split(key)
    # Create problem instance for this (D, V) pair
    problem = PowerLawRF.initialize_random(ALPHA, BETA, V, D, key=newkey)
    
    # Run ADANA optimizer with cosine decay
    key, newkey = random.split(key)
    # adam_cosine_schedule = optax.cosine_decay_schedule(
    #     init_value=INITIAL_LR, 
    #     decay_steps=DECAY_STEPS
    # )
    #(2α+2β−1)/(2α) 
    MAGIC = ((2*problem.alpha+2*problem.beta-1)/(2*problem.alpha))*(2.0-1.0/(2*problem.alpha))
    adam_powerlaw_schedule = optimizers.powerlaw_schedule(INITIAL_LR, 0.0, -MAGIC/2.0, 1)
    g2_adana = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3_adana = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*problem.alpha), 1)
    Delta_adana = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2_adana, g3=g3_adana, Delta=Delta_adana)
    scaled_adana_opt = optax.chain(
        adana_opt,
        optax.scale_by_schedule(adam_powerlaw_schedule)
    )
    adanatimes, adanalosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        scaled_adana_opt,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    # Run regular DANA optimizer
    key, newkey = random.split(key)
    g1_dana = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2_dana = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3_dana = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*problem.alpha), 1)
    Delta_dana = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    dana_opt = optimizers.dana_optimizer(g1=g1_dana, g2=g2_dana, g3=g3_dana, Delta=Delta_dana)
    danatimes, danalosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        dana_opt,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    # Run simple SGD with step size G2_SCALE
    key, newkey = random.split(key)
    sgd_opt = optax.sgd(learning_rate=G2_SCALE)
    sgdtimes, sgdlosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        sgd_opt,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    results_by_dimension.append({
        'D': D,
        'V': V,
        'adana': (adanatimes, adanalosses),
        'dana': (danatimes, danalosses),
        'sgd': (sgdtimes, sgdlosses)
    })
    
    print(f"Completed D={D}, V={V}")
    del problem

# Create plots - single plot with all curves overlaid

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for idx, result in enumerate(results_by_dimension):
    D = result['D']
    V = result['V']
    adanatimes, adanalosses = result['adana']
    danatimes, danalosses = result['dana']
    sgdtimes, sgdlosses = result['sgd']
    
    # Calculate FLOPS (D * iterations)
    adana_flops = D * (adanatimes + 1)
    dana_flops = D * (danatimes + 1)  
    sgd_flops = D * (sgdtimes + 1)
    
    # Use consistent colors for each algorithm, varying transparency/line width by dimension
    alpha = 0.6 + 0.4 * (idx / (len(D_VALUES) - 1))  # Alpha varies from 0.6 to 1.0
    
    # Plot with consistent colors for each optimizer
    ax.loglog(adana_flops, adanalosses, '-', color='red', alpha=alpha, 
              label='ADANA (optimized power LR schedule)' if idx == 0 else None)
    ax.loglog(dana_flops, danalosses, '--', color='green', alpha=alpha,
              label='DANA (regular)' if idx == 0 else None)  
    ax.loglog(sgd_flops, sgdlosses, ':', color='blue', alpha=alpha,
              label='SGD' if idx == 0 else None)

ax.set_xlabel('FLOPS (D × Iterations)')
ax.set_ylabel('Population Risk')
ax.set_title(f'ADANA vs DANA vs SGD Convergence: β={BETA}, Varying Dimensions')
ax.grid(True)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='red', linestyle='-', label='ADANA (optimized power LR schedule)'),
    Line2D([0], [0], color='green', linestyle='--', label='DANA (regular)'),
    Line2D([0], [0], color='blue', linestyle=':', label='SGD')
]

# Add text annotation for dimension range
ax.text(0.02, 0.98, f'D ∈ [{min(D_VALUES)}, {max(D_VALUES)}]', 
        transform=ax.transAxes, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.legend(handles=legend_elements, loc='upper right')
plt.tight_layout()
plt.savefig('results/adana_dana_sgd_varying_dimensions.pdf', bbox_inches='tight')
plt.show()

# Also create a summary plot showing final losses vs dimension
final_adana_losses = []
final_dana_losses = []
final_sgd_losses = []
dimensions = []

for result in results_by_dimension:
    D = result['D']
    adanatimes, adanalosses = result['adana']
    danatimes, danalosses = result['dana']
    sgdtimes, sgdlosses = result['sgd']
    
    final_adana_losses.append(adanalosses[-1])
    final_dana_losses.append(danalosses[-1])
    final_sgd_losses.append(sgdlosses[-1])
    dimensions.append(D)

plt.figure(figsize=(10, 6))
plt.loglog(dimensions, final_adana_losses, 'o-', label='ADANA (optimized power LR schedule)', color='red')
plt.loglog(dimensions, final_dana_losses, 's-', label='DANA (regular)', color='green')
plt.loglog(dimensions, final_sgd_losses, '^-', label='SGD', color='blue')
plt.xlabel('Dimension D')
plt.ylabel('Final Population Risk')
plt.title(f'Final Risk vs Dimension (β={BETA})')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('results/final_risk_vs_dimension_with_sgd.pdf', bbox_inches='tight')
plt.show() 