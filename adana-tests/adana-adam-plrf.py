import jax
import jax.numpy as jnp
import jax.random as random
import optax
import scipy as sp

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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

key = random.PRNGKey(0)

ALPHA = 1.0
BETALIST = [0.3, 0.7, 1.0, 1.3]
V = 2000
D = 500
SGDBATCH = 1
STEPS = 10**4
FIXED_LR = 0.001

# Generate data for each beta value
results_by_beta = []
for beta in BETALIST:
    key, newkey = random.split(key)
    # Create problem instance for this beta
    problem = PowerLawRF.initialize_random(ALPHA, beta, V, D, key=newkey)
    
    # Run Adam with fixed learning rate
    key, newkey = random.split(key)
    optimizer = optax.adam(learning_rate=FIXED_LR)
    adamtimes, adamlosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        optimizer,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    # Run ADANA optimizer
    key, newkey = random.split(key)
    # Set up ADANA parameters
    g2 = optimizers.powerlaw_schedule(FIXED_LR*10.0, 0.0, 0.0, 1)  # gamma2
    g3 = optimizers.powerlaw_schedule(0.0, 0.0, 0.0, 1)      # turned off momentum term
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2, g3=g3, Delta=Delta)
    adanatimes, adanalosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        adana_opt,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    results_by_beta.append({
        'beta': beta,
        'adam': (adamtimes, adamlosses),
        'adana': (adanatimes, adanalosses)
    })

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, result in enumerate(results_by_beta):
    beta = result['beta']
    adamtimes, adamlosses = result['adam']
    adanatimes, adanalosses = result['adana']
    
    # Plot on the corresponding subplot
    axes[idx].loglog(adamtimes+1, adamlosses, label=f'Adam (lr={FIXED_LR})', color='blue')
    axes[idx].loglog(adanatimes+1, adanalosses, label='ADANA', color='red')
    
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel('Population Risk')
    axes[idx].set_title(f'β = {beta}')
    axes[idx].grid(True)
    axes[idx].legend()

plt.tight_layout()
plt.savefig('results/adana_adam_comparison.pdf', bbox_inches='tight')
plt.show()
