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

key = random.PRNGKey(0)

ALPHA = 1.0
BETALIST = [0.3, 0.7, 1.0, 1.3]
V = 2000
D = 500
SGDBATCH = 1
STEPS = 10**4
INITIAL_LR = 0.01
DECAY_STEPS = STEPS  # Decay over the entire training

# Generate data for each beta value
results_by_beta = []
for beta in BETALIST:
    key, newkey = random.split(key)
    # Create problem instance for this beta
    problem = PowerLawRF.initialize_random(ALPHA, beta, V, D, key=newkey)
    
    # Run Adam with cosine decay
    key, newkey = random.split(key)
    adam_cosine_schedule = optax.cosine_decay_schedule(
        init_value=INITIAL_LR, 
        decay_steps=DECAY_STEPS
    )
    adam_optimizer = optax.adam(learning_rate=adam_cosine_schedule)
    adamtimes, adamlosses = lsq_streaming_optax_simple(
        newkey,
        problem.get_data,
        SGDBATCH,
        STEPS,
        adam_optimizer,
        jnp.zeros((problem.d,1)),
        problem.get_population_risk
    )
    
    # Run ADANA optimizer with cosine decay
    key, newkey = random.split(key)
    # Set up ADANA parameters with cosine decay for g2
    g2 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(0.0, 0.0, 0.0, 1)      # turned off momentum term
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2, g3=g3, Delta=Delta)
    scaled_adana_opt = optax.chain(
        adana_opt,
        optax.scale_by_schedule(adam_cosine_schedule)
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
    axes[idx].loglog(adamtimes+1, adamlosses, label=f'Adam (cosine decay)', color='blue')
    axes[idx].loglog(adanatimes+1, adanalosses, label='ADANA (cosine decay)', color='red')
    
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel('Population Risk')
    axes[idx].set_title(f'β = {beta}')
    axes[idx].grid(True)
    axes[idx].legend()

plt.tight_layout()
plt.savefig('results/adana_adam_cosine_decay_comparison.pdf', bbox_inches='tight')
plt.show() 