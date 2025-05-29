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
INITIAL_LR = 1.0/jnp.sqrt(D)
DECAY_STEPS = STEPS  # Decay over the entire training

G2_SCALE = 0.5
G3_IV = 0.1

# Generate data for each beta value
results_by_beta = []
for beta in BETALIST:
    key, newkey = random.split(key)
    # Create problem instance for this beta
    problem = PowerLawRF.initialize_random(ALPHA, beta, V, D, key=newkey)
    
    # Run ADANA optimizer with cosine decay (same as adana-adam-cosine-decay.py)
    key, newkey = random.split(key)
    adam_cosine_schedule = optax.cosine_decay_schedule(
        init_value=INITIAL_LR, 
        decay_steps=DECAY_STEPS
    )
    g2_adana = optimizers.powerlaw_schedule(G2_SCALE, 0.0, 0.0, 1)
    g3_adana = optimizers.powerlaw_schedule(G3_IV, 0.0, -1.0/(2*problem.alpha), 1)
    Delta_adana = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 6.0)
    
    adana_opt = optimizers.adana_optimizer(g2=g2_adana, g3=g3_adana, Delta=Delta_adana)
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
    
    # Run regular DANA optimizer (following det-equivalent-low-alpha-test.py pattern)
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
    
    results_by_beta.append({
        'beta': beta,
        'adana': (adanatimes, adanalosses),
        'dana': (danatimes, danalosses)
    })

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, result in enumerate(results_by_beta):
    beta = result['beta']
    adanatimes, adanalosses = result['adana']
    danatimes, danalosses = result['dana']
    
    # Plot on the corresponding subplot
    axes[idx].loglog(adanatimes+1, adanalosses, label='ADANA (cosine decay)', color='red')
    axes[idx].loglog(danatimes+1, danalosses, label='DANA (regular)', color='green')
    
    axes[idx].set_xlabel('Iteration')
    axes[idx].set_ylabel('Population Risk')
    axes[idx].set_title(f'β = {beta}')
    axes[idx].grid(True)
    axes[idx].legend()

plt.tight_layout()
plt.savefig('results/adana_dana_comparison.pdf', bbox_inches='tight')
plt.show() 