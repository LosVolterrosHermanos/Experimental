# MoE PLRF: Stochastic vs ODE Comparison
"""
This notebook compares stochastic training of Mixture of Experts PLRF models
with theoretical ODE predictions.
"""

#@title Setup and Imports
import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from typing import NamedTuple, Callable, Dict, List, Union
import numpy as np

from power_law_rf.optimizers import powerlaw_schedule, dana_optimizer, tanea_optimizer

# Import MoE PLRF implementations from installed package
from power_law_rf.moe_plrf.moe_plrf import (
    TwoExpertPLRF,
    MoEPLRFTrainer,
    MixtureOfExpertsPLRF
)
from power_law_rf.moe_plrf.moe_plrf_ode import (
    ode_moe_dana_log_implicit,
    MoEODEInputs
)
from power_law_rf.ode import DanaHparams

class TaneaHparams(NamedTuple):
    g2: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g3: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    delta: Callable[[Union[float, jnp.ndarray]], float]  # momentum function

import power_law_rf.deterministic_equivalent as theory

# Set random seed
key = random.PRNGKey(42)

#@title Global Parameters
# Model parameters
ALPHA = 1.0
BETA_LIST = [0.5, 0.8]#, 1.1, 1.4]
# BETA_LIST = [0.8]
# BETA = 0.8
V = 2000  # Hidden dimension
D = 500   # Parameter dimension

# MoE parameters
M = 100  # Number of experts for general MoE
ZETA = 0.9  # Power-law exponent for expert selection

# Training parameters
# BATCH_SIZE = 32
STEPS = 10000
DT = 1e-3  # ODE time step

# Learning rate schedules
BASE_LR = 0.01
MOMENTUM = 0.9

G2_SCALE = 0.2
G3_OVER_G2 = 0.1
BATCH_SIZE = 1
TANEA_LR_SCALAR = 1E-3
TANEA_GLOBAL_EXPONENT = 0.0


def get_traceK(alpha, v):
  x_grid = jnp.arange(1, v+1).reshape(1, v)
  population_eigs = x_grid ** -alpha
  population_trace = jnp.sum(population_eigs**2)
  return population_trace

# def get_sgd_hparams(g2_scale, batch_size, traceK):
#   learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
#   dana_params_sgd = DanaHparams(
#     g1=lambda t: 0.0,  # No momentum accumulation
#     g2=lambda t: learning_rate,  # Direct gradient update
#     g3=lambda t: 0.0,  # No momentum contribution
#     delta=lambda t: 1.0  # Full momentum decay
#   )
#   return dana_params_sgd

def get_tanea_hparams(alpha, beta, d, batch_size, g2_scale, g3_over_g2, traceK, tanea_lr_scalar, tanea_global_exponent):
  kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
  learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
  tanea_params = TaneaHparams(
    g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
    g3=powerlaw_schedule(tanea_lr_scalar*learning_rate*g3_over_g2, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
    delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
  )
  return tanea_params

def get_tarmsprop_sgd_hparams(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
  kappa_b = jnp.log(batch_size) / jnp.log(d)  # exponent for batch wrt d
  learning_rate = g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)
  tanea_params = TaneaHparams(
    g2=powerlaw_schedule(tanea_lr_scalar*learning_rate, 0.0, -tanea_global_exponent, 1.0),
    g3=powerlaw_schedule(0.0, 0.0, -tanea_global_exponent-(1.0 - kappa_b) / (2 * alpha), 1.0),
    delta=powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(alpha+beta)/(2*alpha))
  )
  return tanea_params

def get_adam_lr(alpha, beta, d, batch_size, g2_scale, traceK, tanea_lr_scalar, tanea_global_exponent):
   return g2_scale * jnp.minimum(1.0, jnp.float32(batch_size) / traceK)*tanea_lr_scalar

traceK = get_traceK(ALPHA, V)

#@title Part 4: General MoE with Power-Law Expert Selection
"""
Test the general MoE model with power-law expert selection using SGD and DANA-decaying.
"""

print("\n" + "="*60)
print(f"General MoE PLRF: {M} Experts with Power-Law Selection")
print("="*60)

moe_results = []

for beta in BETA_LIST:
    print(f"\nRunning MoE experiments for β = {beta}")

    # Create MoE model
    key, model_key = random.split(key)
    model = MixtureOfExpertsPLRF(
        alpha=ALPHA,
        beta=beta,
        v=V,
        d=D,
        m=M,
        zeta=ZETA,
        key=model_key
    )

    print(f"  Expert probabilities: {model.expert_probs}")
    print(f"  Optimal risk: {model.population_risk(model.optimal_params_per_expert()):.6f}")

    # Create hyperparameters using the same helper functions
    # sgd_hparams = get_sgd_hparams(G2_SCALE, BATCH_SIZE, traceK)
    # sgd_opt = dana_optimizer(sgd_hparams.g1, sgd_hparams.g2, sgd_hparams.g3, sgd_hparams.delta)
    # dana_decaying_hparams = get_dana_decaying_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, G3_IV, traceK)
    # dana_decaying_opt = dana_optimizer(dana_decaying_hparams.g1, dana_decaying_hparams.g2, dana_decaying_hparams.g3, dana_decaying_hparams.delta)
    tanea_hparams = get_tanea_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, G3_OVER_G2, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tanea_opt = tanea_optimizer(tanea_hparams.g2, tanea_hparams.g3, tanea_hparams.delta)
    tarmsprop_sgd_hparams = get_tarmsprop_sgd_hparams(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT)
    tarmsprop_sgd_opt = tanea_optimizer(tarmsprop_sgd_hparams.g2, tarmsprop_sgd_hparams.g3, tarmsprop_sgd_hparams.delta)

    adam_opt = optax.adam(get_adam_lr(ALPHA, beta, D, BATCH_SIZE, G2_SCALE, traceK, TANEA_LR_SCALAR, TANEA_GLOBAL_EXPONENT),b1=0.0)

    # # SGD experiment
    # sgd_trainer = MoEPLRFTrainer(model, sgd_opt)
    # key, train_key = random.split(key)
    # sgd_results = sgd_trainer.train(
    #     train_key,
    #     num_steps=STEPS,
    #     batch_size=BATCH_SIZE,
    #     track_per_expert_loss=True
    # )

    # # DANA-decaying experiment
    # dana_decaying_trainer = MoEPLRFTrainer(model, dana_decaying_opt)
    # key, train_key = random.split(key)
    # dana_decaying_results = dana_decaying_trainer.train(
    #     train_key,
    #     num_steps=STEPS,
    #     batch_size=BATCH_SIZE,
    #     track_per_expert_loss=True
    # )

    # Tanea experiment
    tanea_trainer = MoEPLRFTrainer(model, tanea_opt)
    key, train_key = random.split(key)
    tanea_results = tanea_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True
    )

    # Tarmsprop SGD experiment
    tarmsprop_sgd_trainer = MoEPLRFTrainer(model, tarmsprop_sgd_opt)
    key, train_key = random.split(key)
    tarmsprop_sgd_results = tarmsprop_sgd_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True
    )

    # Adam experiment
    adam_trainer = MoEPLRFTrainer(model, adam_opt)
    key, train_key = random.split(key)
    adam_results = adam_trainer.train(
        train_key,
        num_steps=STEPS,
        batch_size=BATCH_SIZE,  
        track_per_expert_loss=True
    )

    moe_results.append({
        'beta': beta,
        'model': model,
        # 'sgd': sgd_results,
        # 'dana_decaying': dana_decaying_results,
        'tanea': tanea_results,
        'tarmsprop_sgd': tarmsprop_sgd_results,
        'adam': adam_results
    })


# Plot results
plt.figure(figsize=(12, 8))

# Create plasma colormap for beta values
betas = [result['beta'] for result in moe_results]
colors = plt.cm.plasma(np.linspace(0, 0.8, len(betas)))

for i, result in enumerate(moe_results):
    beta = result['beta']
    color = colors[i]

    # Plot Tanea results
    tanea_times = result['tanea']['timestamps']
    tanea_losses = result['tanea']['losses']
    plt.loglog(tanea_times, tanea_losses, 
                color=color, linestyle='-', alpha=0.8, linewidth=2)

    # Plot Ta-RMS-prop results
    tarmsprop_sgd_times = result['tarmsprop_sgd']['timestamps']
    tarmsprop_sgd_losses = result['tarmsprop_sgd']['losses']
    plt.loglog(tarmsprop_sgd_times, tarmsprop_sgd_losses, 
                color=color, linestyle='--', alpha=0.8, linewidth=2)
    
    # Plot Adam results
    adam_times = result['adam']['timestamps']
    adam_losses = result['adam']['losses']
    plt.loglog(adam_times, adam_losses, 
                color=color, linestyle=':', alpha=0.8, linewidth=2)

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='gray', linestyle='-', linewidth=2, label='Tanea'),
    Line2D([0], [0], color='gray', linestyle='--', linewidth=2, label='Ta-RMS-prop'),
    Line2D([0], [0], color='gray', linestyle=':', linewidth=2, label='Adam (b1=0.0)')
]

# Add beta values to legend
for i, beta in enumerate(betas):
    legend_elements.append(Line2D([0], [0], color=colors[i], linewidth=2, label=f'β={beta}'))

plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('MoE PLRF Training Loss Comparison')
plt.legend(handles=legend_elements, loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
