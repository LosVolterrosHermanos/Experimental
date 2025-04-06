import jax
import jax.numpy as jnp
import jax.random as random
import optax
import scipy as sp
import pickle
import os.path
import hashlib
import numpy as np

from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


import sys
import os
sys.path.append('../')
# sys.path.append('../power_law_rf')
import optimizers
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.ode import ode_resolvent_log_implicit
from power_law_rf.ode import DanaHparams
from power_law_rf.ode import ODEInputs
from power_law_rf.least_squares import lsq_streaming_optax_simple
import power_law_rf.deterministic_equivalent as theory


key = random.PRNGKey(0)

ALPHA = 1.2
BETA = 0.7
V = 4000
D = 1000
SGDBATCH=1
STEPS = 10**5
G2_SCALE = 0.5
G3_IV = 0.1

key, subkey = random.split(key)
problem = PowerLawRF.initialize_random(alpha=ALPHA, beta=BETA, v=V, d=D, key=subkey)

g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
g2 = optimizers.powerlaw_schedule(G2_SCALE/problem.population_trace, 0.0, 0.0, 1)
g3 = optimizers.powerlaw_schedule(G3_IV/problem.population_trace, 0.0, -1.0/(2*problem.alpha), 1)
Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, 4.0+2*(problem.alpha+problem.beta)/(2*problem.alpha))
danadecayopt = optimizers.dana_optimizer(g1=g1,g2=g2,g3=g3,Delta=Delta)

# Create a unique identifier for the current parameter configuration
param_dict = {
    'alpha': ALPHA,
    'beta': BETA,
    'v': V,
    'd': D,
    'sgdbatch': SGDBATCH,
    'steps': STEPS,
    'g1_params': (1.0, 0.0, 0.0, 1),
    'g2_params': (G2_SCALE/problem.population_trace, 0.0, 0.0, 1),
    'g3_params': (G3_IV/problem.population_trace, 0.0, -1.0/(2*problem.alpha), 1),
    'delta_params': (1.0, 0.0, -1.0, 4.0+2*(problem.alpha+problem.beta)/(2*problem.alpha))
}

# Create a hash of the parameters for the filename
param_str = str(param_dict)
filename_hash = hashlib.md5(param_str.encode()).hexdigest()[:10]
pickle_filename = f'loss-curve-cache/lsq_results_alpha{ALPHA}_beta{BETA}_V{V}_D{D}_{filename_hash}.pkl'

# Check if we have saved results with these parameters
if os.path.exists(pickle_filename):
    print(f"Loading saved results from {pickle_filename}")
    with open(pickle_filename, 'rb') as f:
        danadecaytimes, danadecaylosses = pickle.load(f)
else:
    print(f"Generating new results and saving to {pickle_filename}")
    key, newkey = random.split(key)
    danadecaytimes, danadecaylosses = lsq_streaming_optax_simple(newkey, 
                             problem.get_data, 
                             SGDBATCH, 
                             STEPS, 
                             danadecayopt, 
                             jnp.zeros((problem.d,1)), 
                             problem.get_population_risk)
    
    # Save the results
    with open(pickle_filename, 'wb') as f:
        pickle.dump((danadecaytimes, danadecaylosses), f)

#We will plot double the loss
twicedanadecaylosses = np.array(danadecaylosses) * 2.0

#Initialize the rhos
initTheta = jnp.zeros(problem.d, dtype=jnp.float32)
initY = jnp.zeros(problem.d, dtype=jnp.float32)

D=problem.d

lower_bound = 0.5*theory.theory_lambda_min(problem.alpha, problem.v, problem.d)
upper_bound = 1.0*1.1
#fake_eigs = jnp.power(jnp.arange(1,D+1,dtype=jnp.float32),-2.0*problem.alpha)
fake_eigs = theory.deterministic_spectra(problem.v, problem.d, problem.alpha)
fake_eigs = jnp.array(fake_eigs)
b_values = fake_eigs - 0.5 * jnp.diff(fake_eigs, prepend = upper_bound)
a_values = fake_eigs + 0.5 * jnp.diff(fake_eigs, append = lower_bound)

#num_splits = 5
rho_weights = theory.deterministic_rho_weights(problem.v, problem.d, problem.alpha, problem.beta, a_values, b_values)

# Check if any rho weights are negative
negative_rhos = jnp.any(rho_weights < 0)
if negative_rhos:
    print("WARNING: Some rho weights are negative!")
    negative_indices = jnp.where(rho_weights < 0)[0]
    print(f"Negative rho indices: {negative_indices}")
    print(f"Negative rho values: {rho_weights[negative_indices]}")
else:
    print("All rho weights are non-negative.")

riskInftyTheory=problem.get_theory_limit_loss()
print('Initial twice-loss value is is {}'.format(jnp.sum( rho_weights*fake_eigs) + 2.0*riskInftyTheory))
rho_init = rho_weights
sigma_init = jnp.zeros_like(rho_init)
chi_init = jnp.zeros_like(rho_init)

ODE_d = len(fake_eigs)

Dt = 10**(-3)

# Get exact ODE solution
odeTimes_dana_decay3, odeRisks_dana_decay3 = ode_resolvent_log_implicit(
    ODEInputs(fake_eigs, rho_init, chi_init, sigma_init, riskInftyTheory),
    DanaHparams(g1, g2, g3, Delta),
    SGDBATCH, ODE_d, STEPS, Dt)

# Get approximate ODE solution
odeTimes_approx, odeRisks_approx = ode_resolvent_log_implicit(
    ODEInputs(fake_eigs, rho_init, chi_init, sigma_init, riskInftyTheory),
    DanaHparams(g1, g2, g3, Delta),
    SGDBATCH, ODE_d, STEPS, Dt,
    approximate=True)

# Create comparison plot
plt.figure(figsize=(10, 6))
plt.loglog(danadecaytimes, twicedanadecaylosses, 'b-', label='Algorithm (lsq_streaming_optax_simple)', linewidth=2)
plt.loglog(odeTimes_dana_decay3, odeRisks_dana_decay3, 'r--', label='Theory (exact ODE)', linewidth=2)
plt.loglog(odeTimes_approx, odeRisks_approx, 'g--', label='Theory (approximate ODE)', linewidth=2)
plt.axhline(y=2.0*riskInftyTheory/(1-G2_SCALE/2.0), color='k', linestyle=':', label='Theoretical Limit (SGD adjusted)', linewidth=2)

plt.xlabel('Time (log scale)')
plt.ylabel('Twice-Loss')
plt.title(f'Comparison of Algorithm vs Theory (α={ALPHA}, β={BETA}, V={V}, D={D})')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig(f'algorithm_vs_theory_comparison_alpha{ALPHA}_beta{BETA}_V{V}_D{D}.pdf')
plt.show()

# Print final values for comparison
print(f"Final algorithm loss: {twicedanadecaylosses[-1]:.6f}")
print(f"Final exact theory loss: {odeRisks_dana_decay3[-1]:.6f}")
print(f"Final approximate theory loss: {odeRisks_approx[-1]:.6f}")
print(f"Theoretical limit: {riskInftyTheory:.6f}")
print(f"Adjusted theoretical limit (SGD): {2.0*riskInftyTheory/(1-G2_SCALE/2.0):.6f}")