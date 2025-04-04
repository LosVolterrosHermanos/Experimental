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
import optimizers
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.ode import ode_resolvent_log_implicit
from power_law_rf.ode import DanaHparams
from power_law_rf.ode import ODEInputs
from power_law_rf.least_squares import lsq_streaming_optax_simple
import power_law_rf.deterministic_equivalent as theory

# Store the original chunk_weights function
original_chunk_weights = theory.chunk_weights

# Create a new chunk_weights function that handles complex numbers
def patched_chunk_weights(xs, density, a, b):
    # Compute integrals
    integrals = []
    def theoretical_integral(lower, upper):
        # Find indices corresponding to interval [a,b]
        dx = xs[1] - xs[0]
        idx = (xs >= lower) & (xs <= upper)
        integral = jnp.sum(density[idx]) * dx
        # Handle complex numbers by taking real part if needed
        if jnp.iscomplexobj(integral):
            integral = integral.real
        return integral
    
    i = 0
    for lower, upper in zip(a, b):
        integrals.append(theoretical_integral(lower, upper))
        i = i + 1
    return integrals

# Replace the function in the module
theory.chunk_weights = patched_chunk_weights

# Fixed parameters
ALPHA = 0.3
BETA = 0.7
SGDBATCH = 1
STEPS = 10**6

# Dimensions to test
D_VALUES = [250, 500, 750, 1000, 1250]
V_D_RATIO = 4

# Arrays to store results
final_losses = []
theory_limits = []
dimensions = []

for D in D_VALUES:
    V = D * V_D_RATIO
    print(f"\nProcessing D={D}, V={V}")
    
    key = random.PRNGKey(0)
    key, subkey = random.split(key)
    problem = PowerLawRF.initialize_random(alpha=ALPHA, beta=BETA, v=V, d=D, key=subkey)

    # Set up optimizer parameters
    g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = optimizers.powerlaw_schedule(0.5/problem.population_trace, 0.0, 0.0, 1)
    g3 = optimizers.powerlaw_schedule(1.0/problem.population_trace, 0.0, -1.0/(2*problem.alpha), 1)
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
        'g2_params': (0.5/problem.population_trace, 0.0, 0.0, 1),
        'g3_params': (1.0/problem.population_trace, 0.0, -1.0/(2*problem.alpha), 1),
        'delta_params': (1.0, 0.0, -1.0, 4.0+2*(problem.alpha+problem.beta)/(2*problem.alpha))
    }

    # Create a hash of the parameters for the filename
    param_str = str(param_dict)
    filename_hash = hashlib.md5(param_str.encode()).hexdigest()[:10]
    pickle_filename = f'lsq_results_alpha{ALPHA}_beta{BETA}_V{V}_D{D}_{filename_hash}.pkl'

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

    # Correct for the factor of 2 discrepancy
    danadecaylosses = np.array(danadecaylosses) * 0.5
    
    # Get theoretical limit
    riskInftyTheory = problem.get_theory_limit_loss()
    
    # Store results
    final_losses.append(danadecaylosses[-1])
    theory_limits.append(riskInftyTheory)
    dimensions.append(D)

# Create comparison plot
plt.figure(figsize=(10, 6))
plt.plot(dimensions, final_losses, 'bo-', label='SGD Final Loss', linewidth=2)
plt.plot(dimensions, theory_limits, 'ro-', label='Theoretical Limit', linewidth=2)

plt.xlabel('Dimension (D)')
plt.ylabel('Loss')
plt.title(f'Comparison of Final SGD Loss vs Theoretical Limit\n(α={ALPHA}, β={BETA}, V/D={V_D_RATIO})')
plt.grid(True, which="both", ls="-", alpha=0.2)
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig('theory_vs_sgd_comparison.png', dpi=300)
plt.show()

# Print numerical results
print("\nNumerical Results:")
print("D\tV\tSGD Loss\tTheory Limit\tRatio")
print("-" * 60)
for D, V, sgd_loss, theory_loss in zip(dimensions, [d*V_D_RATIO for d in dimensions], final_losses, theory_limits):
    print(f"{D}\t{V}\t{sgd_loss:.6f}\t{theory_loss:.6f}\t{sgd_loss/theory_loss:.3f}")

# Restore the original function
theory.chunk_weights = original_chunk_weights 