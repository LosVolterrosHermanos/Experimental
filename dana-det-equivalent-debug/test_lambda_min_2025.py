import numpy as np
import jax.numpy as jnp
import sys
import os
import jax.random as random

# Add the parent directory to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_law_rf.deterministic_equivalent import theory_lambda_min
from power_law_rf.power_law_rf import PowerLawRF

def test_lambda_min_2025():
    """
    Test the theory_lambda_min function by comparing it with the smallest eigenvalue
    from PowerLawRF for various values of alpha.
    """
    # Parameters
    D = 1000
    V_D_ratio = 4
    V = D * V_D_ratio
    alphas = [0.3, 0.5, 0.7, 0.9]
    beta = 0.5  # Using a fixed beta value
    
    # Create a random key for JAX
    key = random.PRNGKey(42)
    
    print("Testing theory_lambda_min against actual smallest eigenvalues:")
    print("D = {}, V/D = {}".format(D, V_D_ratio))
    print("-" * 60)
    print("{:<10} {:<20} {:<20} {:<20}".format("Alpha", "Theory", "Actual", "Relative Error"))
    print("-" * 60)
    
    for alpha in alphas:
        # Calculate theoretical value
        theory_value = theory_lambda_min(alpha, V, D)
        
        # Create PowerLawRF instance and calculate actual smallest eigenvalue
        rf = PowerLawRF.initialize_random(alpha=alpha, beta=beta, v=V, d=D, key=key)
        spectra = rf.get_hessian_spectra()
        actual_value = np.min(spectra)  # Square the eigenvalues
        
        # Calculate relative error
        relative_error = abs(theory_value - actual_value) / actual_value
        
        print("{:<10.1f} {:<20.10f} {:<20.10f} {:<20.10f}".format(
            alpha, theory_value, actual_value, relative_error))
        
        # Generate a new key for the next iteration
        key, _ = random.split(key)
    
    print("-" * 60)

if __name__ == "__main__":
    test_lambda_min_2025() 