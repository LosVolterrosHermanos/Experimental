import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the Python path to import power_law_rf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from power_law_rf import PowerLawRF

# def solve_least_squares(checkW, checkb):
#     """
#     Solve the least squares problem min_w ||checkW @ w - checkb||^2
    
#     Parameters
#     ----------
#     checkW : ndarray
#         Design matrix
#     checkb : ndarray
#         Target vector
        
#     Returns
#     -------
#     ndarray
#         Solution to the least squares problem
#     """
#     # Using the normal equation: (W^T W)w = W^T b
#     W_T_W = jnp.matmul(checkW.T, checkW)
#     W_T_b = jnp.matmul(checkW.T, checkb)
    
#     # Solve the linear system
#     w = jnp.linalg.solve(W_T_W, W_T_b)
#    return w

def test_limit_loss():
    """
    Test the limit loss prediction against the actual risk of the least-square solution
    for different values of alpha and beta.
    """
    # Fixed parameters
    v = 4000  # Hidden dimensionality
    d = 1000  # Embedded dimensionality
    
    # Alpha and beta values to test
    alphas = [0.3, 0.5, 0.7, 0.9]
    betas = [0.3, 0.5, 0.7, 0.9]
    
    # Initialize random key
    key = jax.random.PRNGKey(0)
    
    # Results storage
    results = []
    
    # Test each combination of alpha and beta
    for alpha in alphas:
        for beta in betas:
            print(f"Testing alpha={alpha}, beta={beta}")
            
            # Create PowerLawRF instance
            plrf = PowerLawRF.initialize_random(alpha=alpha, beta=beta, v=v, d=d, key=key)
            
            # Get theoretical limit loss
            theory_loss = plrf.get_theory_limit_loss()
            
            # Calculate actual risk
            actual_risk = plrf.get_empirical_limit_loss()
            
            # Store results
            results.append({
                'alpha': alpha,
                'beta': beta,
                'theory_loss': theory_loss,
                'actual_risk': actual_risk,
                'relative_error': abs(theory_loss - actual_risk) / theory_loss,
            })
            
            print(f"  Theory loss: {theory_loss:.6f}")
            print(f"  Actual risk: {actual_risk:.6f}")
            print(f"  Relative error: {abs(theory_loss - actual_risk) / theory_loss:.6f}")
    
    # Convert results to numpy array for easier plotting
    results_array = np.array([
        [r['alpha'], r['beta'], r['theory_loss'], r['actual_risk'], 
         r['relative_error']]
        for r in results
    ])
    
    # Plot results
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Theory loss vs Actual risk
    plt.subplot(2, 1, 1)
    for alpha in alphas:
        alpha_mask = results_array[:, 0] == alpha
        plt.plot(betas, results_array[alpha_mask, 2], 'o-', label=f'Theory (α={alpha})')
        plt.plot(betas, results_array[alpha_mask, 3], 's--', label=f'Actual (α={alpha})')
    
    plt.xlabel('β')
    plt.ylabel('Loss')
    plt.title('Theory Loss vs Actual Risk')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Relative error
    plt.subplot(2, 1, 2)
    for alpha in alphas:
        alpha_mask = results_array[:, 0] == alpha
        plt.plot(betas, results_array[alpha_mask, 4], 'o-', label=f'α={alpha}')
    
    plt.xlabel('β')
    plt.ylabel('Relative Error')
    plt.title('Relative Error: |Theory - Actual| / Theory')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    # Save plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'limit_loss_comparison.png'))
    plt.show()
    
    return results

if __name__ == "__main__":
    test_limit_loss() 