import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.deterministic_equivalent import theory_rhos, deterministic_rho_weights, theory_lambda_min, deterministic_spectra

def compute_relative_error(method1_values, method2_values):
    """Compute the relative error between the two methods."""
    return jnp.abs(method1_values - method2_values) / jnp.maximum(jnp.abs(method1_values), jnp.abs(method2_values))

def main():
    # Set parameters
    V = 400  # Hidden dimensionality
    D = 100  # Embedded dimensionality
    alpha = 0.3
    beta = 0.4
    
    # Set random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    print(f"Processing alpha={alpha}, beta={beta}")
    
    # Method 2: Using PowerLawRF class
    plrf = PowerLawRF.initialize_random(alpha, beta, V, D, key)
    hessian_eigs = plrf.get_hessian_spectra()
    plrf_rhos = plrf.get_rhos()
    
    # Method 1: Using deterministic_equivalent.py
    lower_bound = 0.5*theory_lambda_min(alpha, V, D)
    upper_bound = 1.0*1.1
    fake_eigs = deterministic_spectra(V, D, alpha, xsplits=1000)
    fake_eigs = jnp.array(fake_eigs)
    b_values = fake_eigs - 0.5 * jnp.diff(fake_eigs, prepend = upper_bound)
    a_values = fake_eigs + 0.5 * jnp.diff(fake_eigs, append = lower_bound)
    
    fake_rhos = plrf.get_deterministic_rho_weights(a_values, b_values, xs_per_split = 1000)
    
    # Sort eigenvalues and rhos in descending order
    sorted_indices = jnp.argsort(hessian_eigs)[::-1]
    hessian_eigs = hessian_eigs[sorted_indices]
    plrf_rhos = plrf_rhos[sorted_indices]
    
    # Ensure fake_eigs and fake_rhos have the same length as hessian_eigs and plrf_rhos
    min_len = min(len(fake_eigs), len(hessian_eigs))
    fake_eigs = fake_eigs[:min_len]
    fake_rhos = fake_rhos[:min_len]
    hessian_eigs = hessian_eigs[:min_len]
    plrf_rhos = plrf_rhos[:min_len]
    
    # Compute relative errors
    rel_error_eigs = compute_relative_error(fake_eigs, hessian_eigs)
    rel_error_rhos = compute_relative_error(fake_rhos, plrf_rhos)
    
    # Print summary statistics
    print("\nSummary of Results:")
    print("==================")
    print(f"Number of eigenvalues compared: {min_len}")
    print(f"Max relative error in eigenvalues: {jnp.max(rel_error_eigs):.6f}")
    print(f"Average relative error in eigenvalues: {jnp.mean(rel_error_eigs):.6f}")
    print(f"Max relative error in weights: {jnp.max(rel_error_rhos):.6f}")
    print(f"Average relative error in weights: {jnp.mean(rel_error_rhos):.6f}")
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f"Direct Comparison for α={alpha}, β={beta}, V={V}, D={D}", fontsize=16)
    
    # Plot eigenvalues
    ax1 = axes[0, 0]
    ax1.semilogy(range(min_len), hessian_eigs, 'b-', label='Hessian Eigenvalues')
    ax1.semilogy(range(min_len), fake_eigs, 'r--', label='Deterministic Eigenvalues')
    ax1.set_title('Eigenvalues Comparison')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot relative error in eigenvalues
    ax2 = axes[0, 1]
    ax2.semilogy(range(min_len), rel_error_eigs, 'g-')
    ax2.set_title(f'Relative Error in Eigenvalues (Max: {jnp.max(rel_error_eigs):.4f})')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Relative Error (log scale)')
    ax2.grid(True)
    
    # Plot weights
    ax3 = axes[1, 0]
    ax3.semilogy(range(min_len), plrf_rhos, 'b-', label='PowerLawRF Weights')
    ax3.semilogy(range(min_len), fake_rhos, 'r--', label='Deterministic Weights')
    ax3.set_title('Weights Comparison')
    ax3.set_xlabel('Index')
    ax3.set_ylabel('Weight (log scale)')
    ax3.grid(True)
    ax3.legend()
    
    # Plot relative error in weights
    ax4 = axes[1, 1]
    ax4.semilogy(range(min_len), rel_error_rhos, 'g-')
    ax4.set_title(f'Relative Error in Weights (Max: {jnp.max(rel_error_rhos):.4f})')
    ax4.set_xlabel('Index')
    ax4.set_ylabel('Relative Error (log scale)')
    ax4.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_direct_comparison.pdf')
    plt.show()

if __name__ == "__main__":
    main() 