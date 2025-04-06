import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import power_law_rf.deterministic_equivalent as det_eq
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.deterministic_equivalent import theory_lambda_min, deterministic_spectra, deterministic_rho_weights

def compute_relative_error(method1_values, method2_values):
    """Compute the relative error between the two methods."""
    return jnp.abs(method1_values - method2_values) / jnp.maximum(jnp.abs(method1_values), jnp.abs(method2_values))

def integrate_f_measure(v, d, alpha, beta, lower_bound, upper_bound, num_points=10000):
    """
    Directly integrate the theory_f_measure over the interval from lower_bound to upper_bound.
    
    Parameters
    ----------
    v, d, alpha, beta : floats
        parameters of the model
    lower_bound, upper_bound : floats
        the interval to integrate over
    num_points : int
        number of points to use for integration
        
    Returns
    -------
    float
        the integral of the f_measure over the interval
    """
    # Create a fine grid of points over the interval
    xs = jnp.linspace(lower_bound, upper_bound, num_points)
    
    # Define a wrapper function to ensure m_fn is properly passed
    def f_measure_wrapper(v, d, alpha, beta, xs, err=-20.0, time_checks=False, j_batch=100):
        return det_eq.theory_f_measure(v, d, alpha, beta, xs, m_fn=det_eq.theory_m_batched, 
                                     err=err, time_checks=time_checks, j_batch=j_batch)
    
    # Compute the f_measure at each point
    density = f_measure_wrapper(v, d, alpha, beta, xs)
    
    # Integrate using the trapezoidal rule
    dx = xs[1] - xs[0]
    integral = jnp.sum(density*xs) * dx
    
    return integral

def main():
    # Set parameters
    V = 12800  # Hidden dimensionality
    D = 3200  # Embedded dimensionality
    alpha = 0.3
    beta = 0.9
    
    # Set random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    print(f"Processing alpha={alpha}, beta={beta}")
    
    # Method 3: Using PowerLawRF class (random instance)
    plrf = PowerLawRF.initialize_random(alpha, beta, V, D, key)
    hessian_eigs = plrf.get_hessian_spectra()
    plrf_rhos = plrf.get_rhos()
    
    # Sort eigenvalues and rhos in descending order
    sorted_indices = jnp.argsort(hessian_eigs)[::-1]
    hessian_eigs = hessian_eigs[sorted_indices]
    plrf_rhos = plrf_rhos[sorted_indices]
    
    # Method 1: Direct integration of theory_f_measure
    lower_bound = 0.5 * theory_lambda_min(alpha, V, D)
    upper_bound = 1.0 * 1.1
    
    # Integrate the f_measure over the interval
    integrated_measure = integrate_f_measure(V, D, alpha, beta, lower_bound, upper_bound)
    
    # Sum of the rho_j's from the random instance
    sum_plrf_rhos = jnp.sum(plrf_rhos*hessian_eigs)
    
    # Method 2: Using deterministic_rho_weights with deterministic_spectra
    fake_eigs = deterministic_spectra(V, D, alpha, xs_per_split=D)
    fake_eigs = jnp.array(fake_eigs)
    b_values = fake_eigs - 0.5 * jnp.diff(fake_eigs, prepend=upper_bound)
    a_values = fake_eigs + 0.5 * jnp.diff(fake_eigs, append=lower_bound)
    
    # Define a wrapper function to ensure m_fn is properly passed
    def f_measure_wrapper(v, d, alpha, beta, xs, err=-20.0, time_checks=False, j_batch=100):
        return det_eq.theory_f_measure(v, d, alpha, beta, xs, m_fn=det_eq.theory_m_batched, 
                                     err=err, time_checks=time_checks, j_batch=j_batch)
    
    # Get deterministic rho weights
    fake_rhos = deterministic_rho_weights(V, D, alpha, beta, a_values, b_values, 
                                         f_measure_fn=f_measure_wrapper, 
                                         xs_per_split=D)
    
    # Sum of the deterministic rho weights
    sum_deterministic_rhos = jnp.sum(fake_rhos*fake_eigs)
    
    # Compute relative errors
    rel_error_integration = compute_relative_error(integrated_measure, sum_plrf_rhos)
    rel_error_deterministic = compute_relative_error(sum_deterministic_rhos, sum_plrf_rhos)
    rel_error_integration_deterministic = compute_relative_error(integrated_measure, sum_deterministic_rhos)
    
    # Print summary statistics
    print("\nSummary of Results:")
    print("==================")
    print(f"Method 1 - Integrated measure: {integrated_measure:.6f}")
    print(f"Method 2 - Sum of deterministic rho weights: {sum_deterministic_rhos:.6f}")
    print(f"Method 3 - Sum of rho_j's from random instance: {sum_plrf_rhos:.6f}")
    print(f"Relative error (Method 1 vs Method 3): {rel_error_integration:.6f}")
    print(f"Relative error (Method 2 vs Method 3): {rel_error_deterministic:.6f}")
    print(f"Relative error (Method 1 vs Method 2): {rel_error_integration_deterministic:.6f}")
    
    # Create plots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle(f"Three Methods Comparison for α={alpha}, β={beta}, V={V}, D={D}", fontsize=16)
    
    # Plot eigenvalues
    ax1 = axes[0]
    ax1.semilogy(range(len(hessian_eigs)), hessian_eigs, 'b-', label='Hessian Eigenvalues')
    ax1.semilogy(range(len(fake_eigs)), fake_eigs, 'r--', label='Deterministic Eigenvalues')
    ax1.set_title('Eigenvalues Comparison')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.grid(True)
    ax1.legend()
    
    # Plot weights
    ax2 = axes[1]
    ax2.semilogy(range(len(plrf_rhos)), plrf_rhos, 'b-', label='PowerLawRF Weights')
    ax2.semilogy(range(len(fake_rhos)), fake_rhos, 'r--', label='Deterministic Weights')
    ax2.set_title('Weights Comparison')
    ax2.set_xlabel('Index')
    ax2.set_ylabel('Weight (log scale)')
    ax2.grid(True)
    ax2.legend()
    
    # Plot relative errors
    ax3 = axes[2]
    methods = ['Method 1', 'Method 2', 'Method 3']
    values = [integrated_measure, sum_deterministic_rhos, sum_plrf_rhos]
    ax3.bar(methods, values, color=['blue', 'green', 'red'])
    ax3.set_title('Comparison of Sum Values')
    ax3.set_ylabel('Sum Value')
    ax3.grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_three_methods_comparison.pdf')
    plt.show()

if __name__ == "__main__":
    main() 