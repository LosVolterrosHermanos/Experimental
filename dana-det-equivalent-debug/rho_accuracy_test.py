import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import power_law_rf.deterministic_equivalent as det_eq
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.deterministic_equivalent import theory_rhos, deterministic_rho_weights, theory_lambda_min, deterministic_spectra

def compute_sum_rho_lambda_inv(rhos, lambdas, z):
    """Compute the sum of rho_j / (lambda_j + z) for all j."""
    return jnp.sum(rhos / (lambdas + z))

def compute_sum_rho_lambda_exp(rhos, lambdas, z):
    """Compute the sum of rho_j * lambda_j * exp(-z * lambda_j) for all j."""
    return jnp.sum(rhos * lambdas * jnp.exp(-z * lambdas))

def compute_relative_error(method1_sums, method2_sums):
    """Compute the relative error between the two methods."""
    return jnp.abs(method1_sums - method2_sums) / jnp.maximum(jnp.abs(method1_sums), jnp.abs(method2_sums))

def main():
    # Set parameters
    V = 4000  # Hidden dimensionality
    D = 1000  # Embedded dimensionality
    alphas = [0.3, 0.5, 0.7, 0.9]
    betas = [0.3, 0.5, 0.7, 0.9]
    # alphas = [0.3, 0.7]
    # betas = [0.3, 0.7]
    # Set random key for reproducibility
    key = jax.random.PRNGKey(42)
    
    # Create figures to plot results
    fig1, axes1 = plt.subplots(4, 4, figsize=(20, 20))
    fig1.suptitle(f"Comparison of rho accuracy (resolvent) for V={V}, D={D}", fontsize=16)
    
    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))
    fig2.suptitle(f"Relative error between methods (resolvent) for V={V}, D={D}", fontsize=16)
    
    fig3, axes3 = plt.subplots(4, 4, figsize=(20, 20))
    fig3.suptitle(f"Comparison of rho accuracy (exponential) for V={V}, D={D}", fontsize=16)
    
    fig4, axes4 = plt.subplots(4, 4, figsize=(20, 20))
    fig4.suptitle(f"Relative error between methods (exponential) for V={V}, D={D}", fontsize=16)
    
    # Store results for summary
    results = []
    
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            print(f"Processing alpha={alpha}, beta={beta}")
            
            # Method 2: Using PowerLawRF class
            plrf = PowerLawRF.initialize_random(alpha, beta, V, D, key)
            hessian_eigs = plrf.get_hessian_spectra()
            plrf_rhos = plrf.get_rhos()

            # Sort eigenvalues and rhos in descending order
            sorted_indices = jnp.argsort(hessian_eigs)[::-1]
            hessian_eigs = hessian_eigs[sorted_indices]
            plrf_rhos = plrf_rhos[sorted_indices]
            
            # Method 1: Using deterministic_equivalent.py
            lower_bound = 0.5*theory_lambda_min(alpha, V, D)
            upper_bound = 1.0*1.1
            fake_eigs = deterministic_spectra(V, D, alpha, xsplits=1000)
            fake_eigs = jnp.array(fake_eigs)
            b_values = fake_eigs - 0.5 * jnp.diff(fake_eigs, prepend = upper_bound)
            a_values = fake_eigs + 0.5 * jnp.diff(fake_eigs, append = lower_bound)

            # Define a wrapper function to ensure m_fn is properly passed
            def f_measure_wrapper(v, d, alpha, beta, xs, err=-20.0, time_checks=False, j_batch=100):
                return det_eq.theory_f_measure_unweighted(v, d, alpha, beta, xs, m_fn=det_eq.theory_m_batched, 
                                                         err=err, time_checks=time_checks, j_batch=j_batch)
            
            fake_rhos = plrf.get_deterministic_rho_weights(
                a_values, b_values, 
                f_measure_fn = f_measure_wrapper, 
                xs_per_split = 1000)
            
            fake_rhos = fake_rhos/fake_eigs

            # Generate z values for resolvent comparison
            min_z = D**(-2*alpha)
            max_z = 1.0
            z_values_resolvent = jnp.logspace(jnp.log10(min_z), jnp.log10(max_z), 20)
            
            # Generate z values for exponential comparison
            z_values_exp = jnp.linspace(0.0, 5.0, 20)
            
            # Compute sum for both methods (resolvent)
            method1_sums_resolvent = jnp.array([compute_sum_rho_lambda_inv(fake_rhos, fake_eigs, z) for z in z_values_resolvent])
            method2_sums_resolvent = jnp.array([compute_sum_rho_lambda_inv(plrf_rhos, hessian_eigs, z) for z in z_values_resolvent])
            
            # Compute sum for both methods (exponential)
            method1_sums_exp = jnp.array([compute_sum_rho_lambda_exp(fake_rhos, fake_eigs, z) for z in z_values_exp])
            method2_sums_exp = jnp.array([compute_sum_rho_lambda_exp(plrf_rhos, hessian_eigs, z) for z in z_values_exp])
            
            # Compute relative errors
            rel_error_resolvent = compute_relative_error(method1_sums_resolvent, method2_sums_resolvent)
            rel_error_exp = compute_relative_error(method1_sums_exp, method2_sums_exp)
            
            max_rel_error_resolvent = jnp.max(rel_error_resolvent)
            max_rel_error_exp = jnp.max(rel_error_exp)
            
            # Store results
            results.append({
                'alpha': alpha,
                'beta': beta,
                'max_rel_error_resolvent': max_rel_error_resolvent,
                'avg_rel_error_resolvent': jnp.mean(rel_error_resolvent),
                'max_rel_error_exp': max_rel_error_exp,
                'avg_rel_error_exp': jnp.mean(rel_error_exp)
            })
            
            # Plot resolvent comparison
            ax1 = axes1[i, j]
            ax1.loglog(z_values_resolvent, method1_sums_resolvent, 'b-', label='Method 1 (theory_rhos)')
            ax1.loglog(z_values_resolvent, method2_sums_resolvent, 'r--', label='Method 2 (PowerLawRF)')
            ax1.set_title(f'α={alpha}, β={beta}')
            ax1.set_xlabel('z')
            ax1.set_ylabel('∑(ρ_j² / (λ_j + z))')
            ax1.grid(True)
            ax1.legend()
            
            # Plot resolvent relative error
            ax2 = axes2[i, j]
            ax2.semilogx(z_values_resolvent, rel_error_resolvent, 'g-')
            ax2.set_title(f'α={alpha}, β={beta}, Max Error: {max_rel_error_resolvent:.4f}')
            ax2.set_xlabel('z')
            ax2.set_ylabel('Relative Error')
            ax2.grid(True)
            
            # Plot exponential comparison
            ax3 = axes3[i, j]
            ax3.plot(z_values_exp, method1_sums_exp, 'b-', label='Method 1 (theory_rhos)')
            ax3.plot(z_values_exp, method2_sums_exp, 'r--', label='Method 2 (PowerLawRF)')
            ax3.set_title(f'α={alpha}, β={beta}')
            ax3.set_xlabel('z')
            ax3.set_ylabel('∑(ρ_j² * λ_j * exp(-z * λ_j))')
            ax3.grid(True)
            ax3.legend()
            
            # Plot exponential relative error
            ax4 = axes4[i, j]
            ax4.plot(z_values_exp, rel_error_exp, 'g-')
            ax4.set_title(f'α={alpha}, β={beta}, Max Error: {max_rel_error_exp:.4f}')
            ax4.set_xlabel('z')
            ax4.set_ylabel('Relative Error')
            ax4.grid(True)
    
    # Print summary of results
    print("\nSummary of Results:")
    print("==================")
    print("Alpha | Beta | Max Rel Error (Resolvent) | Avg Rel Error (Resolvent) | Max Rel Error (Exp) | Avg Rel Error (Exp)")
    print("------|------|-------------------------|-------------------------|-------------------|------------------")
    for result in results:
        print(f"{result['alpha']:.1f} | {result['beta']:.1f} | {result['max_rel_error_resolvent']:.6f} | {result['avg_rel_error_resolvent']:.6f} | {result['max_rel_error_exp']:.6f} | {result['avg_rel_error_exp']:.6f}")
    
    # Save figures
    plt.figure(fig1.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_comparison_resolvent.pdf')
    
    plt.figure(fig2.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_error_resolvent.pdf')
    
    plt.figure(fig3.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_comparison_exp.pdf')
    
    plt.figure(fig4.number)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('rho_accuracy_error_exp.pdf')
    
    plt.show()

if __name__ == "__main__":
    main() 