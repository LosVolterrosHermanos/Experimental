import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the parent directory to the Python path to import power_law_rf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from power_law_rf import PowerLawRF

def test_kernel_norm():
    """
    Test the kernel norm against the population trace for different values of alpha and beta.
    The kernel norm is defined as the sum of the hessian spectra of the problem.
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
            key, subkey = jax.random.split(key)
            plrf = PowerLawRF.initialize_random(alpha=alpha, beta=beta, v=v, d=d, key=subkey)
            
            # Get population trace
            population_trace = plrf.population_trace
            
            # Calculate kernel norm (sum of hessian spectra)
            kernel_norm = jnp.sum(plrf.get_hessian_spectra())
            
            # Store results
            results.append({
                'alpha': alpha,
                'beta': beta,
                'population_trace': population_trace,
                'kernel_norm': kernel_norm,
                'ratio': kernel_norm / population_trace,
            })
            
            print(f"  Population trace: {population_trace:.6f}")
            print(f"  Kernel norm: {kernel_norm:.6f}")
            print(f"  Ratio: {kernel_norm / population_trace:.6f}")
    
    # Convert results to numpy array for easier plotting
    results_array = np.array([
        [r['alpha'], r['beta'], r['population_trace'], r['kernel_norm'], 
         r['ratio']]
        for r in results
    ])
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Population trace vs Kernel norm
    plt.subplot(2, 2, 1)
    for alpha in alphas:
        alpha_mask = results_array[:, 0] == alpha
        plt.plot(betas, results_array[alpha_mask, 2], 'o-', label=f'Population Trace (α={alpha})')
        plt.plot(betas, results_array[alpha_mask, 3], 's--', label=f'Kernel Norm (α={alpha})')
    
    plt.xlabel('β')
    plt.ylabel('Value')
    plt.title('Population Trace vs Kernel Norm')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Ratio of Kernel Norm to Population Trace
    plt.subplot(2, 2, 2)
    for alpha in alphas:
        alpha_mask = results_array[:, 0] == alpha
        plt.plot(betas, results_array[alpha_mask, 4], 'o-', label=f'α={alpha}')
    
    plt.xlabel('β')
    plt.ylabel('Ratio (Kernel Norm / Population Trace)')
    plt.title('Ratio of Kernel Norm to Population Trace')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Heatmap of Population Trace
    plt.subplot(2, 2, 3)
    heatmap_data = np.zeros((len(alphas), len(betas)))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            mask = (results_array[:, 0] == alpha) & (results_array[:, 1] == beta)
            heatmap_data[i, j] = results_array[mask, 2][0]
    
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Population Trace')
    plt.xticks(range(len(betas)), betas)
    plt.yticks(range(len(alphas)), alphas)
    plt.xlabel('β')
    plt.ylabel('α')
    plt.title('Population Trace Heatmap')
    
    # Plot 4: Heatmap of Kernel Norm
    plt.subplot(2, 2, 4)
    heatmap_data = np.zeros((len(alphas), len(betas)))
    for i, alpha in enumerate(alphas):
        for j, beta in enumerate(betas):
            mask = (results_array[:, 0] == alpha) & (results_array[:, 1] == beta)
            heatmap_data[i, j] = results_array[mask, 3][0]
    
    plt.imshow(heatmap_data, cmap='viridis', aspect='auto')
    plt.colorbar(label='Kernel Norm')
    plt.xticks(range(len(betas)), betas)
    plt.yticks(range(len(alphas)), alphas)
    plt.xlabel('β')
    plt.ylabel('α')
    plt.title('Kernel Norm Heatmap')
    
    plt.tight_layout()
    # Save plot in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, 'kernel_norm_comparison.png'))
    plt.show()
    
    # Print numerical results
    print("\nNumerical Results:")
    print("α\tβ\tPopulation Trace\tKernel Norm\tRatio")
    print("-" * 70)
    for r in results:
        print(f"{r['alpha']}\t{r['beta']}\t{r['population_trace']:.6f}\t{r['kernel_norm']:.6f}\t{r['ratio']:.6f}")
    
    # Print explanation
    print("\nExplanation:")
    print("The population trace is the sum of the power-law decaying eigenvalues: sum(j^(-alpha)) for j from 1 to v.")
    print("The kernel norm is the sum of the squared singular values of the checkW matrix (W * population_eigenvalues.T).")
    print("These are different quantities, and their ratio varies with the random initialization of the W matrix.")
    print("The population trace depends only on alpha, while the kernel norm depends on both alpha and beta.")
    
    return results

if __name__ == "__main__":
    test_kernel_norm() 