import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from power_law_rf.power_law_rf import PowerLawRF
from power_law_rf.deterministic_equivalent import deterministic_spectra 

# Set parameters
V = 4000
D = 1000
alpha = 0.9
beta = 0.9

# Generate deterministic spectra
print("Generating deterministic spectra...")
det_spectra = deterministic_spectra(V, D, alpha, xsplits=1000)
print(f"Generated {len(det_spectra)} deterministic eigenvalues")

# Generate random equivalent
print("\nGenerating random equivalent...")
key = jax.random.PRNGKey(0)
plrf = PowerLawRF.initialize_random(alpha=alpha, beta=beta, v=V, d=D, key=key)
random_spectra = plrf.get_hessian_spectra()
print(f"Generated {len(random_spectra)} random eigenvalues")

# Sort both spectra in descending order
det_spectra_sorted = np.sort(det_spectra)[::-1]
random_spectra_sorted = np.sort(random_spectra)[::-1]

# Plot comparison
plt.figure(figsize=(12, 8))

# Plot deterministic spectra
plt.semilogy(range(1, len(det_spectra_sorted) + 1), det_spectra_sorted, 'b-', label='Deterministic', linewidth=2)

# Plot random spectra
plt.semilogy(range(1, len(random_spectra_sorted) + 1), random_spectra_sorted, 'r--', label='Random', linewidth=2)

plt.xlabel('Index')
plt.ylabel('Eigenvalue (log scale)')
plt.title(f'Comparison of Deterministic vs Random Spectra\nV={V}, D={D}, α={alpha}, β={beta}')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('spectra_comparison.png')
print("Plot saved as 'spectra_comparison.png'")

# Print some statistics
print("\nStatistics:")
print(f"Deterministic spectra: min={det_spectra_sorted.min():.6f}, max={det_spectra_sorted.max():.6f}, count={len(det_spectra_sorted)}")
print(f"Random spectra: min={random_spectra_sorted.min():.6f}, max={random_spectra_sorted.max():.6f}, count={len(random_spectra_sorted)}")

# Compare the first few eigenvalues
print("\nFirst 10 eigenvalues comparison:")
print("Index | Deterministic | Random")
print("-" * 40)
for i in range(min(10, min(len(det_spectra_sorted), len(random_spectra_sorted)))):
    print(f"{i+1:5d} | {det_spectra_sorted[i]:.6f} | {random_spectra_sorted[i]:.6f}") 