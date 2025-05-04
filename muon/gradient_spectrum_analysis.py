#!/usr/bin/env python3
# Copyright 2025 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gradient Spectrum Analysis for SignSVD vs SGD Comparison.

This script analyzes the singular value spectrum of gradients at initialization
for different input distributions, to help understand when SignSVD might outperform SGD.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
import matplotlib.pyplot as plt
import os
from typing import Tuple, Callable

# Import the necessary functions from the main experiment script
from signsvd_vs_sgd_comparison import (
    init_params, lin_mat_model_batched, compute_loss_and_grad, mse_loss,
    get_x_iid, get_x_corr, get_x_gen_cov, gen_cov_mat, get_psd_mat_sqrt
)

# Create a directory for output plots if it doesn't exist
os.makedirs("plots/spectrum", exist_ok=True)

def analyze_gradient_spectrum(
    N_in: int, 
    N_out: int, 
    batch_size: int, 
    sampling_fn: Callable, 
    seed: int = 42,
    num_samples: int = 100,
    experiment_name: str = "default",
    experiment_desc: str = "Default experiment"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze the singular value spectrum of gradients at initialization.
    
    Args:
        N_in: Input dimension
        N_out: Output dimension
        batch_size: Batch size for training
        sampling_fn: Function to sample input data
        seed: Random seed
        num_samples: Number of gradient samples to analyze
        experiment_name: Name for saving files
        experiment_desc: Description for plot titles
        
    Returns:
        Tuple of (singular_values, singular_values_normalized)
    """
    # Initialize keys
    main_key = random.PRNGKey(seed)
    keys = random.split(main_key, num_samples + 2)
    init_key, target_key = keys[0], keys[1]
    sample_keys = keys[2:]
    
    # Initialize parameters
    W = init_params(init_key, N_in, N_out)
    target_W = init_params(target_key, N_in, N_out)
    
    # Collect singular values from multiple gradient samples
    all_singular_values = []
    
    for i in range(num_samples):
        # Generate a batch of data
        key = sample_keys[i]
        x_in_batch, x_out_batch = sampling_fn(key)
        
        # Compute gradient
        _, grad = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
        
        # Compute singular values
        s = jnp.linalg.svd(grad, compute_uv=False)
        all_singular_values.append(s)
    
    # Stack all samples
    singular_values = jnp.vstack(all_singular_values)
    
    # Normalize each row by its maximum value for visualization
    singular_values_normalized = singular_values / jnp.max(singular_values, axis=1, keepdims=True)
    
    # Plot the singular value spectrum
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Raw singular values (average and individual samples)
    plt.subplot(2, 1, 1)
    for i in range(min(10, num_samples)):  # Plot just a few samples to avoid clutter
        plt.semilogy(singular_values[i], 'gray', alpha=0.3)
        
    avg_singular_values = jnp.mean(singular_values, axis=0)
    plt.semilogy(avg_singular_values, 'r-', linewidth=2, label='Mean')
    plt.xlabel('Index')
    plt.ylabel('Singular Value')
    plt.title(f'Singular Value Spectrum at Initialization - {experiment_desc}')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Plot 2: Histogram of singular values
    plt.subplot(2, 1, 2)
    # Flatten all singular values
    flat_values = singular_values.flatten()
    max_val = jnp.max(flat_values)
    
    # Create histogram with 20 bins on a log scale
    log_bins = np.logspace(np.log10(max(1e-10, jnp.min(flat_values))), 
                           np.log10(max_val), 
                           20)
    plt.hist(flat_values, bins=log_bins, alpha=0.7)
    plt.xscale('log')
    plt.xlabel('Singular Value (log scale)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of Singular Values - {experiment_desc}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"plots/spectrum/exp_spectrum_{experiment_name}.pdf")
    plt.close()
    
    return singular_values, singular_values_normalized

def compare_condition_numbers(experiments_data, experiment_names, experiment_descs):
    """Compare condition numbers across different experiments."""
    plt.figure(figsize=(10, 6))
    
    for i, (name, desc, data) in enumerate(zip(experiment_names, experiment_descs, experiments_data)):
        # Calculate condition numbers for each sample
        condition_numbers = data[:, 0] / data[:, -1]  # max / min singular value
        
        # Plot boxplot
        bp = plt.boxplot(condition_numbers, positions=[i+1], patch_artist=True,
                 widths=0.6, showfliers=False)
        
        # Color the boxes
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        bp['boxes'][0].set_facecolor(colors[i % len(colors)])
    
    plt.xlabel('Experiment')
    plt.ylabel('Condition Number (log scale)')
    plt.yscale('log')
    plt.title('Gradient Condition Numbers Across Experiments')
    plt.xticks(range(1, len(experiment_names)+1), [name.replace('_', ' ').title() for name in experiment_names])
    plt.grid(True, which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig("plots/spectrum/condition_number_comparison.pdf")
    plt.close()

def run_gradient_analysis():
    """Run gradient spectrum analysis for all experiments."""
    
    # Common parameters
    N_in = 200
    N_out = 200
    batch_size = 200
    seed = 42
    num_gradient_samples = 100
    
    # --------------------------------
    # Experiment 1: I.I.D. Inputs
    # --------------------------------
    print("Analyzing gradient spectrum for Experiment 1: I.I.D. Inputs")
    
    # Create sampling function for IID inputs
    def sampling_fn_iid(key):
        return get_x_iid(key, N_in=N_in, N_out=N_out, B=batch_size)
    
    sv_iid, sv_iid_norm = analyze_gradient_spectrum(
        N_in, N_out, batch_size, sampling_fn_iid, seed=seed,
        num_samples=num_gradient_samples,
        experiment_name="iid_inputs",
        experiment_desc="IID Inputs"
    )
    
    # --------------------------------
    # Experiment 2: Correlated Inputs
    # --------------------------------
    print("Analyzing gradient spectrum for Experiment 2: Correlated Inputs")
    
    # Create correlation matrix
    corr_key = random.PRNGKey(123)
    proj_mat = random.normal(corr_key, (N_out, N_in)) / jnp.sqrt(N_in)
    correlation_strength = 1.0-(1.0/N_in)  # High correlation
    
    # Create sampling function
    def sampling_fn_corr(key):
        return get_x_corr(key, N_in=N_in, N_out=N_out, 
                        proj_mat=proj_mat, cor=correlation_strength, B=batch_size)
    
    sv_corr, sv_corr_norm = analyze_gradient_spectrum(
        N_in, N_out, batch_size, sampling_fn_corr, seed=seed,
        num_samples=num_gradient_samples,
        experiment_name="correlated_inputs",
        experiment_desc=f"Correlated Inputs (c={correlation_strength})"
    )
    
    # --------------------------------
    # Experiment 3: Ill-Conditioned Input Covariance
    # --------------------------------
    print("Analyzing gradient spectrum for Experiment 3: Ill-Conditioned Input Covariance")
    
    # Create ill-conditioned covariance matrix
    cov_key = random.PRNGKey(456)
    
    # Define eigenvalue spectra with large condition numbers
    # Create power-law spectra where eigenvalues decay as 1/i^alpha
    alpha = 1.0  # Power-law exponent
    spec_x_in = jnp.power(jnp.arange(1, N_in + 1), -alpha)  # Power-law spectrum
    spec_x_out = jnp.power(jnp.arange(1, N_out + 1), -alpha)  # Power-law spectrum
    spec_mixed = jnp.zeros(N_in + N_out)  # No mixed term
    
    # Generate covariance matrix
    cov_mat = gen_cov_mat(cov_key, spec_x_in, spec_x_out, spec_mixed)
    sqrt_cov_mat = get_psd_mat_sqrt(cov_mat)
    
    # Create sampling function
    def sampling_fn_ill(key):
        return get_x_gen_cov(key, N_in=N_in, N_out=N_out, 
                          sqrt_cov_mat=sqrt_cov_mat, B=batch_size)
    
    sv_ill, sv_ill_norm = analyze_gradient_spectrum(
        N_in, N_out, batch_size, sampling_fn_ill, seed=seed,
        num_samples=num_gradient_samples,
        experiment_name="ill_conditioned_inputs",
        experiment_desc="Ill-Conditioned Inputs"
    )
    
    # --------------------------------
    # Learning Rate Sensitivity Experiment
    # --------------------------------
    # This is the same as the ill-conditioned experiment from a gradient perspective
    # So we don't need a separate analysis
    
    # --------------------------------
    # Experiment 4: Mixed Covariance Structure
    # --------------------------------
    print("Analyzing gradient spectrum for Experiment 4: Mixed Covariance Structure")
    
    # Create mixed covariance matrix
    mixed_key = random.PRNGKey(789)
    
    # Define eigenvalue spectra with structure
    spec_x_in = jnp.ones(N_in)  # Unit variance for individual components
    spec_x_out = jnp.ones(N_out)  # Unit variance for individual components
    
    # Add mixed term with some large eigenvalues
    spec_mixed = jnp.zeros(N_in + N_out)
    spec_mixed = spec_mixed.at[:5].set(jnp.array([5.0, 4.0, 3.0, 2.0, 1.0]))  # Some strong correlations
    
    # Generate covariance matrix
    mixed_cov_mat = gen_cov_mat(mixed_key, spec_x_in, spec_x_out, spec_mixed)
    mixed_sqrt_cov_mat = get_psd_mat_sqrt(mixed_cov_mat)
    
    # Create sampling function
    def sampling_fn_mixed(key):
        return get_x_gen_cov(key, N_in=N_in, N_out=N_out, 
                           sqrt_cov_mat=mixed_sqrt_cov_mat, B=batch_size)
    
    sv_mixed, sv_mixed_norm = analyze_gradient_spectrum(
        N_in, N_out, batch_size, sampling_fn_mixed, seed=seed,
        num_samples=num_gradient_samples,
        experiment_name="mixed_covariance",
        experiment_desc="Mixed Covariance Structure"
    )
    
    # Compare condition numbers across experiments
    experiments_data = [sv_iid, sv_corr, sv_ill, sv_mixed]
    experiment_names = ["iid_inputs", "correlated_inputs", "ill_conditioned_inputs", "mixed_covariance"]
    experiment_descs = ["IID Inputs", f"Correlated Inputs (c={correlation_strength})", 
                        "Ill-Conditioned Inputs", "Mixed Covariance Structure"]
    
    compare_condition_numbers(experiments_data, experiment_names, experiment_descs)
    
    print("\nGradient spectrum analysis completed. Results saved to 'plots/spectrum/' directory.")

if __name__ == "__main__":
    run_gradient_analysis()
