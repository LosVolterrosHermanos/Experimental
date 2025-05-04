#!/usr/bin/env python3
# Copyright 2025 Google LLC and Elliot Paquette

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
Comparing SignSVD and Standard SGD for Linear Regression.

This script compares the performance of SignSVD (an idealized version of Shampoo and Muon optimizers) 
with standard Stochastic Gradient Descent (SGD) on matrix-valued linear regression problems.

Background:
The SignSVD optimizer is a preconditioner that reshapes gradients to be better behaved. 
For a gradient G of a matrix-valued parameter M, the update rule is:
M_{t+1} = M_{t} - η G̃_{t}
where η is the learning rate, and
G̃ = U σ(S)V^T, G = USV^T

Here USV^T is the SVD of G, and σ(x) is the sign function. 
In other words, SignSVD replaces the gradient with a whitened version of the matrix.

From the Muon_experiment.md document:
"The muon optimizer is a preconditioner that reshapes gradients to, in some sense, be better behaved.
This will be about the idealization of muon called SignSVD."

The learning problem is a linear regression with matrix-valued parameters. Given:
- x_in ∈ R^N_in, a batch of input features
- x_out ∈ R^N_out, a batch of output features
- W ∈ R^(N_out × N_in), a matrix of parameters
- target W_hat, the ground truth parameter matrix

We model the output as:
y = x_out^T W x_in

And minimize the loss:
L(W) = (1/2) E[(f(x_in, x_out) - y(W, x_in, x_out))^2]

For the targets, we use a noiseless linear teacher:
f(x_in, x_out) = x_out^T W_hat x_in

The experiments in this script explore various input distributions and correlation structures
to understand when SignSVD might outperform standard SGD. We're particularly interested
in challenging optimization landscapes with ill-conditioning and complex correlation structures.
"""

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as random
from functools import partial
from typing import Optional, Sequence, Callable, Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os

# Create a directory for output plots if it doesn't exist
os.makedirs("plots", exist_ok=True)

# Model and loss functions
@jax.jit
def mse_loss(y_pred, y_target):
    """MSE loss from arrays of targets and predictions."""
    return 0.5 * jnp.mean(jnp.square(y_pred - y_target))

@jax.jit
def lin_mat_model(W, x_in, x_out):
    """Output of model for one input."""
    return jnp.dot(jnp.dot(x_out, W), x_in)

@jax.jit
def lin_mat_model_batched(W, x_in, x_out):
    """Outputs for batched inputs."""
    return jnp.einsum('ai,bi,ba->b', W, x_in, x_out)

def init_params(rng, N_in, N_out, num_models=None):
    """Initialize model parameters, such that random O(1) xs give O(1) outputs."""
    if num_models is None:
        M = random.normal(rng, (N_out, N_in))
    else:
        M = random.normal(rng, (num_models, N_out, N_in))
    M = M / jnp.sqrt(N_in*N_out)
    return M

# Gradient transformations
@jax.jit
def identity_transform(M):
    """Standard SGD - no transformation."""
    return M

@jax.jit
def whiten_mat(M):
    """SignSVD transformation - replace singular values with their signs."""
    U, S, VH = jnp.linalg.svd(M, full_matrices=False)
    white_S = jnp.sign(S)
    return jnp.einsum('...nk,...k,...km->...nm', U, white_S, VH)

# Input sampling functions
def get_x_iid(rng, N_in, N_out, B=None):
    """Sample x_in and x_out independently from standard normal distributions."""
    key1, key2 = random.split(rng)
    if B is None:
        x_in = random.normal(key1, (N_in,))
        x_out = random.normal(key2, (N_out,))
    else:
        x_in = random.normal(key1, (B, N_in))
        x_out = random.normal(key2, (B, N_out))
    return x_in, x_out

def get_x_corr(rng, N_in, N_out, proj_mat, cor, B=None):
    """Sample x_in and x_out with correlation parameter cor.
    
    x_out = cor * (proj_mat @ x_in) + sqrt(1-cor^2) * noise
    """
    key1, key2 = random.split(rng)
    if B is None:
        x_in = random.normal(key1, (N_in,))
        x_out_ind = random.normal(key2, (N_out,))
        x_out = cor * proj_mat @ x_in + jnp.sqrt(1-cor**2) * x_out_ind
    else:
        x_in = random.normal(key1, (B, N_in))
        x_out_ind = random.normal(key2, (B, N_out))
        x_out = cor * jnp.einsum('ai,bi->ba', proj_mat, x_in) + jnp.sqrt(1-cor**2) * x_out_ind
    return x_in, x_out

@jax.jit
def gen_cov_mat(rng, spec_x_in, spec_x_out, spec_mixed):
    """Generate covariance matrix for joint distribution of x_in and x_out."""
    N_in = len(spec_x_in)
    N_out = len(spec_x_out)
    N_total = len(spec_mixed)
    assert N_total == N_in + N_out
    
    # Generate random eigenvectors
    rng_in, rng_out, rng_total = random.split(rng, 3)
    U_in = random.orthogonal(rng_in, N_in)
    U_out = random.orthogonal(rng_out, N_out)
    U_total = random.orthogonal(rng_total, N_total)

    H_in = U_in @ jnp.diag(spec_x_in) @ U_in.T
    H_out = U_out @ jnp.diag(spec_x_out) @ U_out.T
    H_total = U_total @ jnp.diag(spec_mixed) @ U_total.T

    # Combine H_in and H_out into block diagonal matrix and add to H_total
    H_in_out = jnp.block([[H_in, jnp.zeros((N_in, N_out))], 
                          [jnp.zeros((N_out, N_in)), H_out]])
    H_total = H_total + H_in_out

    return H_total

@jax.jit
def get_psd_mat_sqrt(M):
    """Get square root of positive semidefinite matrix."""
    s, V = jnp.linalg.eigh(M)
    return V @ jnp.diag(jnp.sqrt(jnp.maximum(s, 0))) @ V.T

def get_x_gen_cov(rng, N_in, N_out, sqrt_cov_mat, B=None):
    """Sample x_in and x_out from general joint covariance matrix."""
    if B is None:
        x_tot = random.normal(rng, (N_in+N_out,))
        x_tot = sqrt_cov_mat @ x_tot
    else:
        x_tot = random.normal(rng, (B, N_in+N_out))
        x_tot = jnp.einsum('ai,bi->ba', sqrt_cov_mat, x_tot)
    return x_tot[..., 0:N_in], x_tot[..., N_in:]

# Training functions
@jax.jit
def compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W):
    """Compute loss and gradient for a batch of samples.
    
    Based on the formulation in sign_svd_linear_regression.ipynb, the stochastic gradient
    in this setup is given by:
    
    G = (1/B)(Z X_out)^T X_in
    
    where:
    - X_in is the B×N_in dimensional matrix of x_in for the batch
    - X_out is defined analogously for x_out (B×N_out)
    - Z is the B×B dimensional diagonal matrix of residuals 
      (y(W, x_in, x_out) - f(x_in, x_out)) over the batch
    - B is the batch size
    
    This structure is similar to gradients in fully connected neural networks and
    helps illustrate why the distribution of inputs matters for optimization behavior.
    When X_in and X_out have structured correlation or ill-conditioning, this directly
    affects the singular value spectrum of the gradient, potentially creating 
    challenges for vanilla SGD.
    """
    # Compute model predictions
    y_pred = lin_mat_model_batched(W, x_in_batch, x_out_batch)
    
    # Compute target values
    y_target = lin_mat_model_batched(target_W, x_in_batch, x_out_batch)
    
    # Compute loss
    loss = mse_loss(y_pred, y_target)
    
    # Compute gradient analytically for efficiency
    residuals = y_pred - y_target
    # Gradient is outer product of x_out and x_in, weighted by residuals
    grad = jnp.einsum('b,bn,bm->nm', residuals, x_out_batch, x_in_batch) / len(residuals)
    
    return loss, grad

@jax.jit
def update_step(W, x_in_batch, x_out_batch, target_W, lr, grad_transform):
    """Update parameters using specified gradient transformation."""
    _, grad = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
    transformed_grad = grad_transform(grad)
    return W - lr * transformed_grad

@jax.jit
def compute_test_loss_single(W, target_W, x_in_test, x_out_test):
    """Compute test loss using given samples."""
    y_pred = lin_mat_model_batched(W, x_in_test, x_out_test)
    y_target = lin_mat_model_batched(target_W, x_in_test, x_out_test)
    return mse_loss(y_pred, y_target)

def compute_test_loss(W, target_W, test_key, N_in, N_out, num_test_samples=1000, 
                      sampling_fn=None):
    """Compute test loss using many samples."""
    if sampling_fn is None:
        # Default to i.i.d. sampling for test loss
        x_in_test, x_out_test = get_x_iid(test_key, N_in=N_in, N_out=N_out, B=num_test_samples)
    else:
        # For consistency, use the same sampling function as training
        # But ensure we use a large number of samples for a good estimate
        new_key1, new_key2 = random.split(test_key)
        x_in_1, x_out_1 = sampling_fn(new_key1)
        batch_size = x_in_1.shape[0]
        
        if num_test_samples > batch_size:
            # We need additional samples
            samples_needed = num_test_samples - batch_size
            batches_needed = (samples_needed + batch_size - 1) // batch_size
            test_samples = [sampling_fn(random.fold_in(new_key2, i)) for i in range(batches_needed)]
            x_in_extra = jnp.concatenate([x[0] for x in test_samples], axis=0)
            x_out_extra = jnp.concatenate([x[1] for x in test_samples], axis=0)
            # Concatenate all samples
            x_in_test = jnp.concatenate([x_in_1, x_in_extra[:samples_needed]], axis=0)
            x_out_test = jnp.concatenate([x_out_1, x_out_extra[:samples_needed]], axis=0)
        else:
            # Just use what we have
            x_in_test, x_out_test = x_in_1, x_out_1
    
    return compute_test_loss_single(W, target_W, x_in_test, x_out_test)

# Create jitted update functions for both gradient transforms
@jax.jit
def sgd_update_step(W, x_in_batch, x_out_batch, target_W, lr):
    """Update parameters using standard SGD."""
    _, grad = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
    return W - lr * grad

@jax.jit
def signsvd_update_step(W, x_in_batch, x_out_batch, target_W, lr):
    """Update parameters using SignSVD."""
    _, grad = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
    white_grad = whiten_mat(grad)
    return W - lr * white_grad

@jax.jit
def training_step_sgd(W, target_W, x_in_batch, x_out_batch, x_in_test, x_out_test, lr):
    """Single step of training with standard SGD."""
    # Compute training loss
    train_loss, _ = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
    
    # Update parameters
    new_W = sgd_update_step(W, x_in_batch, x_out_batch, target_W, lr)
    
    # Compute test loss
    test_loss = compute_test_loss_single(new_W, target_W, x_in_test, x_out_test)
    
    return new_W, train_loss, test_loss

@jax.jit
def training_step_signsvd(W, target_W, x_in_batch, x_out_batch, x_in_test, x_out_test, lr):
    """Single step of training with SignSVD."""
    # Compute training loss
    train_loss, _ = compute_loss_and_grad(W, x_in_batch, x_out_batch, target_W)
    
    # Update parameters
    new_W = signsvd_update_step(W, x_in_batch, x_out_batch, target_W, lr)
    
    # Compute test loss
    test_loss = compute_test_loss_single(new_W, target_W, x_in_test, x_out_test)
    
    return new_W, train_loss, test_loss

def train(N_in, N_out, batch_size, num_steps, learning_rate, grad_transform, 
         sampling_fn, seed=0):
    """Train a model using the specified gradient transformation and sampling function."""
    # Initialize keys
    main_key = random.PRNGKey(seed)
    init_key, target_key, test_key, train_key = random.split(main_key, 4)
    
    # Initialize parameters
    W = init_params(init_key, N_in, N_out)
    target_W = init_params(target_key, N_in, N_out)
    
    # Generate test data once for consistent test loss measurement
    x_in_test, x_out_test = get_x_iid(test_key, N_in=N_in, N_out=N_out, B=1000)
    
    # Compute initial test loss
    initial_test_loss = compute_test_loss_single(W, target_W, x_in_test, x_out_test)
    
    # Pre-generate all batch data for training
    batch_keys = random.split(train_key, num_steps)
    x_in_batches = jnp.array([sampling_fn(key)[0] for key in batch_keys])
    x_out_batches = jnp.array([sampling_fn(key)[1] for key in batch_keys])
    
    # Initialize array for storing results
    train_losses = jnp.zeros(num_steps)
    test_losses = jnp.zeros(num_steps + 1)
    test_losses = test_losses.at[0].set(initial_test_loss)
    
    # Choose the appropriate training step function
    if grad_transform == identity_transform:
        training_step_fn = training_step_sgd
    elif grad_transform == whiten_mat:
        training_step_fn = training_step_signsvd
    else:
        raise ValueError("Unsupported gradient transformation")
    
    # Run training steps
    # Define a function for a single training step to use with scan
    def scan_step(W_carry, step_data):
        x_in_batch, x_out_batch = step_data
        
        # Run one step
        W_new, train_loss, test_loss = training_step_fn(
            W_carry, target_W, x_in_batch, x_out_batch, x_in_test, x_out_test, learning_rate
        )
        
        # Return the new state and the values to store
        return W_new, (train_loss, test_loss)
    
    # Prepare the input data for scan
    scan_data = (x_in_batches, x_out_batches)
    
    # Run the training loop using jax.scan
    W_final, (train_losses, test_losses_steps) = jax.lax.scan(
        scan_step, W, jax.tree.map(lambda x: jnp.array(x), scan_data)
    )
    
    # Update test_losses array (need to handle the initial test loss separately)
    test_losses = jnp.concatenate([jnp.array([initial_test_loss]), test_losses_steps])
    
    return train_losses, test_losses, W


def run_experiments():
    """Run all experiments comparing SignSVD and standard SGD.
    
    This function implements multiple experiments to compare the performance of SignSVD
    (an idealized version of the Muon optimizer) against standard SGD in different
    scenarios with matrix-valued parameters.
    
    Each experiment explores a different aspect of the optimization landscape:
    1. I.I.D. Inputs: Standard setup with independent inputs
    2. Correlated Inputs: Inputs with structured correlation
    3. Ill-Conditioned Input Covariance: Challenging optimization with poor conditioning
    4. Learning Rate Sensitivity: Comparing robustness to learning rate choice
    5. Mixed Covariance Structure: Complex correlations between input dimensions
    
    The experiments are designed to highlight scenarios where SignSVD might provide
    advantages over standard SGD.
    """
    
    # Common parameters
    N_in = 200
    N_out = 200
    batch_size = 200
    num_steps = 1000
    learning_rate_sgd = 0.001
    learning_rate_signsvd = 0.001
    seed = 42
    
    # --------------------------------
    # Experiment 1: I.I.D. Inputs
    # --------------------------------
    """
    In this experiment, we consider the basic setup where both x_in and x_out are
    sampled independently from standard normal distributions. This creates a relatively
    well-behaved optimization landscape.
    
    The model is a matrix-valued linear regression:
    y = x_out^T W x_in
    
    where:
    - y is a scalar output
    - W is an N_out x N_in parameter matrix
    - x_in and x_out are drawn independently from N(0, I)
    
    This baseline experiment helps establish performance in a standard setting where
    we don't expect large differences between optimizers. Since the inputs have no
    structured correlation, the gradient shouldn't have a highly skewed spectrum of
    singular values.
    """
    print("Experiment 1: I.I.D. Inputs")
    
    # Create sampling function for IID inputs
    def sampling_fn_iid(key):
        return get_x_iid(key, N_in=N_in, N_out=N_out, B=batch_size)
    
    # Train models
    print("Training with standard SGD...")
    sgd_train_losses, sgd_test_losses, sgd_W = train(
        N_in, N_out, batch_size, num_steps, learning_rate_sgd, identity_transform, 
        sampling_fn_iid, seed=seed)
    
    print("Training with SignSVD...")
    signsvd_train_losses, signsvd_test_losses, signsvd_W = train(
        N_in, N_out, batch_size, num_steps, learning_rate_signsvd, whiten_mat, 
        sampling_fn_iid, seed=seed)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sgd_train_losses, label='SGD')
    plt.plot(signsvd_train_losses, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss - IID Inputs')
    
    plt.subplot(1, 2, 2)
    plt.plot(sgd_test_losses, label='SGD')
    plt.plot(signsvd_test_losses, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Test Loss - IID Inputs')
    
    plt.tight_layout()
    plt.savefig("plots/exp1_iid_inputs.pdf")
    plt.close()
    
    # --------------------------------
    # Experiment 2: Correlated Inputs
    # --------------------------------
    """
    In this experiment, we introduce correlation between x_in and x_out. We generate
    x_out as a function of x_in plus some independent noise:
    
    x_out = c * (proj_mat @ x_in) + sqrt(1-c^2) * noise
    
    where:
    - c is the correlation strength parameter (between 0 and 1)
    - proj_mat is a random projection matrix
    - noise is sampled from N(0, I)
    
    From the sign_svd_linear_regression notebook:
    '(x_out)_α = cT_αi (x_in)_i + sqrt(1-c^2)Z_α
    where T_αi is an embedding/projection matrix, and Z_α are i.i.d. N(0, 1).
    Here c∈[0, 1] parameterizes the correlation.'
    
    This correlation creates a more structured optimization landscape that could
    benefit from the whitening effect of SignSVD. As the gradient is computed using
    outer products of x_in and x_out, correlation between them can lead to gradients
    with more extreme singular values.
    
    We use a high correlation parameter (1.0-(1.0/N_in)) to make the effect more pronounced.
    """
    print("\nExperiment 2: Correlated Inputs")
    
    # Create correlation matrix
    corr_key = random.PRNGKey(123)
    proj_mat = random.normal(corr_key, (N_out, N_in)) / jnp.sqrt(N_in)
    correlation_strength = 1.0-(1.0/N_in)  # High correlation
    
    # Create sampling function
    def sampling_fn_corr(key):
        return get_x_corr(key, N_in=N_in, N_out=N_out, 
                        proj_mat=proj_mat, cor=correlation_strength, B=batch_size)
    
    # Train models
    print("Training with standard SGD...")
    sgd_train_losses_corr, sgd_test_losses_corr, sgd_W_corr = train(
        N_in, N_out, batch_size, num_steps, learning_rate_sgd, identity_transform, 
        sampling_fn_corr, seed=seed)
    
    print("Training with SignSVD...")
    signsvd_train_losses_corr, signsvd_test_losses_corr, signsvd_W_corr = train(
        N_in, N_out, batch_size, num_steps, learning_rate_signsvd, whiten_mat, 
        sampling_fn_corr, seed=seed)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sgd_train_losses_corr, label='SGD')
    plt.plot(signsvd_train_losses_corr, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Training Loss - Correlated Inputs (c={correlation_strength})')
    
    plt.subplot(1, 2, 2)
    plt.plot(sgd_test_losses_corr, label='SGD')
    plt.plot(signsvd_test_losses_corr, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.title(f'Test Loss - Correlated Inputs (c={correlation_strength})')
    
    plt.tight_layout()
    plt.savefig("plots/exp2_correlated_inputs.pdf")
    plt.close()
    
    # --------------------------------
    # Experiment 3: Ill-Conditioned Input Covariance
    # --------------------------------
    """
    In this experiment, we create inputs with highly ill-conditioned covariance matrices.
    The condition number of a matrix is the ratio of its largest to smallest eigenvalue,
    and a high condition number generally leads to more challenging optimization problems.
    
    We generate x_in and x_out to have power-law spectra where eigenvalues decay as 1/i^alpha.
    
    From the sign_svd_linear_regression notebook, we're constructing a joint covariance 
    matrix Σ of this form:
    
    Σ = [U_in S_in U_in^T    0              ] + U_mixed S_mixed U_mixed^T
        [0                  U_out S_out U_out^T]
    
    where:
    - S_in, S_out are diagonal matrices with eigenvalues that span 3 orders of magnitude
    - U_in, U_out, U_mixed are random orthogonal matrices
    - S_mixed is set to zero in this experiment (we'll add mixed terms in Experiment 5)
    
    This creates challenging gradients with highly varied singular values, where
    standard SGD might struggle due to being pulled too strongly in directions 
    corresponding to the largest singular values. SignSVD should handle this better
    by normalizing all directions.
    """
    print("\nExperiment 3: Ill-Conditioned Input Covariance")
    
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
    
    # Train models
    print("Training with standard SGD...")
    sgd_train_losses_ill, sgd_test_losses_ill, sgd_W_ill = train(
        N_in, N_out, batch_size, num_steps, learning_rate_sgd, identity_transform, 
        sampling_fn_ill, seed=seed)
    
    print("Training with SignSVD...")
    signsvd_train_losses_ill, signsvd_test_losses_ill, signsvd_W_ill = train(
        N_in, N_out, batch_size, num_steps, learning_rate_signsvd, whiten_mat, 
        sampling_fn_ill, seed=seed)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sgd_train_losses_ill, label='SGD')
    plt.plot(signsvd_train_losses_ill, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss - Ill-Conditioned Inputs')
    
    plt.subplot(1, 2, 2)
    plt.plot(sgd_test_losses_ill, label='SGD')
    plt.plot(signsvd_test_losses_ill, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Test Loss - Ill-Conditioned Inputs')
    
    plt.tight_layout()
    plt.savefig("plots/exp3_ill_conditioned_inputs.pdf")
    plt.close()
    
    # --------------------------------
    # Experiment 4: Learning Rate Sensitivity
    # --------------------------------
    """
    In this experiment, we examine how sensitive each optimizer is to the choice of
    learning rate. One potential advantage of SignSVD is improved robustness to
    learning rate selection.
    
    We run the ill-conditioned input experiment (Experiment 3) with various learning
    rates spanning several orders of magnitude (0.001 to 0.3) and compare the final
    test losses.
    
    In practical machine learning, choosing an appropriate learning rate is often
    challenging, and optimizers that work well across a wider range of learning rates
    can be more convenient to use. By normalizing gradient directions in SignSVD,
    we expect to see more consistent performance across different learning rates
    compared to standard SGD.
    
    We use a shorter training run (300 steps instead of 1000) for efficiency since
    we're running multiple training instances.
    """
    print("\nExperiment 4: Learning Rate Sensitivity")
    
    # Shorter training for this experiment
    lr_num_steps = 300
    learning_rates = [0.001, 0.003, 0.005, 0.007, 0.009]
    
    # Use ill-conditioned setup to highlight differences
    sgd_final_losses = []
    signsvd_final_losses = []
    
    for lr in learning_rates:
        print(f"Training with learning rate {lr}...")
        
        # Train with SGD
        _, sgd_test_losses_lr, _ = train(
            N_in, N_out, batch_size, lr_num_steps, lr, identity_transform, 
            sampling_fn_ill, seed=seed)
        
        # Train with SignSVD
        _, signsvd_test_losses_lr, _ = train(
            N_in, N_out, batch_size, lr_num_steps, lr, whiten_mat, 
            sampling_fn_ill, seed=seed)
        
        # Record final losses
        sgd_final_losses.append(sgd_test_losses_lr[-1])
        signsvd_final_losses.append(signsvd_test_losses_lr[-1])
    
    # Plot learning rate sensitivity
    plt.figure(figsize=(10, 6))
    plt.plot(learning_rates, sgd_final_losses, 'o-', label='SGD')
    plt.plot(learning_rates, signsvd_final_losses, 's-', label='SignSVD')
    plt.xlabel('Learning Rate')
    plt.ylabel('Final Test Loss')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Learning Rate Sensitivity')
    plt.grid(True, which='both', ls='--', alpha=0.3)
    plt.savefig("plots/exp4_learning_rate_sensitivity.pdf")
    plt.close()
    
    # --------------------------------
    # Experiment 5: Mixed Covariance Structure
    # --------------------------------
    """
    In this experiment, we create a more complex covariance structure with strong
    correlations between components of x_in and x_out. This is an extension of
    Experiment 3, but now we add non-zero mixed terms to create correlation between
    the input and output features.
    
    From the sign_svd_linear_regression notebook, we construct a joint covariance
    matrix where both the diagonal blocks (controlling variance of individual components)
    and the mixed terms (controlling correlation between x_in and x_out) are significant.
    
    We set the individual input and output variances to 1, but add strong mixed terms
    with eigenvalues [5.0, 4.0, 3.0, 2.0, 1.0] for the first 5 components. This creates
    a rich correlation structure where some modes of x_in are highly informative about
    specific modes of x_out.
    
    In practice, such correlation structures can arise in real data where input and
    output variables have complex relationships. These structures can create challenging
    optimization landscapes where the gradient's singular value spectrum is highly
    non-uniform, potentially showing a larger performance gap between SignSVD and SGD.
    """
    print("\nExperiment 5: Mixed Covariance Structure")
    
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
    
    # Train models
    print("Training with standard SGD...")
    sgd_train_losses_mixed, sgd_test_losses_mixed, sgd_W_mixed = train(
        N_in, N_out, batch_size, num_steps, learning_rate_sgd, identity_transform, 
        sampling_fn_mixed, seed=seed)
    
    print("Training with SignSVD...")
    signsvd_train_losses_mixed, signsvd_test_losses_mixed, signsvd_W_mixed = train(
        N_in, N_out, batch_size, num_steps, learning_rate_signsvd, whiten_mat, 
        sampling_fn_mixed, seed=seed)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sgd_train_losses_mixed, label='SGD')
    plt.plot(signsvd_train_losses_mixed, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Training Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Loss - Mixed Covariance Structure')
    
    plt.subplot(1, 2, 2)
    plt.plot(sgd_test_losses_mixed, label='SGD')
    plt.plot(signsvd_test_losses_mixed, label='SignSVD')
    plt.xlabel('Step')
    plt.ylabel('Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('Test Loss - Mixed Covariance Structure')
    
    plt.tight_layout()
    plt.savefig("plots/exp5_mixed_covariance.pdf")
    plt.close()
    
    print("\nAll experiments completed. Results saved to 'plots/' directory.")
    
    """
    Summary of Findings:
    
    The experiments in this script explore scenarios where SignSVD (an idealized version
    of the Muon optimizer) might outperform standard SGD. We should expect to see:
    
    1. Similar performance in well-behaved settings (Experiment 1 with IID inputs)
    
    2. Potential advantages for SignSVD in:
       - Correlated inputs (Experiment 2)
       - Ill-conditioned problems (Experiment 3)
       - Robustness to learning rate choice (Experiment 4)
       - Complex correlation structures (Experiment 5)
    
    The SignSVD approach normalizes all directions in the gradient, which can help avoid
    disproportionate influence from a few dominant directions. This should be particularly
    helpful in scenarios where the gradient's singular value spectrum is highly non-uniform.
    
    In real-world machine learning contexts, such challenging optimization landscapes are
    common, suggesting potential practical benefits for preconditioned optimizers like Muon.
    """

if __name__ == "__main__":
    run_experiments()
