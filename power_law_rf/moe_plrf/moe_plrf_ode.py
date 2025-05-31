import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Union, Optional, Literal
from power_law_rf.ode import DanaHparams


class MoEODEInputs(NamedTuple):
    eigs_K: jnp.ndarray          # eigenvalues of covariance matrix (W^TDW)
    rho_init: jnp.ndarray        # initial rho_j's (rho_j^2)
    chi_init: jnp.ndarray        # initialization of chi's
    sigma_init: jnp.ndarray      # initialization of sigma's (xi^2_j)
    risk_infinity: float         # risk value at time infinity
    expert_probs: jnp.ndarray    # expert selection probabilities p(i), shape (m,)


def ode_moe_dana_log_implicit(
    inputs: MoEODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    m: int,  # number of experts
    t_max: float,
    dt: float,
    approximate: bool = False,
    adaptive: Optional[Literal['adam', 'rmsprop_dana']] = None,
):
    """Generate the theoretical solution for MoE PLRF with DANA.
    Outputs TWICE the risk. Vectorized version processes all experts simultaneously.
    Assumes the loss is AVERAGED over the batch, not summed.

    Parameters
    ----------
    inputs : MoEODEInputs
        eigs_K : array d
            eigenvalues of covariance matrix (W^TDW)
        rho_init : array d
            initial rho_j's (rho_j^2)
        chi_init : array (d)
            initialization of chi's
        sigma_init : array (d)
            initialization of sigma's (xi^2_j)
        risk_infinity : scalar
            represents the risk value at time infinity
        expert_probs : array (m)
            expert selection probabilities p(i)

    opt_hparams : optimizer hyperparameters for Dana
        g1, g2, g3 : function(time)
            learning rate functions
        delta : function(time)
            momentum function

    batch : int
        batch size
    D : int
        number of eigenvalues (i.e. shape of eigs_K)
    m : int
        number of experts
    t_max : float
        The number of epochs
    dt : float
        time step used in Euler
    approximate : bool
        Whether to use the approximate ODE (drops higher order terms)
    adaptive : Optional[Literal['adam', 'rmsprop_dana']]
        Type of adaptive optimizer normalization:
        - None: no normalization
        - 'adam': normalize g3 terms (momentum entering parameters)
        - 'rmsprop_dana': normalize g1 terms (gradients entering momentum)

    Returns
    -------
    t_grid: numpy.array(float)
        the time steps used, which will discretize (0,t_max) into n_grid points
    expert_twice_risks: numpy.array(float)
        twice the values of the risk for each expert, shape (n_times, m)
    total_twice_risks: numpy.array(float)
        twice the total MoE risk, shape (n_times,)
    """
    g1_fn, g2_fn, g3_fn, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
    eigs_K = inputs.eigs_K
    expert_probs = inputs.expert_probs  # shape (m,)
    
    # Initialize m copies of the state variables
    # Shape: (3, D, m) where 3 is for [rho, sigma, chi]
    rho_init_all = jnp.tile(inputs.rho_init[:, None], (1, m))
    chi_init_all = jnp.tile(inputs.chi_init[:, None], (1, m))
    sigma_init_all = jnp.tile(inputs.sigma_init[:, None], (1, m))
    
    twice_risk_infinity = 2.0 * inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    
    # Initial risks for each expert
    risk_init_all = twice_risk_infinity + jnp.sum(inputs.eigs_K[:, None] * rho_init_all, axis=0)

    def get_normalization_factors(grad_norm):
        """Compute normalization factors based on adaptive optimizer type."""
        if adaptive == 'adam':
            return 1.0, grad_norm
        elif adaptive == 'rmsprop_dana':
            return grad_norm, 1.0
        else:
            return 1.0, 1.0

    def inverse_3x3(omega):
        """Vectorized inverse for shape (m, 3, 3, D) -> (m, 3, 3, D)"""
        # Extract elements - shape (m, D)
        a11, a12, a13 = omega[:, 0, 0], omega[:, 0, 1], omega[:, 0, 2]
        a21, a22, a23 = omega[:, 1, 0], omega[:, 1, 1], omega[:, 1, 2]
        a31, a32, a33 = omega[:, 2, 0], omega[:, 2, 1], omega[:, 2, 2]

        det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
               - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

        # Build inverse matrix
        inv = jnp.zeros_like(omega)
        
        inv = inv.at[:, 0, 0].set((a22*a33 - a23*a32) / det)
        inv = inv.at[:, 0, 1].set((a13*a32 - a12*a33) / det)
        inv = inv.at[:, 0, 2].set((a12*a23 - a13*a22) / det)
        
        inv = inv.at[:, 1, 0].set((a23*a31 - a21*a33) / det)
        inv = inv.at[:, 1, 1].set((a11*a33 - a13*a31) / det)
        inv = inv.at[:, 1, 2].set((a13*a21 - a11*a23) / det)
        
        inv = inv.at[:, 2, 0].set((a21*a32 - a22*a31) / det)
        inv = inv.at[:, 2, 1].set((a12*a31 - a11*a32) / det)
        inv = inv.at[:, 2, 2].set((a11*a22 - a12*a21) / det)
        
        return inv

    def omega_full_all(time_plus, grad_norm_all):
        """Compute omega for all experts at once - shape (m, 3, 3, D)"""
        # Get normalization factors for each expert
        g1_norm_all, g3_norm_all = get_normalization_factors(grad_norm_all)
        
        # Normalize hyperparameters
        g1 = g1_fn(time_plus) / g1_norm_all[:, None]  # shape (m, 1)
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm_all[:, None]  # shape (m, 1)
        delta = delta_fn(time_plus)
        
        # Broadcast expert_probs to shape (m, 1) for easier operations
        p = expert_probs[:, None]  # shape (m, 1)
        
        # Broadcast eigs_K to shape (1, D) -> operations will broadcast to (m, D)
        eigs = eigs_K[None, :]  # shape (1, D)
        
        omega_all = jnp.zeros((m, 3, 3, D))
        
        # Row 1: Evolution of rho
        omega_all = omega_all.at[:, 0, 0].set(
            -2.0 * p * (g2 + g1 * g3) * eigs + 
            p * ((batch + 1.0) / batch) * (g2**2 + 2.0 * g1 * g3 * g2 + g1**2 * g3**2) * eigs**2
        )
        omega_all = omega_all.at[:, 0, 1].set(
            g3**2 * (1.0 - delta)**2 * jnp.ones((m, D))
        )
        omega_all = omega_all.at[:, 0, 2].set(
            -2.0 * g3 * (1.0 - delta) + 
            2.0 * p * (g2 * g3 + g3**2 * g1) * (1.0 - delta) * eigs
        )
        
        # Row 2: Evolution of sigma
        omega_all = omega_all.at[:, 1, 0].set(
            p * ((batch + 1.0) / batch) * g1**2 * eigs**2
        )
        omega_all = omega_all.at[:, 1, 1].set(
            (-2.0 * delta + delta**2) * jnp.ones((m, D))
        )
        omega_all = omega_all.at[:, 1, 2].set(
            2.0 * p * g1 * eigs * (1.0 - delta)
        )
        
        # Row 3: Evolution of chi
        omega_all = omega_all.at[:, 2, 0].set(
            p * g1 * eigs - p * ((batch + 1.0) / batch) * eigs**2 * (g1 * g2 + g1**2 * g3)
        )
        omega_all = omega_all.at[:, 2, 1].set(
            (-g3 + g3 * delta * (2.0 - delta)) * jnp.ones((m, D))
        )
        omega_all = omega_all.at[:, 2, 2].set(
            -delta - p * (g2 - g2 * delta + 2.0 * (1.0 - delta) * g1 * g3) * eigs
        )
        
        return omega_all

    def omega_approximate_all(time_plus, grad_norm_all):
        """Compute approximate omega for all experts - shape (m, 3, 3, D)"""
        # Get normalization factors for each expert
        g1_norm_all, g3_norm_all = get_normalization_factors(grad_norm_all)
        
        # Normalize hyperparameters
        g1 = g1_fn(time_plus) / g1_norm_all[:, None]
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm_all[:, None]
        delta = delta_fn(time_plus)
        
        p = expert_probs[:, None]
        eigs = eigs_K[None, :]
        
        omega_all = jnp.zeros((m, 3, 3, D))
        
        omega_all = omega_all.at[:, 0, 0].set(-2.0 * p * g2 * eigs)
        omega_all = omega_all.at[:, 0, 1].set(jnp.zeros((m, D)))
        omega_all = omega_all.at[:, 0, 2].set(-2.0 * g3 * jnp.ones((m, D)))
        
        omega_all = omega_all.at[:, 1, 0].set(jnp.zeros((m, D)))
        omega_all = omega_all.at[:, 1, 1].set(-2.0 * delta * jnp.ones((m, D)))
        omega_all = omega_all.at[:, 1, 2].set(2.0 * p * g1 * eigs)
        
        omega_all = omega_all.at[:, 2, 0].set(p * g1 * eigs)
        omega_all = omega_all.at[:, 2, 1].set(-g3 * jnp.ones((m, D)))
        omega_all = omega_all.at[:, 2, 2].set(-delta - p * g2 * eigs)
        
        return omega_all

    def forcing_term_all(time_plus, grad_norm_all):
        """Compute forcing term for all experts - shape (m, 3, D)"""
        # Get normalization factors for each expert
        g1_norm_all, g3_norm_all = get_normalization_factors(grad_norm_all)
        
        # Normalize hyperparameters
        g1 = g1_fn(time_plus) / g1_norm_all[:, None]
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm_all[:, None]
        
        p = expert_probs[:, None]  # shape (m, 1)
        
        gamma_all = jnp.stack([
            p * (g2**2 + 2.0 * g1 * g2 * g3 + g1**2 * g3**2) / batch,
            p * g1**2 / batch,
            p * (-g1 * g2 - g1**2 * g3) / batch
        ], axis=1)  # shape (m, 3)
        
        # Broadcast with eigs_K: (m, 3, 1) * (1, 1, D) -> (m, 3, D)
        return gamma_all[:, :, None] * eigs_K[None, None, :]

    def forcing_term_approximate_all(time_plus, grad_norm_all):
        """Compute approximate forcing term for all experts - shape (m, 3, D)"""
        # Get normalization factors for each expert
        g1_norm_all, g3_norm_all = get_normalization_factors(grad_norm_all)
        
        # Normalize hyperparameters
        g1 = g1_fn(time_plus) / g1_norm_all[:, None]
        g2 = g2_fn(time_plus)
        
        p = expert_probs[:, None]
        
        gamma_all = jnp.stack([
            p * g2**2 / batch,
            p * g1**2 / batch,
            jnp.zeros_like(p)
        ], axis=1)  # shape (m, 3)
        
        return gamma_all[:, :, None] * eigs_K[None, None, :]

    def ode_update(carry, time):
        v_all, twice_risk_all = carry  # v_all: (3, D, m), twice_risk_all: (m,)
        time_plus = jnp.exp(time + dt)
        time_plus_minus_one = time_plus - 1.0
        
        # Transpose v_all to (m, 3, D) for easier batch processing
        v_all_transposed = jnp.transpose(v_all, (2, 0, 1))  # (m, 3, D)
        
        # Use sqrt(risk) as proxy for gradient norm when adaptive is not None
        grad_norm_all = jnp.sqrt(twice_risk_all / 2.0) if adaptive is not None else jnp.ones(m)
        
        # Get omega for all experts - shape (m, 3, 3, D)
        omega_all = omega_approximate_all(time_plus_minus_one, grad_norm_all) if approximate else omega_full_all(time_plus_minus_one, grad_norm_all)
        
        # Create identity matrix broadcasted for all experts
        identity_all = jnp.eye(3)[None, :, :, None] * jnp.ones((m, 1, 1, D))  # (m, 3, 3, D)
        
        # Compute A for all experts
        A_all = inverse_3x3(identity_all - (dt * time_plus) * omega_all)  # (m, 3, 3, D)
        
        # Get forcing term for all experts - shape (m, 3, D)
        G_lambda_all = forcing_term_approximate_all(time_plus_minus_one, grad_norm_all) if approximate else forcing_term_all(time_plus_minus_one, grad_norm_all)
        
        # Compute x_temp for all experts - shape (m, 3, D)
        x_temp_all = v_all_transposed + dt * time_plus * twice_risk_infinity * G_lambda_all
        
        # Apply A to x_temp: (m, 3, 3, D) @ (m, 3, D) -> (m, 3, D)
        x_all = jnp.einsum('mijk,mjk->mik', A_all, x_temp_all)
        
        # Apply A to G_lambda
        y_all = jnp.einsum('mijk,mjk->mik', A_all, G_lambda_all)
        
        # Compute the z terms - shape (m, D)
        z_dot_x = jnp.sum(x_all[:, 0, :] * eigs_K[None, :], axis=1)  # (m,)
        z_dot_y = jnp.sum(y_all[:, 0, :] * eigs_K[None, :], axis=1)  # (m,)
        
        # Final update for all experts
        scale = (dt * time_plus * z_dot_x / (1.0 - dt * time_plus * z_dot_y))[:, None, None]  # (m, 1, 1)
        v_new_all_transposed = x_all + scale * y_all  # (m, 3, D)
        
        # Transpose back to (3, D, m)
        v_new_all = jnp.transpose(v_new_all_transposed, (1, 2, 0))
        
        # Compute new risks for all experts
        twice_risk_new_all = twice_risk_infinity + jnp.sum(eigs_K[:, None] * v_new_all[0, :, :], axis=0)  # (m,)
        
        # Total MoE risk
        total_twice_risk = jnp.sum(expert_probs * twice_risk_new_all)
        
        return (v_new_all, twice_risk_new_all), (twice_risk_new_all, total_twice_risk)

    init_v_all = jnp.stack([rho_init_all, sigma_init_all, chi_init_all], axis=0)  # (3, D, m)
    init_carry = (init_v_all, risk_init_all)
    
    _, (expert_twice_risks, total_twice_risks) = jax.lax.scan(ode_update, init_carry, times)
    
    return jnp.exp(times)-1.0, expert_twice_risks, total_twice_risks
