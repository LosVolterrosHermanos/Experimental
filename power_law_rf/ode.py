import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Union, Optional, Literal


class ODEInputs(NamedTuple):
    eigs_K: jnp.ndarray      # eigenvalues of covariance matrix (W^TDW)
    rho_init: jnp.ndarray    # initial rho_j's (rho_j^2)
    chi_init: jnp.ndarray    # initialization of chi's
    sigma_init: jnp.ndarray  # initialization of sigma's (xi^2_j)
    risk_infinity: float     # risk value at time infinity


class DanaHparams(NamedTuple):
    g1: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g2: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    g3: Callable[[Union[float, jnp.ndarray]], float]  # learning rate function
    delta: Callable[[Union[float, jnp.ndarray]], float]  # momentum function


def ode_resolvent_log_implicit(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float,
    approximate: bool = False,
    adaptive: Optional[Literal['adam', 'rmsprop_dana']] = None,
):
    """Generate the theoretical solution to momentum.
    Outputs TWICE the risk. Full ODE does NOT use coin-flip momentum.
    Approximate ODE for non-coin-flip and coin-flip are the same after dropping higher-order terms.
    Assumes the loss is AVERAGED over the batch, not summed.

    Parameters
    ----------
    inputs : ODEInputs
        eigs_K : array d
            eigenvalues of covariance matrix (W^TDW)
        rho_init : array d
            initial rho_j's (rho_j^2)
        chi_init : array (d)
            initialization of chi's
        sigma_init : array (d)
            initialization of sigma's (xi^2_j)
        risk_infinity : scalar
            represents the risk value at time infinity (note:
            this is NOT twice the risk)

    opt_hparams : optimizer hyperparameters for Dana
        g1, g2, g3 : function(time)
            learning rate functions
        delta : function(time)
            momentum function

    batch : int
        batch size
    D : int
        number of eigenvalues (i.e. shape of eigs_K)
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
    twice_risks: numpy.array(float)
        twice the values of the risk, as used in the paper.
    """
    g1_fn, g2_fn, g3_fn, delta_fn = opt_hparams.g1, opt_hparams.g2, opt_hparams.g3, opt_hparams.delta
    eigs_K = inputs.eigs_K
    rho_init, chi_init, sigma_init = inputs.rho_init, inputs.chi_init, inputs.sigma_init
    twice_risk_infinity = 2.0*inputs.risk_infinity
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = twice_risk_infinity + jnp.sum(inputs.eigs_K * inputs.rho_init)

    def get_normalization_factors(grad_norm):
        """Compute normalization factors based on adaptive optimizer type."""
        if adaptive == 'adam':
            return 1.0, grad_norm
        elif adaptive == 'rmsprop_dana':
            return grad_norm, 1.0
        else:
            return 1.0, 1.0

    def inverse_3x3(omega):
        # Extract matrix elements
        a11, a12, a13 = omega[0][0], omega[0][1], omega[0][2]
        a21, a22, a23 = omega[1][0], omega[1][1], omega[1][2]
        a31, a32, a33 = omega[2][0], omega[2][1], omega[2][2]

        # Calculate determinant
        det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
               - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

        # Calculate each element of inverse matrix
        inv = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

        inv[0][0] = (a22*a33 - a23*a32) / det
        inv[0][1] = (a13*a32 - a12*a33) / det
        inv[0][2] = (a12*a23 - a13*a22) / det

        inv[1][0] = (a23*a31 - a21*a33) / det
        inv[1][1] = (a11*a33 - a13*a31) / det
        inv[1][2] = (a13*a21 - a11*a23) / det

        inv[2][0] = (a21*a32 - a22*a31) / det
        inv[2][1] = (a12*a31 - a11*a32) / det
        inv[2][2] = (a11*a22 - a12*a21) / det

        return jnp.array(inv)

    def omega_full(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        delta = delta_fn(time_plus)
        
        # Row 1: Evolution of rho
        omega_11 = -2.0 * (g2 + g1 * g3) * eigs_K + \
                   ((batch + 1.0) / batch) * (g2**2 + 2.0 * g1 * g3 * g2 + g1**2 * g3**2) * eigs_K**2
        omega_12 = g3**2 * (1.0 - delta)**2 * jnp.ones_like(eigs_K)
        omega_13 = -2.0 * g3 * (1.0 - delta) + \
                   2.0 * (g2 * g3 + g3**2 * g1) * (1.0 - delta) * eigs_K
        omega_1 = jnp.array([omega_11, omega_12, omega_13])

        # Row 2: Evolution of sigma
        omega_21 = ((batch + 1.0) / batch) * g1**2 * eigs_K**2
        omega_22 = (-2.0 * delta + delta**2) * jnp.ones_like(eigs_K)
        omega_23 = 2.0 * g1 * eigs_K * (1.0 - delta)
        omega_2 = jnp.array([omega_21, omega_22, omega_23])

        # Row 3: Evolution of chi
        omega_31 = g1 * eigs_K - ((batch + 1.0) / batch) * eigs_K**2 * (g1 * g2 + g1**2 * g3)
        omega_32 = (-g3 + g3 * delta * (2.0 - delta)) * jnp.ones_like(eigs_K)
        omega_33 = -delta - (g2 - g2 * delta + 2.0 * (1.0 - delta) * g1 * g3) * eigs_K
        omega_3 = jnp.array([omega_31, omega_32, omega_33])

        omega = jnp.array([omega_1, omega_2, omega_3])  # 3 x 3 x d
        return omega

    def omega_approximate(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        delta = delta_fn(time_plus)
        
        omega11 = -2.0 * g2 * eigs_K
        omega12 = jnp.zeros_like(eigs_K)
        omega13 = -2.0 * g3 * jnp.ones_like(eigs_K)
        omega1 = jnp.array([omega11, omega12, omega13])

        omega21 = jnp.zeros_like(eigs_K)
        omega22 = -2.0 * delta * jnp.ones_like(eigs_K)
        omega23 = 2.0 * g1 * eigs_K
        omega2 = jnp.array([omega21, omega22, omega23])

        omega31 = g1 * eigs_K
        omega32 = -g3 * jnp.ones_like(eigs_K)
        omega33 = -delta - g2 * eigs_K
        omega3 = jnp.array([omega31, omega32, omega33])

        omega = jnp.array([omega1, omega2, omega3])  # 3 x 3 x d
        return omega

    def forcing_term(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        g3 = g3_fn(time_plus) / g3_norm
        
        Gamma = jnp.array([
            (g2**2 + 2.0 * g1 * g2 * g3 + g1**2 * g3**2) / batch,
            g1**2 / batch,
            (-g1 * g2 - g1**2 * g3) / batch
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d
    
    def forcing_term_approximate(time_plus, grad_norm):
        g1_norm, g3_norm = get_normalization_factors(grad_norm)
        
        # Normalize hyperparameters to apply adaptivity
        g1 = g1_fn(time_plus) / g1_norm
        g2 = g2_fn(time_plus)
        
        Gamma = jnp.array([
            g2**2 / batch,
            g1**2 / batch,
            0.0
        ])
        return jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d

    def ode_update(carry, time):
        v, twice_risk = carry
        time_plus = jnp.exp(time + dt)
        time_plus_minus_one = time_plus - 1.0
        
        # Use sqrt(risk) as proxy for gradient norm when adaptive is not None
        grad_norm = jnp.sqrt(twice_risk / 2.0) if adaptive is not None else 1.0
        
        omega = omega_approximate(time_plus_minus_one, grad_norm) if approximate else omega_full(time_plus_minus_one, grad_norm)
        identity = jnp.tensordot(jnp.eye(3), jnp.ones(D), 0)

        A = inverse_3x3(identity - (dt * time_plus) * omega)  # 3 x 3 x d

        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)

        G_lambda = forcing_term_approximate(time_plus_minus_one, grad_norm) if approximate else forcing_term(time_plus_minus_one, grad_norm)
        x_temp = v + dt * time_plus * twice_risk_infinity * G_lambda

        x = jnp.einsum('ijk, jk -> ik', A, x_temp)

        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)

        v_new = x + (dt * time_plus * y * jnp.sum(x * z) /
                    (1.0 - dt * time_plus * jnp.sum(y * z)))

        twice_risk_new = twice_risk_infinity + jnp.sum(eigs_K * v_new[0])
        return (v_new, twice_risk_new), twice_risk

    init_carry = (jnp.array([rho_init, sigma_init, chi_init]), risk_init)
    _, twice_risks = jax.lax.scan(ode_update, init_carry, times)
    return jnp.exp(times)-1.0, twice_risks
