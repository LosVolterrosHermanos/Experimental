import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable


class ODEInputs(NamedTuple):
    eigs_K: jnp.ndarray      # eigenvalues of covariance matrix (W^TDW)
    rho_init: jnp.ndarray    # initial rho_j's (rho_j^2)
    chi_init: jnp.ndarray    # initialization of chi's
    sigma_init: jnp.ndarray  # initialization of sigma's (xi^2_j)
    risk_infinity: float     # risk value at time infinity


class DanaHparams(NamedTuple):
    g1: Callable[[float], float]  # learning rate function
    g2: Callable[[float], float]  # learning rate function
    g3: Callable[[float], float]  # learning rate function
    delta: Callable[[float], float]  # momentum function


def ode_resolvent_log_implicit_full(
    inputs: ODEInputs,
    opt_hparams: DanaHparams,
    batch: int,
    D: int,
    t_max: float,
    dt: float
):
    """Generate the theoretical solution to momentum

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
            represents the risk value at time infinity

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

    Returns
    -------
    t_grid: numpy.array(float)
        the time steps used, which will discretize (0,t_max) into n_grid points
    risks: numpy.array(float)
        the values of the risk
    """
    times = jnp.arange(0, jnp.log(t_max), step=dt, dtype=jnp.float32)
    risk_init = inputs.risk_infinity + jnp.sum(inputs.eigs_K * inputs.rho_init)

    def inverse_3x3(omega):
        # Extract matrix elements
        a11, a12, a13 = omega[0][0], omega[0][1], omega[0][2]
        a21, a22, a23 = omega[1][0], omega[1][1], omega[1][2]
        a31, a32, a33 = omega[2][0], omega[2][1], omega[2][2]

        # Calculate determinant
        det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
               - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

        #if abs(det) < 1e-10:
        #    raise ValueError("Matrix is singular or nearly singular")

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

    def ode_update(carry, time):
        v, risk = carry
        time_plus = jnp.exp(time + dt)

        omega_11 = -2.0 * batch * opt_hparams.g2(time_plus) * inputs.eigs_K + batch * (batch + 1.0) * opt_hparams.g2(time_plus)**2 * inputs.eigs_K**2
        omega_12 = opt_hparams.g3(time_plus)**2 * jnp.ones_like(inputs.eigs_K)
        omega_13 = 2.0 * opt_hparams.g3(time_plus) * (-1.0 + opt_hparams.g2(time_plus) * batch * inputs.eigs_K)
        omega_1 = jnp.array([omega_11, omega_12, omega_13])

        omega_21 = batch * (batch + 1.0) * opt_hparams.g1(time_plus)**2 * inputs.eigs_K**2
        omega_22 = (-2.0 * opt_hparams.delta(time_plus) + opt_hparams.delta(time_plus)**2) * jnp.ones_like(inputs.eigs_K)
        omega_23 = 2.0 * opt_hparams.g1(time_plus) * inputs.eigs_K * batch * (1.0 - opt_hparams.delta(time_plus))
        omega_2 = jnp.array([omega_21, omega_22, omega_23])

        omega_31 = opt_hparams.g1(time_plus) * batch * inputs.eigs_K
        omega_32 = -opt_hparams.g3(time_plus) * jnp.ones_like(inputs.eigs_K)
        omega_33 = -opt_hparams.delta(time_plus) - opt_hparams.g2(time_plus) * batch * inputs.eigs_K
        omega_3 = jnp.array([omega_31, omega_32, omega_33])

        omega = jnp.array([omega_1, omega_2, omega_3])  # 3 x 3 x d

        identity = jnp.tensordot(jnp.eye(3), jnp.ones(D), 0)

        A = inverse_3x3(identity - (dt * time_plus) * omega)  # 3 x 3 x d

        Gamma = jnp.array([batch * opt_hparams.g2(time_plus)**2,
                          batch * opt_hparams.g1(time_plus)**2, 0.0])
        z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), inputs.eigs_K)
        G_lambda = jnp.einsum('i,j->ij', Gamma, inputs.eigs_K)  # 3 x d

        x_temp = v + dt * time_plus * inputs.risk_infinity * G_lambda
        x = jnp.einsum('ijk, jk -> ik', A, x_temp)

        y = jnp.einsum('ijk, jk -> ik', A, G_lambda)

        v_new = x + (dt * time_plus * y * jnp.sum(x * z) /
                    (1.0 - dt * time_plus * jnp.sum(y * z)))

        risk_new = inputs.risk_infinity + jnp.sum(inputs.eigs_K * v_new[0])
        return (v_new, risk_new), risk

    init_carry = (jnp.array([inputs.rho_init, inputs.sigma_init, inputs.chi_init]), risk_init)
    _, risks = jax.lax.scan(ode_update, init_carry, times)
    return jnp.exp(times), risks/2
