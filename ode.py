import jax.numpy as jnp

def ode_resolvent_log_implicit_full(eigs_K, rho_init, chi_init, sigma_init, risk_infinity,
                  g1, g2, g3, delta, batch, D, t_max, Dt):
  """Generate the theoretical solution to momentum

  Parameters
  ----------
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
  WtranD : array (v x d)
      WtranD where D = diag(j^(-2alpha)) and W is the random matrix
  alpha : float
      data complexity
  V : float
      vocabulary size
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
  Dt : float
      time step used in Euler

  Returns
  -------
  t_grid: numpy.array(float)
      the time steps used, which will discretize (0,t_max) into n_grid points
  risks: numpy.array(float)
      the values of the risk

  """
  #times = jnp.arange(0, t_max, step = Dt, dtype= jnp.float64)
  times = jnp.arange(0, jnp.log(t_max), step = Dt, dtype= jnp.float32)

  risk_init = risk_infinity + jnp.sum(eigs_K * rho_init)

  def inverse_3x3(Omega):
      # Extract matrix elements
      a11, a12, a13 = Omega[0][0], Omega[0][1], Omega[0][2]
      a21, a22, a23 = Omega[1][0], Omega[1][1], Omega[1][2]
      a31, a32, a33 = Omega[2][0], Omega[2][1], Omega[2][2]

      # Calculate determinant
      det = (a11*a22*a33 + a12*a23*a31 + a13*a21*a32
            - a13*a22*a31 - a11*a23*a32 - a12*a21*a33)

      #if abs(det) < 1e-10:
      #    raise ValueError("Matrix is singular or nearly singular")

      # Calculate each element of inverse matrix
      inv = [[0,0,0],[0,0,0],[0,0,0]]

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

  def odeUpdate(stuff, time):
    v, risk = stuff
    timePlus = jnp.exp(time + Dt)

    Omega11 = -2.0 * batch * g2(timePlus) * eigs_K + batch * (batch + 1.0) * g2(timePlus)**2 * eigs_K**2
    Omega12 = g3(timePlus)**2 * jnp.ones_like(eigs_K)
    Omega13 = 2.0 * g3(timePlus) * (-1.0 + g2(timePlus) * batch * eigs_K)
    Omega1 = jnp.array([Omega11, Omega12, Omega13])

    Omega21 = batch * (batch + 1.0) * g1(timePlus)**2 * eigs_K**2
    Omega22 = ( -2.0 * delta(timePlus) + delta(timePlus)**2 ) * jnp.ones_like(eigs_K)
    Omega23 = 2.0 * g1(timePlus) * eigs_K * batch * ( 1.0 - delta(timePlus) )
    Omega2 = jnp.array([Omega21, Omega22, Omega23])

    Omega31 = g1(timePlus) * batch * eigs_K
    Omega32 = -g3(timePlus) * jnp.ones_like(eigs_K)
    Omega33 = -delta(timePlus) - g2(timePlus) * batch * eigs_K
    Omega3 = jnp.array([Omega31, Omega32, Omega33])

    Omega = jnp.array([Omega1, Omega2, Omega3]) #3 x 3 x d

    Identity = jnp.tensordot( jnp.eye(3), jnp.ones(D), 0 )

    A = inverse_3x3(Identity - (Dt * timePlus) * Omega) #3 x 3 x d

    Gamma = jnp.array([batch * g2(timePlus)**2, batch * g1(timePlus)**2, 0.0])
    z = jnp.einsum('i, j -> ij', jnp.array([1.0, 0.0, 0.0]), eigs_K)
    G_Lambda = jnp.einsum('i,j->ij', Gamma, eigs_K) #3 x d

    x_temp = v + Dt * timePlus * risk_infinity * G_Lambda
    x = jnp.einsum('ijk, jk -> ik', A, x_temp)

    y = jnp.einsum('ijk, jk -> ik', A, G_Lambda)

    vNew = x + ( Dt * timePlus * y * jnp.sum(x * z) / (1.0 - Dt * timePlus * jnp.sum(y * z)) )
    #vNew = vNew.at[0].set(jnp.maximum(vNew[0], 0.0))
    #vNew[0] = jnp.maximum(vNew[0], 10**(-7))
    #vNew[2] = jnp.maximum(vNew[2], 10**(-7))

    riskNew = risk_infinity + jnp.sum(eigs_K * vNew[0])
    return (vNew, riskNew), risk #(risk, vNew[0])

  _, risks = jax.lax.scan(odeUpdate,(jnp.array([rho_init,sigma_init, chi_init]),risk_init),times)
  return jnp.exp(times), risks/2
