import jax
import jax.numpy as jnp
import scipy as sp
import time

def theory_limitloss(alpha, beta,V,D):
    """Generate the 'exact' finite V, D expression the residual risk level (risk at time infinity)

    Parameters
    ----------
    alpha,beta : floats
        parameters of the model, ASSUMES V>D

    Returns
    -------
    theoretical prediction for the norm
    """
    cstar = 0.0
    if 2*alpha >= 1.0:
        kappa = theory_kappa(alpha,V,D)
        cstar = jnp.sum( jnp.arange(1,V,1.0)**(-2.0*(beta+alpha))/( jnp.arange(1,V,1.0)**(-2.0*(alpha))*kappa*(D**(2*alpha)) + 1.0))

    if 2*alpha < 1.0:
        #tau = D/jnp.sum( jnp.arange(1,V,1.0)**(-2.0*alpha))
        tau = theory_tau(alpha,V,D)
        cstar = jnp.sum( jnp.arange(1,V,1.0)**(-2.0*(beta+alpha))/( jnp.arange(1,V,1.0)**(-2.0*(alpha))*tau + 1.0))


    return cstar

def theory_kappa(alpha, V,D):
    """Generate coefficient kappa with finite sample corrections.
    Parameters
    ----------
    alpha : float
        parameter of the model.
    V,D : integers
        parameters of the model.

    Returns
    -------
    theoretical prediction for kappa parameter
    """

    TMAX = 1000.0
    c, _ = sp.integrate.quad(lambda x: 1.0/(1.0+x**(2*alpha)),0.0,TMAX)
    kappa=c**(-2.0*alpha)

    kappa_it = lambda k : sp.integrate.quad(lambda x: 1.0/(k+x**(2*alpha)),0.0,V/D)[0]
    eps = 10E-4
    error = 1.0
    while error > eps:
        kappa1 = 1.0/kappa_it(kappa)
        error = abs(kappa1/kappa - 1.0)
        kappa = kappa1
    return kappa

def theory_tau(alpha, V,D):
    """Generate coefficient tau with finite sample corrections.
    Parameters
    ----------
    alpha : float
        parameter of the model.
    V,D : integers
        parameters of the model.

    Returns
    -------
    theoretical prediction for kappa parameter
    """

    tau_it = lambda k : jnp.sum( 1.0/(D*(jnp.arange(1,V,1)**(2*alpha) +k)))
    tau = tau_it(0)
    eps = 10E-4
    error = 1.0
    while error > eps:
        tau1 = 1.0/tau_it(tau)
        error = abs(tau1/tau - 1.0)
        tau = tau1
    return tau

def get_hessian_spectra(self):
    """Get eigenvalues of the Hessian matrix of the problem

    Returns
    -------
    ndarray
        Array containing the eigenvalues of the Hessian matrix, computed as 
        the squared singular values of the checkW matrix.
    """
    _, s, _ =jnp.linalg.svd(self.checkW,full_matrices=False)
    return s**2

def get_rhos(self):
    """Get squared-projections of the residual (b) in the direction of the eigenmodes of the Hessian.

    Returns
    -------
    ndarray
        Array containing the squared-projections of the residual vector b onto the eigenvectors
        of the Hessian matrix, normalized by the corresponding eigenvalues.
    """
    Uvec, s, _ =jnp.linalg.svd(self.checkW,full_matrices=False)

    #Compute < ( D^1/2 W W^T D^(1/2) - z)^{-1}, D^(1/2) b >
    check_beta_weight = jnp.tensordot(self.checkb,Uvec,axes=[[0],[0]])[0]

    rhos = (check_beta_weight)**2 / s**2
    rhos.astype(jnp.float32)
    return rhos


def theory_lambda_min(alpha):
  """Generate left edge of the spectral measure (not accurate and only for alpha > 0.5)

  Parameters
  ----------
  alpha : float
      parameter of the model, ASSUMES V>D

  Returns
  -------
  theoretical prediction for the norm
  """

  TMAX = 1000.0
  c, _ = sp.integrate.quad(lambda x: 1.0/(1.0+x**(2*alpha)),0.0,TMAX)

  return (1/(2*alpha-1))*((2*alpha/(2*alpha-1)/c)**(-2*alpha))

def theory_m_batched(v,d, alpha, xs,
              eta = -6,
              eta0 = 6.0,
              etasteps=50,
              batches = 100,
              zbatch=1000):
  """Generate the powerlaw m by newton's method


  Parameters
  ----------
  v,d,alpha : floats
      parameters of the model
  xs : vector
      The vector of x-positions at which to estimate the spectrum.  Complex is also possible.
  eta : float
      Error tolerance

  Returns
  -------
  m_Lambda: vector
      m_Lambda evaluated at xs.
  """
  #if zbatch > 0:
  #    xsplit = jnp.split(xs,jnp.arange(1,len(xs)//zbatch,1)*zbatch)
  #    ms = jnp.concatenate( [jax_gen_m_batched(v,d,alpha,x,eta,eta0,etasteps,batches,zbatch=0) for x in xsplit] )
  #    return ms
  v=jnp.int32(v)
  d=jnp.complex64(d)
  xs=jnp.complex64(xs)
  xsplit = jnp.split(xs,jnp.arange(1,len(xs)//zbatch,1)*zbatch)


  #print("xs length = {}".format(len(xs)))

  js=jnp.arange(1,v+1,1,dtype=jnp.complex64)**(-2.0*alpha)
  jt=jnp.reshape(js,(batches,-1))
  onesjtslice=jnp.ones_like(jt)[0]

  # One Newton's method update step for current estimate m on a single value of z
  def mup_single(m,z):
      m1 = m
      F=m1
      Fprime=jnp.ones_like(m1,dtype=jnp.complex64)
      for j in range(batches):
          denom = (jnp.outer(jt[j],m1) - jnp.outer(onesjtslice,z))
          F += (1.0/d)*jnp.sum(jnp.outer(jt[j],m1)/denom,axis=0)
          Fprime -= (1.0/d)*jnp.sum(jnp.outer(jt[j],z)/(denom**2),axis=0)
      return (-F + 1.0)/Fprime + m1
      #return 0.1*jnp.where(mask, m1, newm1)+0.9*m1

#    mup_single = jax.jit(mup_single, static_argnums=(0,1))

  def mup_scanner(ms,z,x):
      return mup_single(ms,z*1.0j+x), False

  #mup_scanner = jax.jit(mup_scanner, static_argnums=(0,1))
  #mup_scannerjit =  jax.jit(mup_scanner)

  etas = jnp.logspace(eta0,eta,num = etasteps)
  ms = jnp.concatenate( [jax.lax.scan(lambda m,z: mup_scanner(m,z,x),jnp.ones_like(x, dtype = jnp.complex64),etas)[0] for x in xsplit] )
  #ms, _ = jax.lax.scan(mup_scanner,jnp.ones_like(xs),etas)

  return ms

def theory_f_measure(v,d, alpha, beta, xs,
              err = -6.0, timeChecks = False, batches=100):
  """Generate the trace resolvent


  Parameters
  ----------
  v,d,alpha,beta : floats
      parameters of the model
  xs : floats
      X-values at which to return the trace-resolvent
  err : float
      Error tolerance, log scale
  timeChecks: bool
      Print times for each part

  Returns
  -------
  Volterra: vector
      values of the solution of the Volterra
  """

  eps = 10.0**(err)

  zs = xs + 1.0j*eps

  if timeChecks:
      print("The number of points on the spectral curve is {}".format(len(xs)))

  eta = jnp.log10(eps*(d**(-2*alpha)))
  eta0 = 6
  etasteps = jnp.int32(40 + 10*(2*alpha)*jnp.log(d))

  start=time.time()
  if timeChecks:
      print("Running the Newton generator with {} steps".format(etasteps))

  ms = theory_m_batched(v,d,alpha,zs,eta,eta0,etasteps,batches)

  end = time.time()
  if timeChecks:
      print("Completed Newton in {} time".format(end-start) )
  start = end

  js=jnp.arange(1,v+1,1)**(-2.0*alpha)
  jbs=jnp.arange(1,v+1,1)**(-2.0*(alpha+beta))

  jt=jnp.reshape(js,(batches,-1))
  jbt=jnp.expand_dims(jnp.reshape(jbs,(batches,-1)),-1)
  onesjtslice=jnp.ones_like(jt)[0]

  Fmeasure = jnp.zeros_like(ms)

  for j in range(batches):
      Fmeasure += jnp.sum(jbt[j]/(jnp.outer(jt[j],ms) - jnp.outer(onesjtslice,zs + 1.0j*(10**eta))),axis=0)



  return jnp.imag(Fmeasure/zs) / jnp.pi

def chunk_weights(xs, density, a, b):
    # Compute integrals
      integrals = []
      def theoretical_integral(lower, upper):
        # Normalize density to make it a probability measure
        dx = xs[1] - xs[0]
        #norm = jnp.sum(density) * dx
        #density = density / norm

        # Find indices corresponding to interval [a,b]
        idx = (xs >= lower) & (xs <= upper)
        integral = jnp.sum(density[idx]) * dx
        return float(integral)
      i = 0
      for lower, upper in zip(a,b):
        integrals.append(theoretical_integral(lower, upper))
        #integrals.at[i].set(theoretical_integral(lower, upper))
        i = i+ 1
      return integrals


def theory_rho_weights(v,d,alpha,beta,num_splits, a, b, xs_per_split = 10000):
  """Generate the initial rho_j's deterministically.
  This performs many small contour integrals each surrounding the real eigenvalues
  where the vector a contains the values for the lower (left) edges of the
  contours and the vector b contains the values of the upper (right) edges of the
  contours.

  The quantity we want to calculate is these contour integrals over the density
  of zs, but we are choosing the xs to discretize this density. We therefore need
  to choose the xs to be in a fine enough grid to give the desired accuracy.

  This code uses a hacky method to choose the xs where the eigenvalues are divided
  into num_splits different chunks (each containing the same num of eigenvalues)
  so that the range of x values spanned is large for the large eigenvalues and
  small for the small eigenvalues. Then this uses a linearly spaced grid within
  each split so that each split uses the same number of xs.

  The smallest eigenvalues actually don't need this dense of a grid, because they
  make very small contributions, and the largest eigenvalues don't need this dense
  of a grid because they are far apart. It is actually the intermediate
  eigenvalues that are tricky because they are close together but still contribute
  significantly.

  Parameters
  ----------
  num_splits (int): number of splits
  a (vector): lower values of z's to be used to compute the density starting
              from largest j^{-2alpha} to smallest j^{-2alpha}
  b (vector): upper values of z's to be used to compute the density starting from
              largest j^{-2alpha} to smallest j^{-2alpha}
  xs_per_split (int): the number of x values to use per split

  Returns
  -------
  rho_weights: vector
      returns rho_j weights in order of largest j^{-2alpha} to smallest j^{-2alpha}
  """
  a_splits = jnp.split(a, num_splits)
  b_splits = jnp.split(b, num_splits)

  # Vectorize lower and upper bounds
  lower_bounds = jnp.array([jnp.min(split) for split in a_splits])
  upper_bounds = jnp.array([jnp.max(split) for split in b_splits])

  # Generate xs and zs for all splits
  xs = jnp.vstack([jnp.linspace(lower, upper, xs_per_split) for lower, upper in zip(lower_bounds, upper_bounds)])
  zs = xs.astype(jnp.complex64)

  rho_weights = jnp.array([])
  for a_split, b_split in zip(a_splits, b_splits):
    lower_bound_split = jnp.min(a_split)
    upper_bound_split = jnp.max(b_split)
    xs = jnp.linspace(lower_bound_split, upper_bound_split, xs_per_split)
    err = -10
    batches = 1

    zs = xs.astype(jnp.complex64)
    density = theory_f_measure(v, d, alpha, beta, zs, err=err, batches = batches)

    rho_weights_split = chunk_weights(xs, density, a_split, b_split)
    rho_weights_split = jnp.array(rho_weights_split)
    rho_weights = jnp.concatenate([rho_weights, rho_weights_split], axis=0)

  return rho_weights
# Compute density for all splits
#density = jax.vmap(lambda z: jax_gen_trace_fmeasure(V, D, alpha, beta, z, err=-10, batches=1))(zs)

# Compute rho_weights for all splits
#rho_weights = jnp.array([weights(x, d, a_s, b_s) for x, d, a_s, b_s in zip(xs, density, a_splits, b_splits)])

# Flatten the rho_weights
#rho_weights = rho_weights.flatten()
