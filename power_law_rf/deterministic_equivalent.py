import jax
import jax.numpy as jnp
import scipy as sp
import time


def theory_limit_loss(alpha, beta, V, D):
    """Generate the 'exact' finite V, D expression the residual risk level (risk at time infinity)

    RISK is defined as the MSE/2.0

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
        kappa = theory_kappa(alpha, V, D)
        cstar = jnp.sum(jnp.arange(1, V, 1.0) ** (-2.0 * (beta + alpha)) / (jnp.arange(1, V, 1.0) ** (-2.0 * (alpha)) * kappa * (D ** (2 * alpha)) + 1.0))

    if 2*alpha < 1.0:
        tau = theory_tau(alpha,V,D)
        cstar = jnp.sum(jnp.arange(1, V, 1.0) ** (-2.0 * (beta + alpha)) / (jnp.arange(1, V, 1.0) ** (-2.0 * (alpha)) * tau + 1.0))


    return cstar/2.0




def theory_kappa(alpha, V, D):
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
    c, _ = sp.integrate.quad(lambda x: 1.0/ (1.0 + x ** (2 * alpha)), 0.0, TMAX)
    kappa = c ** (-2.0 * alpha)

    kappa_it = lambda k : sp.integrate.quad(lambda x: 1.0 / (k + x ** (2 * alpha)), 0.0, V / D)[0]
    eps = 10E-4
    error = 1.0
    while error > eps:
        kappa1 = 1.0 / kappa_it(kappa)
        error = abs(kappa1 / kappa - 1.0)
        kappa = kappa1
    return kappa


def theory_tau(alpha, V, D):
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

    tau_it = lambda k : jnp.sum(1.0 / (D * (jnp.arange(1, V, 1) ** (2 * alpha) + k)))
    tau = tau_it(0)
    eps = 10E-4
    error = 1.0
    while error > eps:
        tau1 = 1.0 / tau_it(tau)
        error = abs(tau1 / tau - 1.0)
        tau = tau1
    return tau

def theory_lambda_min(alpha, V, D):
    """Find the solution of the equation 1 = (1/d) * sum_{j=1}^v j^{-4alpha} * u^2 / (j^{-2alpha} * u - 1)^2
    in the interval [-(v/d)*kappa * d^{2alpha}, 0] using binary search.
    
    Then calculate z = 1/u - (1/d) * sum_{j=1}^v j^{-2alpha} / (j^{-2alpha} * u - 1),
    whhich is the minimum eigenvalue of the deterministic equivalent.
    
    Parameters
    ----------
    alpha : float
        parameter of the model
    V, D : integers
        parameters of the model
        
    Returns
    -------
    z : float
        The calculated value of z, which should be positive
    """
    # Calculate kappa
    kappa = theory_kappa(alpha, V, D)
    
    # Define the interval for binary search
    a = -(V/D)*kappa * (D ** (2 * alpha))
    b = 0.0
    
    # Define the function to find the root of
    def equation(u):
        js = jnp.arange(1, V+1, 1)
        j_2alpha = js ** (-2.0 * alpha)
        j_4alpha = js ** (-4.0 * alpha)
        
        # Calculate the sum
        sum_term = jnp.sum(j_4alpha * (u ** 2) / ((j_2alpha * u - 1.0) ** 2))
        result = 1.0 - (1.0 / D) * sum_term
        return result
    
    # Binary search to find the solution
    tolerance = 1e-10
    max_iterations = 100
    
    for i in range(max_iterations):
        u = (a + b) / 2.0
        f_u = equation(u)
        
        if abs(f_u) < tolerance:
            break
            
        if f_u > 0:
            b = u
        else:
            a = u
    
    # Now calculate z using the found u
    js = jnp.arange(1, V+1, 1)
    j_2alpha = js ** (-2.0 * alpha)
    
    # Calculate the sum term
    sum_term = jnp.sum(j_2alpha / (j_2alpha * u - 1.0))
    
    # Calculate z
    z = 1.0 / u - (1.0 / D) * sum_term
    
    return z


def theory_rhos(alpha, beta, d):
    """Generate the initial rho_j's deterministically using 
    the analysis of the deterministic equivalent.

    Parameters
    ----------
    alpha, beta, d : floats
        parameters of the model

    Returns
    -------
    fake_eigs, fake_weights_pp + fake_weights_ac : vectors
        fake_eigs: vector of fake eigenvalues (j^{-2alpha})
        fake_weights_pp + fake_weights_ac: vector of fake weights, which from the theory
        should approximate well the rho_j
    """ 
    fake_eigs = jnp.arange(1,d+1,1) ** (-2*alpha)
    c_beta = 0.0
    if beta > 0.5 and alpha > 0.5:
        c_beta = jnp.sum( jnp.arange(1,d+1,1) ** (-2*(beta)))
    fake_weights_pp = jnp.arange(1,d+1,1) ** (-2*(beta))
    fake_weights_ac = (c_beta/d) * jnp.ones_like(fake_eigs)
    return fake_eigs, fake_weights_pp + fake_weights_ac



def theory_m_batched(v, d, alpha, xs,
                     eta = -6,
                     eta0 = 6.0,
                     eta_steps = 50,
                     j_batch = 100,
                     x_batch = 1000):
  """Split xs into batches then generate the powerlaw m by Newton's method on each batch.

  Parameters
  ----------
  v,d,alpha : floats
      parameters of the model
  xs : vector
      The vector of x-positions at which to estimate the spectrum.  Complex is also possible.
  eta : float
      Error tolerance
  j_batch: int
      Batch size for j, which ranges from 1 to v. This prevents memory usage from growing with v
      inside the Newton method update.
  x_batch: int
      Batch size for x's. More x's creates a finer grid used to discretize the density when integrating
      so batching x's is required to prevent memory usage from growing as we increase the accuracy.

  Returns
  -------
  ms: vector
      m_Lambda evaluated at xs.
  """

  xs=jnp.complex64(xs)
  xsplits = jnp.split(xs,jnp.arange(1, len(xs) // x_batch, 1) * x_batch)
  ms = jnp.concatenate([theory_m_batched_xsplit(v, d, alpha, xsplit, eta, eta0, eta_steps, j_batch) for xsplit in xsplits])

  return ms


def theory_m_batched_xsplit(v, d, alpha, xsplit, eta, eta0, eta_steps, j_batch):
  """Generate the powerlaw m by Newton's method.


  Parameters
  ----------
  v,d,alpha : floats
      parameters of the model
  xsplit : vector
      The vector of x-positions at which to estimate the spectrum.  Complex is also possible.
  eta : float
      Error tolerance
  j_batch: int
      Batch size for j, which ranges from 1 to v. This prevents memory usage from growing with v
      inside the Newton method update.

  Returns
  -------
  msplit: vector
      m_Lambda evaluated at xsplit.
  """
  v = jnp.int32(v)
  d = jnp.complex64(d)
  js = jnp.arange(1, v+1, 1, dtype=jnp.complex64) ** (-2.0 * alpha)
  jt = jnp.reshape(js, (j_batch, -1))
  ones_jt_slice = jnp.ones_like(jt)[0]

  # One Newton's method update step for current estimate m on a single value of z
  def mup_single(m,z):
      m1 = m
      F = m1
      Fprime = jnp.ones_like(m1, dtype=jnp.complex64)
      for j in range(j_batch):
          denom = (jnp.outer(jt[j], m1) - jnp.outer(ones_jt_slice, z))
          F += (1.0 / d) * jnp.sum(jnp.outer(jt[j], m1) / denom, axis=0)
          Fprime -= (1.0 / d) * jnp.sum(jnp.outer(jt[j], z) / (denom ** 2), axis=0)
      return (-F + 1.0) / Fprime + m1

  def mup_scan_body(ms, z, x):
      return mup_single(ms, z*1.0j+x), False

  etas = jnp.logspace(eta0, eta, num=eta_steps, dtype=jnp.float32)
  msplit = jax.lax.scan(lambda m, z: mup_scan_body(m, z, xsplit), jnp.ones_like(xsplit, dtype=jnp.complex64), etas)[0]
  return msplit


def theory_f_measure(v, d, alpha, beta, xs, m_fn = theory_m_batched,
                     err = -6.0, time_checks = False, j_batch=100):
  """Generate the trace resolvent, weighted by the j^{-2beta},
    and then biased by 1/z (used for rho_j weights)


  Parameters
  ----------
  v, d, alpha, beta : floats
      parameters of the model
  xs : floats
      X-values at which to return the trace-resolvent
  err : float
      Error tolerance, log scale
  m_fn: function
      A function that will return ms on a set of zs
  time_checks: bool
      Print times for each part
  j_batch: batch size across v dimension

  Returns
  -------
  Volterra: vector
      values of the solution of the Volterra
  """

  eps = 10.0**(err)
  zs = xs + 1.0j*eps

  if time_checks:
      print("The number of points on the spectral curve is {}".format(len(xs)))

  eta = jnp.log10(eps * (d ** (-2 * alpha)))
  eta0 = 6
  eta_steps = jnp.int32(40 + 10 * (2 * alpha) * jnp.log(d))

  start = time.time()
  if time_checks:
      print("Running the Newton generator with {} steps".format(eta_steps))

  ms = m_fn(v, d, alpha, zs, eta, eta0, eta_steps, j_batch)

  end = time.time()
  if time_checks:
      print("Completed Newton in {} time".format(end - start))
  start = end

  js = jnp.arange(1, v+1, 1) ** (-2.0 * alpha)
  jbs = jnp.arange(1, v+1, 1) ** (-2.0 * (alpha + beta))

  jt = jnp.reshape(js, (j_batch, -1))
  jbt = jnp.expand_dims(jnp.reshape(jbs, (j_batch, -1)), -1)
  ones_jt_slice = jnp.ones_like(jt)[0]

  F_measure = jnp.zeros_like(ms)

  for j in range(j_batch):
      F_measure += jnp.sum(jbt[j] / (jnp.outer(jt[j], ms) - jnp.outer(ones_jt_slice, zs + 1.0j * (10 ** eta))), axis=0)

  return jnp.imag(F_measure/zs) / jnp.pi


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
        i = i+ 1
    return integrals


def deterministic_rho_weights(v, d, alpha, beta, a, b, f_measure_fn = theory_f_measure, xs_per_split = None):
  """Generate the initial rho_j's deterministically.
  This performs many small contour integrals each surrounding the real eigenvalues
  where the vector a contains the values for the lower (left) edges of the
  contours and the vector b contains the values of the upper (right) edges of the
  contours.

  The quantity we want to calculate is these contour integrals over the density
  of zs, but we are choosing the xs to discretize this density. We therefore need
  to choose the xs to be in a fine enough grid to give the desired accuracy.

  This code uses a method to choose the xs where the eigenvalues are divided
  into chunks which are smaller than geometrically decreasing chunks.

  Parameters
  ----------
  a (vector): lower values of z's to be used to compute the density starting
              from largest j^{-2alpha} to smallest j^{-2alpha}
  b (vector): upper values of z's to be used to compute the density starting from
              largest j^{-2alpha} to smallest j^{-2alpha}
  xs_per_split (int): the number of x values to use per split

  Returns
  -------
  rho_weights: vector
      returns rho_j weights in order of largest eigenvalue to smallest eigenvalue
  """
  if xs_per_split is None:
    xs_per_split = d
  # Sort a and b in decreasing order
  sort_idx = jnp.argsort(-a)  # Negative to sort in descending order
  a = a[sort_idx]
  b = b[sort_idx]

  n = len(a)

  # Create sequence starting from right edge, decreasing by factor that depends on step j
  right_edge = 2.0*1.1
  left_edge = jnp.min(a)*0.5
  sequence = []
  current = right_edge
  j = 1.0
  while current >= left_edge:
    sequence.append(current)
    # Calculate reduction factor based on alpha and step j
    reduction = 1.0 - 1.0/jnp.sqrt(j+2)
    current = current * reduction
    j += 1.0

  # Create bin endpoints by shifting sequence
  left_bin_endpoints = sequence[1:]  # All but first element
  right_bin_endpoints = sequence[:-1]  # All but last element

  a_splits = []
  b_splits = []

  # For each bin defined by the endpoints
  for left, right in zip(left_bin_endpoints, right_bin_endpoints):
      # Find indices where a values fall within this bin
      bin_indices = jnp.where((a >= left) & (a <= right))[0]
      
      if len(bin_indices) > 0:
          # Add the a and b values for these indices to the splits
          a_splits.append(a[bin_indices])
          b_splits.append(b[bin_indices])

  rho_weights = jnp.array([])
  for a_split, b_split in zip(a_splits, b_splits):
    lower_bound_split = jnp.min(a_split)
    upper_bound_split = jnp.max(b_split)
    xs = jnp.linspace(lower_bound_split, upper_bound_split, xs_per_split)
    err = -20.0
    batches = 1

    zs = xs.astype(jnp.complex64)
    density = f_measure_fn(v, d, alpha, beta, zs, err=err, j_batch=batches)

    rho_weights_split = chunk_weights(xs, density, a_split, b_split)
    rho_weights_split = jnp.array(rho_weights_split)
    rho_weights = jnp.concatenate([rho_weights, rho_weights_split], axis=0)

  # Ensure we return exactly n weights
  if len(rho_weights) != n:
      raise ValueError(f"Generated {len(rho_weights)} weights but expected {n}")

  return rho_weights


def theory_empirical_measure(v, d, alpha, xs, m_fn = theory_m_batched,
                     err = -6.0, time_checks = False, j_batch=100):
  """Generate the trace resolvent


  Parameters
  ----------
  v, d, alpha : floats
      parameters of the model
  xs : floats
      X-values at which to return the trace-resolvent
  err : float
      Error tolerance, log scale
  m_fn: function
      A function that will return ms on a set of zs
  time_checks: bool
      Print times for each part
  j_batch: batch size across v dimension

  Returns
  -------
  density: vector
      values of the density of the deterministic equivalent for the empirical measure
  """

  eps = 10.0**(err)
  zs = xs + 1.0j*eps

  if time_checks:
      print("The number of points on the spectral curve is {}".format(len(xs)))

  eta = jnp.log10(eps * (d ** (-2 * alpha)))
  eta0 = 6
  eta_steps = jnp.int32(40 + 10 * (2 * alpha) * jnp.log(d))

  start = time.time()
  if time_checks:
      print("Running the Newton generator with {} steps".format(eta_steps))

  ms = m_fn(v, d, alpha, zs, eta, eta0, eta_steps, j_batch)

  end = time.time()
  if time_checks:
      print("Completed Newton in {} time".format(end - start))
  start = end

  js = jnp.arange(1, v+1, 1) ** (-2.0 * alpha)
  jbs = jnp.ones_like(js)

  jt = jnp.reshape(js, (j_batch, -1))
  jbt = jnp.expand_dims(jnp.reshape(jbs, (j_batch, -1)), -1)
  ones_jt_slice = jnp.ones_like(jt)[0]

  empirical_measure = jnp.zeros_like(ms)

  for j in range(j_batch):
      empirical_measure += jnp.sum(jbt[j] / (jnp.outer(jt[j], ms) - jnp.outer(ones_jt_slice, zs + 1.0j * (10 ** eta))), axis=0)

  return jnp.imag(empirical_measure) / jnp.pi

def deterministic_spectra(v,d,alpha, xs_per_split=None):
    """Generate the eigenvalues of the deterministic equivalent.

    These are taken as points at which the (unnormalized) 
    cdf of the measure crosses integer + 0.5 thresholds.

    Parameters
    ----------
    v, d, alpha : floats
        parameters of the model
    xs_per_split : int
        the number of x values to use per split

    Returns
    -------
    spectra : vector
        the eigenvalues of the deterministic equivalent
    """
    if xs_per_split is None:
        xs_per_split = d
    right_edge = 2.0*1.1
    left_edge = theory_lambda_min(alpha,v,d)*0.5
    # Create sequence starting from right edge, decreasing by factor that depends on step j
    sequence = []
    current = right_edge
    j = 1.0
    while current >= left_edge:
        sequence.append(current)
        # Calculate reduction factor based on alpha and step j
        #reduction = 1 - max(min(2*alpha/j, 0.5),1.0/jnp.sqrt(j))
        reduction = 1.0 - 1.0/jnp.sqrt(j+2)
        current = current * reduction
        j += 1.0

    # Create bin endpoints by shifting sequence
    left_bin_endpoints = sequence[1:]  # All but first element
    right_bin_endpoints = sequence[:-1]  # All but last element

    spectra = []
    current_cdf = 0.0

    for left, right in zip(left_bin_endpoints, right_bin_endpoints):
        xs = jnp.linspace(left, right, xs_per_split)
        err = -20.0
        batches = 1
        zs = xs.astype(jnp.complex64)
        density = theory_empirical_measure(v, d, alpha, zs, err=err, j_batch=batches)

        cdf = jnp.cumsum(density)*(xs[1]-xs[0])
        cdf = (cdf[-1]+current_cdf)-cdf
        current_cdf = cdf[0]
        def frac(x):
            return x - jnp.floor(x)
        # Find points where cdf crosses integer + 0.5 thresholds
        for i in range(len(xs)-1):
            if cdf[i] > cdf[i+1]: # Check we're going in descending order
                if frac(cdf[i]) > 0.5 and frac(cdf[i+1]) < 0.5:
                    spectra.append(xs[i])

    return jnp.sort(jnp.array(spectra))[::-1]
