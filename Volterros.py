import jax
import jax.numpy as jnp
import jax.random as random
import scipy as sp

from typing import NamedTuple, Optional, Union
import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils
from optax.transforms import _accumulation
from optax.transforms import _adding
import chex

from tqdm import tqdm
import time

########################################################
# Power-law random features regression class
# Theory tools are callable from the class
# Instantiate the class for a sample of the PLRF
########################################################

class power_law_RF:
  """
  A class that generates power-law random features regression problems.

  This class creates synthetic regression problems with power-law decaying eigenvalues 
  and target coefficients. The features are generated by first sampling random Gaussian 
  features and then scaling them according to a power law.

  Attributes:
      alpha (float): Power law exponent for eigenvalue decay
      beta (float): Power law exponent for target coefficient decay  
      W (ndarray): Random features matrix of shape (v, d)
      v (int): Hidden dimensionality
      d (int): Embedded dimensionality
      x_grid (ndarray): Grid of indices from 1 to v, shape (1,v)
      population_eigenvalues (ndarray): Power-law decaying eigenvalues
      b (ndarray): Power-law decaying target coefficients
      population_trace (float): Sum of population eigenvalues
      checkW (ndarray): Scaled random features matrix
      checkb (ndarray): Scaled target coefficients
  """

  def __init__(self, alpha, beta, W):
      self.alpha = alpha
      self.beta = beta
      self.W = W
      self.v = self.W.shape[0]
      self.d = self.W.shape[1]
      self.x_grid=jnp.arange(1, self.v+1).reshape(1,self.v)
      self.population_eigenvalues = self.x_grid**(-self.alpha)
      self.b = self.x_grid.transpose()**(-beta)
      self.population_trace = jnp.sum(self.population_eigenvalues)
      self.checkW = W * self.population_eigenvalues.T
      self.checkb = self.x_grid.transpose()**(-alpha-beta)
      
  @classmethod
  def initialize_random(cls, alpha, beta, v, d, key):
      """
      Creates a new power_law_RF instance with randomly initialized features.

      Args:
          alpha (float): Power law exponent for eigenvalue decay
          beta (float): Power law exponent for target coefficient decay
          v (int): Hidden dimensionality
          d (int): Embedded dimensionality
          key (PRNGKey): JAX random number generator key

      Returns:
          power_law_RF: A new instance with randomly sampled features matrix W
                        scaled to have variance 1/d
      """
      # Sample random features matrix with variance 1/d
      W = random.normal(key, (v, d)) / jnp.sqrt(d)
      return cls(alpha=alpha, beta=beta, W=W)
  
  def get_population_risk(self, w):
      """
      Calculates the population risk for given weights.
      
      The population risk is the expected squared error over the data distribution.
      For power-law random features regression, this can be computed analytically
      without sampling data.
      
      Args:
          w (ndarray): Weight vector of shape (d,)
      
      Returns:
          float: Population risk value
      """
      # Project weights onto random features
      proj = jnp.matmul(self.checkW, w)
      
      # Calculate population risk using eigenvalues and target coefficients
      risk = jnp.sum((proj - self.checkb)**2)
      return risk / 2
  

  def get_data(self, key, batch):
      """
      Generates a batch of synthetic data points.
      
      Args:
          key (PRNGKey): JAX random number generator key
          batch (int): Number of data points to generate
          
      Returns:
          tuple: (X, y) where:
              X (ndarray): Input features of shape (batch, d)
              y (ndarray): Target values of shape (batch, 1)
      """
      # Generate random features
      x = random.normal(key, (batch, self.v))
      
      return jnp.matmul(x, self.checkW), jnp.matmul(x, self.checkb)
  
  def get_theory_limitloss(self):
      """Returns the theoretical limit of the loss (residual risk) for the current model parameters.
      
      Calculates the theoretical prediction for the residual risk level (risk at infinite time)
      using the model's alpha, beta, v (number of random features), and d (input dimension) parameters.
      
      Returns:
          float: Theoretical prediction for the residual risk level
      """
      return power_law_RF.theory_limitloss(self.alpha,self.beta,self.v,self.d)
  
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
          kappa = power_law_RF.theory_kappa(alpha,V,D)
          cstar = jnp.sum( jnp.arange(1,V,1.0)**(-2.0*(beta+alpha))/( jnp.arange(1,V,1.0)**(-2.0*(alpha))*kappa*(D**(2*alpha)) + 1.0))

      if 2*alpha < 1.0:
          #tau = D/jnp.sum( jnp.arange(1,V,1.0)**(-2.0*alpha))
          tau = power_law_RF.theory_tau(alpha,V,D)
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

    ms = power_law_RF.theory_m_batched(v,d,alpha,zs,eta,eta0,etasteps,batches)

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

  def get_theory_rho_weights(self,num_splits, a, b, xs_per_split = 10000):
    """Generate the initial rho_j's deterministically.
    This performs many small contour integrals each surrounding the real eigenvalues
    where the vector a contains the values for the lower (left) edges of the
    contours and the vector b contains the values of the upper (right) edges of the
    contours.
    """
    v, d, alpha, beta = self.v, self.d, self.alpha, self.beta
    return power_law_RF.theory_rho_weights(v, d, alpha, beta, num_splits, a, b, xs_per_split)

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
      density = power_law_RF.theory_f_measure(v, d, alpha, beta, zs, err=err, batches = batches)

      rho_weights_split = power_law_RF.chunk_weights(xs, density, a_split, b_split)
      rho_weights_split = jnp.array(rho_weights_split)
      rho_weights = jnp.concatenate([rho_weights, rho_weights_split], axis=0)

    return rho_weights
  # Compute density for all splits
  #density = jax.vmap(lambda z: jax_gen_trace_fmeasure(V, D, alpha, beta, z, err=-10, batches=1))(zs)

  # Compute rho_weights for all splits
  #rho_weights = jnp.array([weights(x, d, a_s, b_s) for x, d, a_s, b_s in zip(xs, density, a_splits, b_splits)])

  # Flatten the rho_weights
  #rho_weights = rho_weights.flatten()

  

########################################################
# ODE solver
########################################################

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


########################################################
# Optimizers
########################################################

def lsq_streaming_optax_simple(
        key,
        data_oracle,
        batch_size,
        steps,
        optimizer,
        init,
        loss_oracle,
        tqdm_bar = True
):
    """Trains a linear model using stochastic gradient descent with a simple learning rate schedule,
    and outputs the loss curve.  The steps at which the losses are recorded are exponentially spaced,
    which is appropriate for power-law type loss decays.
    
    Args:
        key: JAX PRNGKey for random number generation
        data_oracle: Function that generates batches of (embeddings, labels) training data
        batch_size: Number of samples per batch
        steps: Total number of optimization steps
        learning_rate: Fixed learning rate for SGD
        init: Initial parameter vector
        loss_oracle: Function that computes population loss given parameters
        tqdm_bar: Whether to display a tqdm progress bar
    Returns:
        tuple: (timestamps, losses) where timestamps are the iteration numbers at which
        losses were recorded and losses are the corresponding population loss values
    """
    tx = optimizer
    params = init
    opt_state = tx.init(params)
    state = (params, opt_state)

    def batch_mse(params, embeddings, y):
        y_pred = embeddings@params
        return jnp.mean(optax.l2_loss(y_pred, y))

    def train_step(state,key):
        params, opt_state = state
        (embeddings, y) = data_oracle(key, batch_size)
        loss_fn = lambda params : batch_mse(params, embeddings, y)
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        #return (params, opt_state), loss
        return (params, opt_state), None
    

    # This generates an exponentially spaced sequence of times at which we record the loss.
    # This is to ensure that we are recording the loss at a reasonable frequency.
    # We start at 0 and go up to the total number of steps.
    # We then add the number of steps to the list.
    # We then take the unique values and sort them.
    # This gives us the times at which we record the loss.
    
    losses=[]
    losstimes = jnp.unique(jnp.concatenate(
        [jnp.array([0]),
        jnp.int32(
            1.1**jnp.arange(1,jnp.ceil(jnp.log(steps)/jnp.log(1.1)))
        ),
        jnp.array([steps])]
    ))

    losstime_steps = losstimes[1:]-losstimes[:-1]
    losses.append(loss_oracle(init))
    if tqdm_bar:
        for increment in tqdm(losstime_steps):
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, _ = jax.lax.scan(train_step, state, keyz)
            pop_loss = loss_oracle(state[0])
            losses.append(pop_loss)
    else:
        for increment in losstime_steps:
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, _ = jax.lax.scan(train_step, state, keyz)
            pop_loss = loss_oracle(state[0])
            losses.append(pop_loss)
    return losstimes, losses
    

def powerlaw_schedule(
    init_value: chex.Scalar,
    saturation_value: chex.Scalar,
    power: chex.Scalar,
    time_scale: int,
) -> base.Schedule:
  """Constructs power-law schedule.

  This function decays (or grows) the learning rate, until it is below
  the saturation_value, at which time it is held. The formula is given by
  :math:`max{ I*(1+t / time_scale) ^ {power}, saturation_value}`

  where :math:`I` is the initial value, :math:`t` is the current iteration, 
   :math:`time_scale` is the time scale of the power law, 
   :math:`power` is the power, and :math:`saturation_value` is the value
   at which the power law is saturated.

  Args:
    init_value: initial value for the scalar to be annealed.
    saturation_value: end value of the scalar to be annealed.
    power: the power of the power law.
    time_scale: number of steps over which the power law takes place.
      The scalar starts changing at ``transition_begin`` steps and completes
      the transition by ``transition_begin + transition_steps`` steps.
      If ``transition_steps <= 0``, then the entire annealing process is
      disabled and the value is held fixed at ``init_value``.
    transition_begin: must be positive. After how many steps to start annealing
      (before this many steps the scalar value is held fixed at ``init_value``).

  Returns:
    schedule
      A function that maps step counts to values.

  Examples:
    >>> schedule_fn = optax.powerlaw_schedule(
    ...    init_value=1.0, saturation_value=0.01, time_scale=100, power=2)
    >>> schedule_fn(0)  # learning rate on the first iteration
    Array(1., dtype=float32, weak_type=True)
    >>> schedule_fn(100)  # learning rate on the last iteration
    Array(0.01, dtype=float32, weak_type=True)
  """

  def schedule(count):
    frac = 1 + count / time_scale
    return jnp.maximum((init_value) * (frac**power),saturation_value)

  return schedule


class DanaOptimizerState(NamedTuple):
  """State for the Dana algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  y: base.Updates

def dana_optimizer(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    y_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    ) -> base.GradientTransformation:
    """Rescale updates according to the Adam algorithm.

    See :func:`optax.adam` for more details.

    Args:
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared grads.
        eps: Term added to the denominator to improve numerical stability.
        eps_root: Term added to the denominator inside the square-root to improve
        numerical stability when backpropagating gradients through the rescaling.
        mu_dtype: Optional `dtype` to be used for the first order accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.
        nesterov: Whether to use Nesterov momentum. The variant of Adam with
        Nesterov momentum is described in [Dozat 2016]

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    def init_fn(params):
        y = otu.tree_zeros_like(params, dtype=y_dtype)  # Momentum
        return DanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        newg2 = g2(state.count)
        newg3 = g3(state.count)

        y = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + newg1*u,
            state.y,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m,u : newg2*u if m is None else -1.0*(newg2*u + newg3*m),
            y,
            updates,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaOptimizerState(count=count_inc, y=y)

    return base.GradientTransformation(init_fn, update_fn)