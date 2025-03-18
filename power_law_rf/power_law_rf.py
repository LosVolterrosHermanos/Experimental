import jax
import jax.numpy as jnp
import jax.random as random
import scipy as sp

from power_law_rf.deterministic_equivalent import theory_limit_loss, deterministic_rho_weights,theory_rhos

########################################################
# Power-law random features regression class
# Theory tools are callable from the class
# Instantiate the class for a sample of the PLRF
########################################################

class PowerLawRF:
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
  
  def get_theory_limit_loss(self):
      """Returns the theoretical limit of the loss (residual risk) for the current model parameters.
      
      Calculates the theoretical prediction for the residual risk level (risk at infinite time)
      using the model's alpha, beta, v (number of random features), and d (input dimension) parameters.
      
      Returns:
          float: Theoretical prediction for the residual risk level
      """
      return theory_limit_loss(self.alpha,self.beta,self.v,self.d)
  
  def get_theory_rhos(self):
    """Get the theoretical rhos for the current model parameters.

    Returns
    -------
    ndarray,ndarray
        The first array contains (approximate) eigenvalues 
        of the Hessian, and the second contains the 
        corresponding rhos.  These are chosen to be a good
        approximation of the true eigenvalues and rhos, in
        in the sense of a inducing similar measures. In
        particular, the eigenvalues do not need to match 
        well for large index.
    """
    return theory_rhos(self.alpha,self.beta,self.d)

  def get_deterministic_rho_weights(self,num_splits, a, b, xs_per_split = 10000):
    """Generate the initial rho_j's deterministically (via self-consistent theory)
    This performs many small contour integrals each surrounding the real eigenvalues
    where the vector a contains the values for the lower (left) edges of the
    contours and the vector b contains the values of the upper (right) edges of the
    contours.
    """
    v, d, alpha, beta = self.v, self.d, self.alpha, self.beta
    return deterministic_rho_weights(v, d, alpha, beta, num_splits, a, b, xs_per_split)
