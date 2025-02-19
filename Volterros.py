import jax
import jax.numpy as jnp
import numpy as np

def jax_lsq_momentum_simple(key,
                g1, g2, g3, delta, batch, steps, init_x, init_w,
                t_oracle, loss, loss_times = jnp.array([0])
                ):
  """ This routine generates losses for SGD on the least squares
  problem with scalar targets, constant learning rate and constant batch size.
  It stores the losses at every iteration, and performs a single loop.

  Parameters
  ----------
  key : PRNGKey
    Jax PRNGKey
  g1, g2, g3 : function(time)
    The learning rate functions
  delta : function(time)
    The momentum function
  batch : int
    The batch-size to use
  steps : int
    The number of steps of minibatch SGD to generate
  init_x : vector
    The initial state for SGD to use
  init_w : vector
    The initial state for momentum
  traceK : float
    Trace of the covariance matrix of the data; i.e., K = E[aa^T]
  t_oracle: callable
    Takes as an argument a jax RNG key and a batch-size.
    Expects in return two tensors (A, y)
    of dimension (batch x data-dimension) and dimension (batch).
  loss: callable
    Takes as an argument a vector of length data-dimension,
    which is the current linear model parameters, and returns the
    loss.
  loss_times: vector
    Iteration counts at which to compute the loss

  Returns
  -------
  losses: vector
    An array of length 'steps' containing the losses
  loss_times: vector
    Iteration counts at which the losses were computed
  """

  if loss_times.shape[0]==1:
    loss_times = jnp.arange(steps)
  x = jnp.reshape(init_x,(len(init_x),1))
  w = jnp.reshape(init_w,(len(init_w),1))


  def update(z, things):
    keyz, iteration = things
    A,y = t_oracle(keyz, batch)
    x,w = z
    grad = jnp.tensordot(A,jnp.tensordot(A,x,axes=1)-y,axes=[[0],[0]])
    neww = (1.0 - delta(iteration)) * w + g1(iteration) * grad
    newx = x - g2(iteration) * grad - neww * g3(iteration)
    return (newx,neww), x

  keys=jax.random.split(key,steps)
  iters =  jnp.linspace(0.0, steps, num = steps)


 # update_jit = jax.jit(update)
  _, states = jax.lax.scan(update,(x,w),(keys,iters))

  return jax.lax.map(loss, states[loss_times[loss_times< steps]]), loss_times[loss_times< steps], states[-1]


def jax_lsq_momentum_old(key,
                g1, g2, g3, delta, batch, steps, init_x, init_w,
                t_oracle, loss
                ):
  """ This routine generates losses for SGD on the least squares
  problem with scalar targets, constant learning rate and constant batch size.
  It generates the same states as jax_lsq_momentum_simple, but it only computes the loss at sparse collection of times, which correspond to exponentially-spaced times (i.e. an equally spaced collection of times inlog-space).

  Parameters
  ----------
  key : PRNGKey
    Jax PRNGKey
  lr1 : float
    The learning rate to use \gamma_1; should be constant/(tr(K)^2)
  lr2 : float
    The learning rate to use \gamma_2; should be constant/(tr(K))
  theta : float
    The momentum parameter to use
  batch : int
    The batch-size to use
  steps : int
    The number of steps of minibatch SGD to generate
  init_x : vector
    The initial state for SGD to use
  init_w : vector
    The initial state for momentum
  traceK : float
    Trace of the covariance matrix of the data; i.e., K = E[aa^T]
  t_oracle: callable
    Takes as an argument a jax RNG key and a batch-size.
    Expects in return two tensors (A, y)
    of dimension (batch x data-dimension) and dimension (batch).
  loss: callable
    Takes as an argument a vector of length data-dimension,
    which is the current linear model parameters, and returns the
    loss.
  loss_times: vector
    Iteration counts at which to compute the loss

  Returns
  -------
  losses: vector
    An array of length 'steps' containing the losses
  loss_times: vector
    Iteration counts at which the losses were computed
  """

  if steps < 10**5:
    return jax_lsq_momentum_simple(key,
                g1, g2, g3, delta, batch, steps, init_x, init_w,
                t_oracle, loss)
  x = jnp.reshape(init_x,(len(init_x),1))
  w = jnp.reshape(init_w,(len(init_w),1))

  def update(z, things): #things = keys for your stochastic updates and momentum terms
    keyz, iteration = things
    A,y = t_oracle(keyz, batch)
    x,w = z
    #delta = theta / (iteration + traceK)
    grad = jnp.tensordot(A,jnp.tensordot(A,x,axes=1)-y,axes=[[0],[0]])
    neww = (1.0 - delta(iteration)) * w + g1(iteration) * grad
    newx = x - g2(iteration) * grad - neww * g3(iteration)
    return (newx,neww), x

  def skinny_update(z, things):
    keyz, iteration = things
    A,y = t_oracle(keyz, batch)
    x,w = z
    #delta = theta / (iteration + traceK)
    grad = jnp.tensordot(A,jnp.tensordot(A,x,axes=1)-y,axes=[[0],[0]])
    neww = ( 1.0 - delta(iteration) ) * w + g1(iteration) * grad
    newx = x - g2(iteration) * grad - neww * g3(iteration)
    return (newx,neww), False

  p = np.int32(np.ceil(np.log10(steps+1)))

  mkey1,mkey2= jax.random.split(key)
  keys=jax.random.split(mkey1,10**5)
  #deltas = theta / ( jnp.linspace(0.0, 10**5, num = 10**5) + traceK)
  iters = jnp.linspace(0.0, 10**5, num = 10**5)

 # update_jit = jax.jit(update)
  z, states = jax.lax.scan(update,(x,w),(keys, iters))

  losslist = jax.lax.map(loss,states)
  timelist = jnp.arange(1,10**5+1,1)
  lastiter = 10**5

  mkeyout =  jax.random.split(mkey2,p-5)
  for j, mkey in enumerate(mkeyout,start=5):
    u=j-2
    def outerloop(xw, thingz):
      keyz,currentiter = thingz
      mkeys = jax.random.split(keyz, 10**u)
      iterlist = currentiter + jnp.arange(0, 10**u, dtype = jnp.float32)
      #deltas= theta / (iterlist + traceK)
      (newx,neww), _ = jax.lax.scan(skinny_update,xw,(mkeys,iterlist))
      return (newx,neww), loss(newx)
    outerloopsteps = min( (steps-lastiter)//(10**u), 100)
    #outerloopitercounts = lastiter + (10**u)*jnp.arange(1,outerloopsteps+1,1)
    outerloopitercounts = lastiter + (10**u)*jnp.arange(0,outerloopsteps,1)
    keys=jax.random.split(mkey,outerloopsteps)
    z, late_loss = jax.lax.scan(outerloop,z,(keys,outerloopitercounts))
    losslist=jnp.concatenate([losslist,late_loss])
    timelist =jnp.concatenate([timelist,outerloopitercounts])
    lastiter += 10**j
  return losslist, timelist, z[0]

  # return losses,loss_times

def jax_lsq_momentum(key,
                g1, g2, g3, delta, batch, steps, init_x, init_w,
                t_oracle, loss
                ):
  """ This routine generates losses for SGD on the least squares
  problem with scalar targets, constant learning rate and constant batch size.
  It generates the same states as jax_lsq_momentum_simple, but it only computes the loss at sparse collection of times, which correspond to exponentially-spaced times (i.e. an equally spaced collection of times inlog-space).

  Parameters
  ----------
  key : PRNGKey
    Jax PRNGKey
  lr1 : float
    The learning rate to use \gamma_1; should be constant/(tr(K)^2)
  lr2 : float
    The learning rate to use \gamma_2; should be constant/(tr(K))
  theta : float
    The momentum parameter to use
  batch : int
    The batch-size to use
  steps : int
    The number of steps of minibatch SGD to generate
  init_x : vector
    The initial state for SGD to use
  init_w : vector
    The initial state for momentum
  traceK : float
    Trace of the covariance matrix of the data; i.e., K = E[aa^T]
  t_oracle: callable
    Takes as an argument a jax RNG key and a batch-size.
    Expects in return two tensors (A, y)
    of dimension (batch x data-dimension) and dimension (batch).
  loss: callable
    Takes as an argument a vector of length data-dimension,
    which is the current linear model parameters, and returns the
    loss.
  loss_times: vector
    Iteration counts at which to compute the loss

  Returns
  -------
  losses: vector
    An array of length 'steps' containing the losses
  loss_times: vector
    Iteration counts at which the losses were computed
  x: vector
    The final state of the SGD
  """

  @jax.jit
  def skinny_update(z, things):
    keyz, iteration = things
    A,y = t_oracle(keyz, batch)
    x,w = z
    #delta = theta / (iteration + traceK)
    grad = jnp.tensordot(A,jnp.tensordot(A,x,axes=1)-y,axes=[[0],[0]])
    neww = ( 1.0 - delta(iteration) ) * w + g1(iteration) * grad
    newx = x - g2(iteration) * grad - neww * g3(iteration)
    return (newx,neww), False
  
  loss_times = np.unique(
    np.concatenate(
        (np.int32(np.floor((11/10)**np.arange(1,np.log10(steps)/np.log10(11/10),1))),
        [steps])
        )
    )
  loss_time_diffs = jnp.concatenate([jnp.array([1]), loss_times[1:] - loss_times[:-1]])
  loss_times_shifted = jnp.concatenate([jnp.array([0]), loss_times[:-1]])
  (x,w) = (init_x,init_w)

  mkeyout =  jax.random.split(key,len(loss_time_diffs))
  losses = []
  for outerloop_time, sgdsteps, mkey in zip(loss_times_shifted,loss_time_diffs, mkeyout):
    iterlist = jnp.arange(outerloop_time,outerloop_time+sgdsteps,dtype=jnp.int32)
    mkeys = jax.random.split(mkey,sgdsteps)
    (newx,neww), _ = jax.lax.scan(skinny_update,(x,w),(mkeys,iterlist))
    x = newx
    w = neww
    losses.append(loss(x))

  losses = jnp.concatenate([loss(init_x), jnp.array(losses)])
  loss_times = jnp.concatenate([jnp.array([0]), loss_times])
  return losses, loss_times, x

  # return losses,loss_times
