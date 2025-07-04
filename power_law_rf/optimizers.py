from typing import NamedTuple, Optional, Callable

import jax
import jax.numpy as jnp
import chex

import optax
from optax import tree_utils as otu
from optax._src import base
from optax._src import numerics
from optax._src import utils


def powerlaw_schedule(
    init_value: chex.Scalar,
    saturation_value: chex.Scalar,
    power: chex.Scalar,
    time_scale: chex.Scalar,
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
  dimensions: base.Updates

def dana_optimizer(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """DANA optimizer.

    Args:
        g1: A scalar or schedule determining the first gradient coefficient.
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    def init_fn(params):
        y = otu.tree_zeros_like(params, dtype=y_dtype)  # Momentum
        return DanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, dimensions=y)

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

        return updates, DanaOptimizerState(count=count_inc, y=y, dimensions=state.dimensions)

    return base.GradientTransformation(init_fn, update_fn)

def dana_optimizer_layerwise(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
    ) -> base.GradientTransformation:
    """DANA optimizer, with layerwise dimension scaling.

    This differs from the decaying momentum version, in that each layer is scaled by its dimension.
    Args:
        g1: A scalar or schedule determining the first gradient coefficient.
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)

    def init_fn(params):
        y = otu.tree_zeros_like(params, dtype=y_dtype)  # Momentum
        dimensions = jax.tree.map(lambda x: jnp.float32(len(jnp.reshape(x,-1))), params)
        return DanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, dimensions=dimensions)

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
            lambda m,u,d : newg2*u if m is None else -1.0*(newg2*u + newg3*m/d),
            y,
            updates,
            state.dimensions,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, DanaOptimizerState(count=count_inc, y=y, dimensions=state.dimensions)

    return base.GradientTransformation(init_fn, update_fn)



class TaneaOptimizerState(NamedTuple):
  """State for the Tanea algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  m: base.Updates
  v: base.Updates
  tau: base.Updates


def tanea_optimizer(
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    epsilon: float = 1e-8,
    beta_m : Optional[base.ScalarOrSchedule] = None,
    g1: Optional[base.ScalarOrSchedule] = None,
    beta_v : Optional[base.ScalarOrSchedule] = None,
    magic_tau : float = 1.0,
    wd : Optional[base.ScalarOrSchedule] = None,
    momentum_flavor: str = "effective-clip",
    tau_flavor: str = "second-moment",
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """Tanea optimizer.

    Args:
        g2: A scalar or schedule determining the second gradient coefficient.
        g3: A scalar or schedule determining the third gradient coefficient.
        Delta: A scalar or schedule determining the momentum decay rate.
        epsilon: Small constant for numerical stability.
        beta_m: Optional scalar or schedule for momentum decay rate.
        g1: Optional scalar or schedule for first gradient coefficient.
        beta_v: Optional scalar or schedule for second momentum decay rate.
        magic_tau: Scaling factor for tau updates.
        wd: Optional scalar or schedule for weight decay.
        momentum_flavor: Type of momentum term for g3. Options are "effective-clip" (default) or "theory".
        y_dtype: Optional `dtype` to be used for the momentum accumulator; if
        `None` then the `dtype` is inferred from `params` and `updates`.

    Returns:
        A :class:`optax.GradientTransformation` object.
    """

    y_dtype = utils.canonicalize_dtype(y_dtype)
    if beta_m is None:
        beta_m = Delta
    elif not callable(beta_m):
        beta_m = lambda _: beta_m
    if beta_v is None:
        beta_v = Delta
    elif not callable(beta_v):
        beta_v = lambda _: beta_v
    if g1 is None:
        g1 = lambda _: 1.0
    elif not callable(g1):
        g1 = lambda _: g1
    if wd is None:
        wd = lambda _: 0.0
    elif not callable(wd):
        wd = lambda _: wd

    
    ##  The power of 1.0 ought to correctly initialize the tau estimate (~~tau will be like ~p once 1/t is smaller)
    tau_reg = lambda tau, t : jnp.maximum(tau, jnp.pow(1.0+t,-1.0))
    root_tau_reg = lambda tau, t : jnp.sqrt(tau_reg(tau, t))
    effective_time = lambda tau, t: jnp.maximum(tau*t,1.0)  
    quarter_root_tau_reg = lambda tau, t : jnp.power(tau_reg(tau, t),0.25)



    tau_updater = lambda tau,u,v,t : (u**2)*(root_tau_reg(tau,t)*magic_tau) / ( (u**2)*(root_tau_reg(tau, t)*magic_tau) + v + epsilon**2)
    if tau_flavor == "second-moment":
        tau_updater = lambda tau,u,v,t : (u**2)*(root_tau_reg(tau,t)*magic_tau) / ( (u**2)*(root_tau_reg(tau, t)*magic_tau) + v + epsilon**2)
    elif tau_flavor == "first-moment":
        tau_updater = lambda tau,u,v,t : jnp.abs(u)*(quarter_root_tau_reg(tau,t)*magic_tau) / ( jnp.abs(u*(quarter_root_tau_reg(tau, t)*magic_tau)) + jnp.sqrt(v) + epsilon)
    else:
        raise ValueError(f"Unknown tau_flavor: {tau_flavor}. Must be 'second-moment' or 'first-moment'")

    ## The g3_momentum_term will be used to multiply the first moment estimator $m$ and the schedule.  The standard Adam scaling would simply output 1/(sqrt(v)+epsilon), times a learning rate, which is here g3(effective_time(tau, t)).  
    ## Now, in the sparse-in-time settig, where updates occur with some probability $p$, we ideally have something like $m = p*E(g)$, where $E(g)$ is some partial expectation of the gradient achieved by time averaging.  This $E(g)$ is a 'DANA-type' momentum estimate.  The $v = p*E(g^2)$ with the same sense of partial expectation.  The $tau$ is an approximation of $p$, and $\tau_reg$ stabilizes the estimate.  
    ## Now we want the momentum term to only be updated at the same speed as when the gradient terms are added.  
    ## To accomplish this, we consider the following instantaneous parameter-update-speed rule:
    ## abs(u * sqrt(tau_reg))/( abs(u * sqrt(tau_reg)) + sqrt(v) + epsilon) 
    ## This is similar to what is used to define tau itself.
    ## We then want to normalize the parameter updates, and so we also introduce
    ## sqrt(v/tau_reg) + epsilon/sqrt(tau_reg)
    ## 1. The "theory" version now takes the product of these factors.
    ## 2. The "effective-clip" version uses a more conservative estimate.  In the theory version we can represent the denominator as (a+b)*a, where a = sqrt(v)+epsilon and b = abs(u)*sqrt(tau_reg).  The 'effective-clip' version replaces this by (a+b)**2, which is always larger and moreover, is substantially larger if $b^2 \gg a^2$.  This can occur in settings where individual gradients have relatively heavy tails, in which case we expect the 'effective-clip' version to be more stable.
    ## 3. The "always-on" version allows momentum updates to always occur.  Since $m$ is effectiely scaled by the time-scale $p$, we expect to update (1/p) times between g2 updates.  Hence in mean this should behave the same way as the 'theory' version, but we expect it to be less stable.  This is the same as what is used for the 'g2' pure gradient term.
    ## 4. The "strong-clip" version is similar to the 'effective-clip'
    g3_momentum_term = lambda u, v, tau, t: abs(u)/((u**2) * tau_reg(tau, t)+v+epsilon**2)
    # Create lambda function for g3 momentum term based on flavor
    if momentum_flavor == "effective-clip":
        g3_momentum_term = lambda u, v, tau, t: abs(u)/((u**2) * tau_reg(tau, t)+v+epsilon**2)
    elif momentum_flavor == "theory":
        g3_momentum_term = lambda u, v, tau, t: abs(u)/((jnp.abs(u)*root_tau_reg(tau, t)+jnp.sqrt(v)+epsilon) * (jnp.sqrt(v)+epsilon) )
    elif momentum_flavor == "always-on":
        g3_momentum_term = lambda u, v, tau, t: root_tau_reg(tau, t)/((jnp.sqrt(v)+epsilon))
    elif momentum_flavor == "strong-clip":
        g3_momentum_term = lambda u, v, tau, t: jnp.minimum(abs(u),(jnp.sqrt(v/tau_reg(tau, t))))/(v+epsilon**2)
    else:
        raise ValueError(f"Unknown momentum_flavor: {momentum_flavor}. Must be 'effective-clip' or 'theory'")  

    def init_fn(params):

        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  #Tau
        return TaneaOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

    def update_fn(updates, state, params):

        new_wd = wd(state.count)
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        new_beta_m = beta_m(state.count)
        new_beta_v = beta_v(state.count)

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-new_beta_m) + newg1*u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )
        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-new_beta_v) + new_beta_v*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )

        new_tau = jax.tree.map(
            lambda tau,u,v : None if tau is None else tau*(1-newDelta) + newDelta*tau_updater(tau, u, v, state.count),
            state.tau,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
            )
        

        updates = jax.tree.map(
            lambda m,u,v,tau : -1.0*g2(effective_time(tau, state.count))*u 
            if m is None 
            else -1.0*(g2(effective_time(tau, state.count))*u*root_tau_reg(tau, state.count))/(jnp.sqrt(v)+epsilon)-(g3(effective_time(tau, state.count))*m*g3_momentum_term(u, v, tau, state.count)),
            new_m,
            updates,
            new_v,
            new_tau,
            #jnp.maximum(new_tau, 1.0/(1.0+state.count)),
            is_leaf=lambda x: x is None,
        )

        #Apply weight decay
        updates = jax.tree.map(
            lambda u,p : u+(-1.0*new_wd)*p,
            updates,
            params,
            is_leaf=lambda x: x is None,
        )
        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        new_tau = otu.tree_cast(new_tau, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, TaneaOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

    return base.GradientTransformation(init_fn, update_fn)
