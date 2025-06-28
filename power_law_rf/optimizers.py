from typing import NamedTuple, Optional

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
    *,
    y_dtype: Optional[chex.ArrayDType] = None,
  ) -> base.GradientTransformation:
    """Tanea optimizer.

    Args:
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
        m = otu.tree_zeros_like(params, dtype=y_dtype)  #First-Momentum
        v = otu.tree_zeros_like(params, dtype=y_dtype)  #Second-Momentum
        tau = otu.tree_zeros_like(params, dtype=y_dtype)  #Tau
        return TaneaOptimizerState(count=jnp.zeros([], jnp.int32), m=m, v=v, tau=tau)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)

        new_m = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + u,
            state.m,
            updates,
            is_leaf=lambda x: x is None,
        )
        new_v = jax.tree.map(
            lambda v,u : None if v is None else v*(1-newDelta) + newDelta*(u**2),
            state.v,
            updates,
            is_leaf=lambda x: x is None,
        )
        new_tau = jax.tree.map(
            lambda tau,u,v : None if tau is None else tau*(1-newDelta) + newDelta*jnp.abs(u)/(jnp.sqrt(v+u**2)+epsilon),
            state.tau,
            updates,
            new_v,
            is_leaf=lambda x: x is None,
        )
        newg2 = g2(new_tau*state.count)
        newg3 = g3(new_tau*state.count)
        updates = jax.tree.map(
            lambda m,u,v,tau : -1.0*newg2*u if m is None else -1.0*(newg2*u + newg3*m)*jnp.sqrt(tau)/(jnp.sqrt(v)+epsilon),
            new_m,
            updates,
            new_v,
            new_tau,
            is_leaf=lambda x: x is None,
        )
        new_m = otu.tree_cast(new_m, y_dtype)
        new_v = otu.tree_cast(new_v, y_dtype)
        new_tau = otu.tree_cast(new_tau, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, TaneaOptimizerState(count=count_inc, m=new_m, v=new_v, tau=new_tau)

    return base.GradientTransformation(init_fn, update_fn)
