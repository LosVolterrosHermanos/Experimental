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

class adanaOptimizerState(NamedTuple):
  """State for the Dana algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  y: base.Updates
  nu: base.Updates

def adana_optimizer(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    kappa4: Callable[[chex.Array], chex.Array],
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
        y = otu.tree_zeros_like(params, dtype=y_dtype)
        nu = otu.tree_zeros_like(params)   #second momen
        return adanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        newg2 = g2(state.count)
        newg3 = g3(state.count)
        nu = jax.tree.map(
            lambda m, u : None if m is None else m+u**2,
            state.nu,
            updates,
            is_leaf=lambda x: x is None,
        )
        nu = otu.tree_cast(nu, y_dtype)

        y = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + newg1*u,
            state.y,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m,u,n : -1.0*newg2*jnp.sign(u) if m is None else -1.0*(newg2*jnp.sign(u) + newg3*m*kappa4(n)),
            y,
            updates,
            nu,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, adanaOptimizerState(count=count_inc, y=y, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)

class adanaOptimizerState(NamedTuple):
  """State for the Dana algorithm."""
  count: chex.Array  # shape=(), dtype=jnp.int32.
  y: base.Updates
  nu: base.Updates

def adana_optimizer_mk2(
    g1: base.ScalarOrSchedule,
    g2: base.ScalarOrSchedule,
    g3: base.ScalarOrSchedule,
    Delta: base.ScalarOrSchedule,
    kappa4: Callable[[chex.Array], chex.Array],
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
        y = otu.tree_zeros_like(params, dtype=y_dtype)
        nu = otu.tree_zeros_like(params)   #second momen
        return adanaOptimizerState(count=jnp.zeros([], jnp.int32), y=y, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        newDelta = Delta(state.count)
        newg1 = g1(state.count)
        newg2 = g2(state.count)
        newg3 = g3(state.count)

        nu = jax.tree.map(
            lambda m, u : None if m is None else m*(1-newDelta)+u**2,
            state.nu,
            updates,
            is_leaf=lambda x: x is None,
        )
        nu = otu.tree_cast(nu, y_dtype)

        y = jax.tree.map(
            lambda m,u : None if m is None else m*(1-newDelta) + newg1*u,
            state.y,
            updates,
            is_leaf=lambda x: x is None,
        )
        updates = jax.tree.map(
            lambda m,u,n : -1.0*newg2*u/jnp.sqrt(1E-8+0.01*u**2 + newDelta*n) if m is None else 
                -1.0*(newg2*u/jnp.sqrt(1E-8+0.01*u**2 + newDelta*n) + newg3*newDelta*(m/jnp.sqrt(1E-8+0.01*u**2 + newDelta*n))*kappa4(n/(u**2+1E-8))),
            y,
            updates,
            nu,
            is_leaf=lambda x: x is None,
        )
        y = otu.tree_cast(y, y_dtype)
        count_inc = numerics.safe_increment(state.count)

        return updates, adanaOptimizerState(count=count_inc, y=y, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)