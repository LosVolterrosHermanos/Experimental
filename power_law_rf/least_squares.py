import jax
import jax.numpy as jnp
import jax.random as random
import optax
from tqdm import tqdm

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
        return (params, opt_state), loss
    

    # This generates an exponentially spaced sequence of times at which we record the loss.
    # This is to ensure that we are recording the loss at a reasonable frequency.
    # We start at 0 and go up to the total number of steps.
    # We then add the number of steps to the list.
    # We then take the unique values and sort them.
    # This gives us the times at which we record the loss.
    
    losses=[]
    loss_times = jnp.unique(jnp.concatenate(
        [jnp.array([0]),
        jnp.int32(
            1.1**jnp.arange(1,jnp.ceil(jnp.log(steps)/jnp.log(1.1)))
        ),
        jnp.array([steps])]
    ))

    loss_time_steps = loss_times[1:]-loss_times[:-1]
    losses.append(loss_oracle(init))
    if tqdm_bar:
        for increment in tqdm(loss_time_steps):
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, _ = jax.lax.scan(train_step, state, keyz)
            pop_loss = loss_oracle(state[0])
            losses.append(pop_loss)
    else:
        for increment in loss_time_steps:
            key, subkey = random.split(key)
            keyz = random.split(subkey, increment)
            state, _ = jax.lax.scan(train_step, state, keyz)
            pop_loss = loss_oracle(state[0])
            losses.append(pop_loss)
    return loss_times, losses
