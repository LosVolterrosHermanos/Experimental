import jax
import jax.numpy as jnp
import jax.random as random
import optax
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, Dict, List
import tqdm
from functools import partial
import numpy as np

class PowerLawRandomFeatures:
    """Clean implementation of Power-Law Random Features model.

    The model generates synthetic regression problems with power-law decaying
    eigenvalues and target coefficients.
    """

    def __init__(self, alpha: float, beta: float, v: int, d: int, key: random.PRNGKey):
        """Initialize the PLRF model.

        Args:
            alpha: Power law exponent for eigenvalue decay
            beta: Power law exponent for target coefficient decay
            v: Hidden dimension (number of random features)
            d: Embedded dimension (parameter dimension)
            key: JAX random key
        """
        self.alpha = alpha
        self.beta = beta
        self.v = v
        self.d = d

        # Initialize random features matrix W with variance 1/d
        self.W = random.normal(key, (v, d)) / jnp.sqrt(d)

        # Compute power-law eigenvalues and coefficients
        indices = jnp.arange(1, v + 1)
        self.eigenvalues = indices ** (-alpha)
        self.b = indices ** (-beta)

        # Precompute scaled quantities for efficiency
        self.checkW = self.W * self.eigenvalues.reshape(-1, 1)  # (v, d)
        self.checkb = indices ** (-alpha - beta)  # (v,)

    def generate_batch(self, key: random.PRNGKey, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Generate a batch of (X, y) training data.

        Args:
            key: JAX random key
            batch_size: Number of samples to generate

        Returns:
            X: Input features of shape (batch_size, d)
            y: Target values of shape (batch_size,)
        """
        # Generate random features x ~ N(0, 1)
        x = random.normal(key, (batch_size, self.v))

        # Transform to get inputs and targets
        X = jnp.matmul(x, self.checkW)  # (batch_size, d)
        y = jnp.matmul(x, self.checkb)  # (batch_size,)

        return X, y

    def population_risk(self, params: jnp.ndarray) -> float:
        """Compute exact population risk for given parameters.

        The population risk is E[(y - f(x))^2] / 2 where f(x) = <W^T x, params>.

        Args:
            params: Parameter vector of shape (d,)

        Returns:
            Population risk value
        """
        # Project parameters onto random features
        proj = jnp.matmul(self.checkW, params)  # (v,)

        # Compute squared error
        risk = jnp.sum((proj - self.checkb) ** 2)

        # Return risk divided by 2 (matching the original implementation)
        return risk / 2

    def optimal_params(self) -> jnp.ndarray:
        """Compute optimal parameters via closed form solution.

        Returns:
            Optimal parameter vector of shape (d,)
        """
        # Solve normal equations: (W^T W) params = W^T b
        W_T_W = jnp.matmul(self.checkW.T, self.checkW)
        W_T_b = jnp.matmul(self.checkW.T, self.checkb)

        # Solve the linear system
        params_opt = jnp.linalg.solve(W_T_W, W_T_b)

        return params_opt

    def optimal_risk(self) -> float:
        """Compute the population risk at the optimal parameters.

        Returns:
            Minimum achievable population risk
        """
        params_opt = self.optimal_params()
        return self.population_risk(params_opt)


class PLRFTrainer:
    """Handles training of PLRF models with various optimizers."""

    def __init__(self, model: PowerLawRandomFeatures, optimizer: optax.GradientTransformation):
        """Initialize the trainer.

        Args:
            model: PowerLawRandomFeatures instance
            optimizer: Optax optimizer
        """
        self.model = model
        self.optimizer = optimizer

    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[jnp.ndarray] = None,
              eval_freq: Optional[int] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Train the model and return loss curve.

        Args:
            key: JAX random key
            num_steps: Number of optimization steps
            batch_size: Batch size for SGD
            init_params: Initial parameters (default: zeros)
            eval_freq: Frequency of evaluation (default: exponentially spaced)

        Returns:
            timestamps: Array of step numbers where loss was evaluated
            losses: Array of population losses at those steps
        """
        # Initialize parameters
        if init_params is None:
            init_params = jnp.zeros(self.model.d)

        params = init_params
        opt_state = self.optimizer.init(params)

        # Determine evaluation times (exponentially spaced by default)
        if eval_freq is None:
            # Create exponentially spaced evaluation times
            eval_times = jnp.unique(jnp.concatenate([
                jnp.array([0]),
                jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
                jnp.array([num_steps])
            ]))
        else:
            eval_times = jnp.arange(0, num_steps + 1, eval_freq)

        # Batch loss function for training
        @jax.jit
        def batch_loss(params, X, y):
            """Compute mean squared error loss on a batch."""
            y_pred = jnp.matmul(X, params)
            return jnp.mean(optax.l2_loss(y_pred, y))

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single SGD step."""
            # Generate batch
            X, y = self.model.generate_batch(key, batch_size)

            # Compute loss and gradients
            loss, grads = jax.value_and_grad(batch_loss)(params, X, y)

            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            return params, opt_state, loss

        # Training loop with progress bar
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm.tqdm(range(num_steps)):
            # Split key for this step
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state, batch_loss_val = train_step(params, opt_state, subkey)

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        return jnp.array(timestamps), jnp.array(losses)


class MixtureOfExpertsPLRF(PowerLawRandomFeatures):
    """Mixture of Experts extension of the Power-Law Random Features model.

    Each expert shares the same random features matrix W but has its own
    parameter vector θ^(i). Expert selection follows a power-law distribution.
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 v: int,
                 d: int,
                 m: int,
                 zeta: float,
                 key: random.PRNGKey):
        """Initialize the MoE PLRF model.

        Args:
            alpha: Power law exponent for eigenvalue decay
            beta: Power law exponent for target coefficient decay
            v: Hidden dimension (number of random features)
            d: Embedded dimension (parameter dimension)
            m: Number of experts
            zeta: Power law exponent for expert selection (p(i) ∝ i^(-zeta))
            key: JAX random key
        """
        # Initialize base PLRF model
        super().__init__(alpha, beta, v, d, key)

        self.m = m
        self.zeta = zeta

        # Compute expert selection probabilities: p(i) ∝ i^(-zeta)
        expert_indices = jnp.arange(1, m + 1)
        unnormalized_probs = expert_indices ** (-zeta)
        self.expert_probs = unnormalized_probs / jnp.sum(unnormalized_probs)

    def sample_expert(self, key: random.PRNGKey) -> int:
        """Sample an expert according to the power-law distribution.

        Args:
            key: JAX random key

        Returns:
            Expert index (0-based)
        """
        return random.choice(key, self.m, p=self.expert_probs)

    def sample_expert_batch(self, key: random.PRNGKey, batch_size: int) -> jnp.ndarray:
        """Sample experts for a batch of samples.

        Args:
            key: JAX random key
            batch_size: Number of experts to sample

        Returns:
            Array of expert indices of shape (batch_size,)
        """
        return random.choice(key, self.m, shape=(batch_size,), p=self.expert_probs)

    def create_routing_matrix(self, expert_indices: jnp.ndarray, batch_size: int) -> jnp.ndarray:
        """Create binary routing matrix R from expert indices.

        Args:
            expert_indices: Array of expert indices for each sample
            batch_size: Number of samples

        Returns:
            Routing matrix R of shape (m, batch_size)
        """
        # Create one-hot encoding for each sample
        R = jax.nn.one_hot(expert_indices, self.m).T  # (m, batch_size)
        return R

    @partial(jax.jit, static_argnums=(0,))
    def mixture_population_risk(self, params: jnp.ndarray) -> float:
        """Compute exact population risk for MoE model efficiently (jittable).

        Vectorized computation that avoids loops over experts.

        Args:
            params: Parameter matrix of shape (d, m) where column i contains
                   parameters for expert i

        Returns:
            Population risk value
        """
        # Project all experts onto random features at once
        # checkW is (v, d), params is (d, m) -> proj is (v, m)
        proj = jnp.matmul(self.checkW, params)  # (v, m)
        
        # Broadcast checkb to match proj shape: (v, 1)
        checkb_broadcast = self.checkb.reshape(-1, 1)  # (v, 1)
        
        # Compute squared errors for all experts at once: (v, m)
        squared_errors = (proj - checkb_broadcast) ** 2
        
        # Sum over features for each expert: (m,)
        expert_risks = jnp.sum(squared_errors, axis=0) / 2
        
        # Weight by expert probabilities and sum
        total_risk = jnp.sum(self.expert_probs * expert_risks)
        
        return total_risk

    @partial(jax.jit, static_argnums=(0,))
    def per_expert_population_risk(self, params: jnp.ndarray) -> jnp.ndarray:
        """Compute population risk for each expert efficiently (jittable).

        Args:
            params: Parameter matrix of shape (d, m) where column i contains
                   parameters for expert i

        Returns:
            Array of population risk values, one for each expert (shape: m,)
        """
        # Project all experts onto random features at once
        # checkW is (v, d), params is (d, m) -> proj is (v, m)
        proj = jnp.matmul(self.checkW, params)  # (v, m)
        
        # Broadcast checkb to match proj shape: (v, 1)
        checkb_broadcast = self.checkb.reshape(-1, 1)  # (v, 1)
        
        # Compute squared errors for all experts at once: (v, m)
        squared_errors = (proj - checkb_broadcast) ** 2
        
        # Sum over features for each expert and divide by 2: (m,)
        expert_risks = jnp.sum(squared_errors, axis=0) / 2
        
        return expert_risks

    def population_risk(self, params: jnp.ndarray) -> float:
        """Compute exact population risk for MoE model.

        For MoE, the population risk is the expected risk over expert selection.

        Args:
            params: Parameter matrix of shape (d, m) where column i contains
                   parameters for expert i

        Returns:
            Population risk value
        """
        if params.ndim == 1:
            # If given a single parameter vector, treat it as single expert
            return super().population_risk(params)

        # Use the efficient vectorized implementation
        return self.mixture_population_risk(params)

    def optimal_params_per_expert(self) -> jnp.ndarray:
        """Compute optimal parameters for each expert independently.

        In the basic MoE model, each expert solves the same problem,
        so they all have the same optimal parameters.

        Returns:
            Parameter matrix of shape (d, m)
        """
        # Get single expert optimal params
        single_expert_params = self.optimal_params()

        # Replicate for all experts
        return jnp.tile(single_expert_params.reshape(-1, 1), (1, self.m))


class TwoExpertPLRF(MixtureOfExpertsPLRF):
    """Special case of MoE PLRF with exactly two experts and fixed probabilities.

    Expert 1 is selected with probability (1-p)
    Expert 2 is selected with probability p
    """

    def __init__(self,
                 alpha: float,
                 beta: float,
                 v: int,
                 d: int,
                 p: float,
                 key: random.PRNGKey):
        """Initialize the two-expert PLRF model.

        Args:
            alpha: Power law exponent for eigenvalue decay
            beta: Power law exponent for target coefficient decay
            v: Hidden dimension (number of random features)
            d: Embedded dimension (parameter dimension)
            p: Probability of selecting expert 2
            key: JAX random key
        """
        # Initialize with m=2 experts
        # zeta doesn't matter since we'll override the probabilities
        super().__init__(alpha, beta, v, d, m=2, zeta=1.0, key=key)

        self.p = p
        # Override expert probabilities
        self.expert_probs = jnp.array([1 - p, p])


class MoEPLRFTrainer(PLRFTrainer):
    """Trainer for Mixture of Experts PLRF models."""

    def __init__(self,
                 model: Union[MixtureOfExpertsPLRF, TwoExpertPLRF],
                 optimizer: optax.GradientTransformation,
                 per_expert_updates: bool = False):
        """Initialize the MoE trainer.

        Args:
            model: MoE PLRF model instance
            optimizer: Optax optimizer
            per_expert_updates: If True, track update counts per expert
        """
        self.model = model
        self.optimizer = optimizer
        self.per_expert_updates = per_expert_updates

    def train(self,
              key: random.PRNGKey,
              num_steps: int,
              batch_size: int,
              init_params: Optional[jnp.ndarray] = None,
              eval_freq: Optional[int] = None,
              track_per_expert_loss: bool = False,
              track_update_history: bool = False) -> Dict:
        """Train the MoE model and return training metrics.

        Args:
            key: JAX random key
            num_steps: Number of optimization steps
            batch_size: Batch size for SGD
            init_params: Initial parameters of shape (d, m)
            eval_freq: Frequency of evaluation
            track_per_expert_loss: If True, track loss per expert
            track_update_history: If True, track update/sample counts over time

        Returns:
            Dictionary containing:
                - timestamps: Evaluation timestamps
                - losses: Population losses
                - per_expert_losses: Loss per expert (if tracked)
                - expert_update_counts: Total number of updates per expert
                - expert_sample_counts: Total number of samples per expert
                - update_history: History of counts at eval times (if tracked)
        """
        # Initialize parameters for all experts
        if init_params is None:
            init_params = jnp.zeros((self.model.d, self.model.m))

        params = init_params
        opt_state = self.optimizer.init(params)

        # Determine evaluation times
        if eval_freq is None:
            eval_times = jnp.unique(jnp.concatenate([
                jnp.array([0]),
                jnp.int32(1.1 ** jnp.arange(1, jnp.ceil(jnp.log(num_steps) / jnp.log(1.1)))),
                jnp.array([num_steps])
            ]))
        else:
            eval_times = jnp.arange(0, num_steps + 1, eval_freq)

        # Batch loss function for MoE
        @jax.jit
        def batch_loss_moe(params, X, y, expert_indices):
            """Compute mean squared error loss with expert routing."""
            # Create routing matrix
            R = self.model.create_routing_matrix(expert_indices, batch_size)

            # Compute predictions for all experts
            all_predictions = jnp.matmul(X, params)  # (batch_size, m)

            # Select predictions based on routing
            # For each sample, select the prediction from its assigned expert
            predictions = jnp.sum(all_predictions * R.T, axis=1)  # (batch_size,)

            # Compute loss
            return jnp.mean(optax.l2_loss(predictions, y))

        # Gradient computation for MoE
        @jax.jit
        def compute_moe_gradients(params, X, y, expert_indices):
            """Compute gradients with proper expert routing."""
            # Create routing matrix
            R = self.model.create_routing_matrix(expert_indices, batch_size)

            # Count samples per expert
            samples_per_expert = jnp.sum(R, axis=1)  # (m,)

            # Function to compute loss for gradient
            # Loss averages over the batch size B (not num samples per expert)
            def loss_fn(params):
                all_predictions = jnp.matmul(X, params)  # (batch_size, m)
                predictions = jnp.sum(all_predictions * R.T, axis=1)
                return jnp.mean(optax.l2_loss(predictions, y))

            # Get gradients
            grads = jax.grad(loss_fn)(params)

            return grads, samples_per_expert

        # Training step
        @jax.jit
        def train_step(params, opt_state, key):
            """Single MoE SGD step."""
            # Split keys
            key_data, key_expert = random.split(key)

            # Generate batch and sample experts
            X, y = self.model.generate_batch(key_data, batch_size)
            expert_indices = self.model.sample_expert_batch(key_expert, batch_size)

            # Compute gradients with routing
            grads, samples_per_expert = compute_moe_gradients(params, X, y, expert_indices)

            # Update parameters
            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)

            # Compute batch loss for monitoring
            batch_loss_val = batch_loss_moe(params, X, y, expert_indices)

            return params, opt_state, batch_loss_val, samples_per_expert

        # Training loop
        losses = [self.model.population_risk(init_params)]
        timestamps = [0]
        expert_update_counts = jnp.zeros(self.model.m)
        expert_sample_counts = jnp.zeros(self.model.m)  # Track total samples per expert

        # Optionally track per-expert losses
        if track_per_expert_loss:
            # Use vectorized per-expert risk computation and store as arrays
            initial_per_expert_risks = self.model.per_expert_population_risk(init_params)
            per_expert_losses = [initial_per_expert_risks]

        # Optionally track update history
        if track_update_history:
            update_history = {
                'timestamps': [0],
                'update_counts': [expert_update_counts.copy()],
                'sample_counts': [expert_sample_counts.copy()]
            }

        eval_idx = 1
        next_eval = eval_times[eval_idx] if eval_idx < len(eval_times) else num_steps + 1

        for step in tqdm.tqdm(range(num_steps)):
            # Split key
            key, subkey = random.split(key)

            # Perform training step
            params, opt_state, batch_loss_val, samples_per_expert = train_step(params, opt_state, subkey)

            # Update expert counts
            expert_update_counts += (samples_per_expert > 0).astype(jnp.float32)
            expert_sample_counts += samples_per_expert

            # Evaluate if needed
            if step + 1 == next_eval:
                pop_risk = self.model.population_risk(params)
                losses.append(pop_risk)
                timestamps.append(step + 1)

                if track_per_expert_loss:
                    # Use vectorized per-expert risk computation
                    current_per_expert_risks = self.model.per_expert_population_risk(params)
                    per_expert_losses.append(current_per_expert_risks)

                if track_update_history:
                    update_history['timestamps'].append(step + 1)
                    update_history['update_counts'].append(expert_update_counts.copy())
                    update_history['sample_counts'].append(expert_sample_counts.copy())

                eval_idx += 1
                if eval_idx < len(eval_times):
                    next_eval = eval_times[eval_idx]
                else:
                    next_eval = num_steps + 1

        # Prepare results
        results = {
            'timestamps': jnp.array(timestamps),
            'losses': jnp.array(losses),
            'expert_update_counts': expert_update_counts,
            'expert_sample_counts': expert_sample_counts,
            'final_params': params
        }

        if track_per_expert_loss:
            # Convert list of arrays to a single array of shape (n_eval_times, m)
            results['per_expert_losses'] = jnp.array(per_expert_losses)

        if track_update_history:
            results['update_history'] = {
                'timestamps': jnp.array(update_history['timestamps']),
                'update_counts': jnp.array(update_history['update_counts']),
                'sample_counts': jnp.array(update_history['sample_counts'])
            }

        return results
