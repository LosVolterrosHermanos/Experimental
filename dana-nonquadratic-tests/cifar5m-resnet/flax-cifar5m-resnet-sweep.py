import tensorflow as tf

tf.config.set_visible_devices([], 'GPU')
tf.random.set_seed(0)  # Set the random seed for reproducibility.

import jax
from jax.tree_util import tree_leaves, tree_map
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
from functools import partial
import glob
from tqdm import tqdm

#from flaxresnets import ResNet18
from typing import Sequence
from typing import Dict, Any
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats

import pickle


#From LosVolterros
import sys
sys.path.append('../../')
import optimizers




# Configuration
BATCH_SIZE = 128
TRAIN_STEPS = 5000*20
POPULATION_TRACE = 4.0
#NUM_FILTERS = 32 
CHUNK_SIZE = 10000  # Number of examples to load at once
BASE_PATH = '/home/elliotpaquette/Documents/SGD/cifar5m'

TRAIN_FILES = sorted(glob.glob(f"{BASE_PATH}/cifar5m_part[0-3]*.npz"))
VAL_FILE = f"{BASE_PATH}/cifar5m_part4.npz"




def count_parameters(model: nnx.Module):
    """
    Computes the number of parameters in a Flax NNX model.
    
    Args:
        model: An nnx.Module instance
        
    Returns:
        An integer representing the total number of parameters in the model.
    """
    # Get the state dict from the model
    state = nnx.state(model)
    return np.sum([np.prod(x.shape) for x in tree_leaves(state)])



class ResNetSoftSignBlock(nnx.Module):
    """
    A ResNet block implementation that uses SoftSign activation function instead of the traditional ReLU.
    
    This block consists of two convolutional layers with SoftSign activations and implements a residual
    connection. If the input and output dimensions differ or if strides are not (1,1), a projection
    convolution is applied to the residual path to match dimensions.
    
    Attributes:
        strides: Tuple of integers specifying the stride length for the first convolution
        conv1: First convolutional layer
        conv2: Second convolutional layer
        conv_proj: Optional projection convolution for the residual path
    """
    def __init__(self, in_features: int, out_features: int, strides: tuple[int, int], rngs: nnx.Rngs):
        super().__init__()
        self.strides = strides
        
        # Define the layers with their initializers
        self.conv1 = nnx.Conv(
            in_features=in_features,
            out_features=out_features,
            kernel_size=(3, 3),
            strides=self.strides,
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs
        )
        self.conv2 = nnx.Conv(
            in_features=out_features,
            out_features=out_features,
            kernel_size=(3, 3),
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs
        )
        
        # Projection layers for residual (only if shapes differ)
        if in_features != out_features or strides != (1, 1):
            self.conv_proj = nnx.Conv(
                in_features=in_features,
                out_features=out_features,
                kernel_size=(1, 1),
                strides=self.strides,
                use_bias=True,
                kernel_init=nnx.initializers.lecun_normal(),
                rngs=rngs
            )
        
    def __call__(self, x, train: bool = True):
        residual = x
        
        # First conv block
        x = self.conv1(x)
        x = nnx.soft_sign(x)
        
        # Second conv block
        x = self.conv2(x)
        x = nnx.soft_sign(x)
        
        # Adjust residual if needed
        if hasattr(self, 'conv_proj'):
            residual = self.conv_proj(residual)
            residual = nnx.soft_sign(residual)
        
        return nnx.relu(residual + x)

class ResNetSoftSign(nnx.Module):
    """
    A ResNet model implementation that uses SoftSign activation function instead of the traditional ReLU (AND NO BATCH NORM).
    
    This model consists of an initial convolutional layer followed by a series of ResNet blocks with SoftSign activations.
    The final layer is a dense layer for classification.
    """
    def __init__(self, stage_sizes: Sequence[int], num_classes: int, num_filters: int, rngs: nnx.Rngs):
        super().__init__()
        
        # Initial conv block
        self.conv_init = nnx.Conv(
            in_features=3,
            out_features=num_filters,
            kernel_size=(7, 7),
            strides=(2, 2),
            padding=[(3, 3), (3, 3)],
            use_bias=True,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs
        )        
        # Residual blocks
        self.blocks = []
        in_features = num_filters
        for i, block_size in enumerate(stage_sizes):
            out_features = num_filters * 2**i
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                block = ResNetSoftSignBlock(
                    in_features=in_features,
                    out_features=out_features,
                    strides=strides,
                    rngs=rngs
                )
                self.blocks.append(block)
                in_features = out_features
        
        # Final dense layer
        final_features = num_filters * 2**(len(stage_sizes)-1)
        self.linear = nnx.Linear(
            in_features=final_features,
            out_features=num_classes,
            kernel_init=nnx.initializers.lecun_normal(),
            rngs=rngs
        )
    
    def __call__(self, x, train: bool = True):
        # Initial conv block
        x = self.conv_init(x)
        x = nnx.soft_sign(x)
        x = nnx.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        
        # Residual blocks
        for block in self.blocks:
            x = block(x, train)
        
        # Final layers
        x = jnp.mean(x, axis=(1, 2))
        x = self.linear(x)
        return x

class ChunkedNpzDataset:

    """
    A dataset class for loading and processing CIFAR-5M data in chunks.
    
    This class provides a way to load CIFAR-5M data from a .npz file in memory-mapped mode,
    and apply optional data augmentation to the images. The data is then batched and yielded in chunks.
    """
    def __init__(self, file_path, batch_size, chunk_size=10000, apply_augmentation=True):
        self.npz = np.load(file_path, mmap_mode='r')  # Memory-mapped mode
        self.images = self.npz['X']
        self.labels = self.npz['Y']
        self.num_examples = len(self.labels)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.apply_augmentation = apply_augmentation
        
    def __iter__(self):
        # Calculate number of chunks
        num_chunks = (self.num_examples + self.chunk_size - 1) // self.chunk_size
        
        for chunk_idx in range(num_chunks):
            # Get sequential indices for this chunk
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, self.num_examples)
            
            # Load chunk into memory and normalize
            chunk_images = self.images[start_idx:end_idx].astype(np.float32) / 255.0
            chunk_labels = self.labels[start_idx:end_idx]
            
            # Create dataset for this chunk
            dataset = tf.data.Dataset.from_tensor_slices({
                'image': chunk_images,
                'label': chunk_labels
            })
            
            # Apply data augmentation if enabled
            if self.apply_augmentation:
                dataset = dataset.map(
                    self._augment_data,
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.batch(self.batch_size)
            
            # Yield batches from this chunk
            for batch in dataset:
                yield {k: v.numpy() for k, v in batch.items()}
    
    def _augment_data(self, data):
        """Apply standard data augmentation to images."""
        image = data['image']
        label = data['label']
        
        # Random horizontal flip
        image = tf.image.random_flip_left_right(image)
        
        # Random crop and resize back to original size
        image = tf.image.resize_with_crop_or_pad(image, 40, 40)  # Pad to larger size
        image = tf.image.random_crop(image, [32, 32, 3])  # Random crop back to original
        
        # Random 90-degree rotation
        k = tf.random.uniform([], 0, 4, dtype=tf.int32)  # Random integer from [0, 1, 2, 3]
        image = tf.image.rot90(image, k)  # Rotate by k*90 degrees
        
        # Random brightness, contrast, and saturation
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # Ensure values stay in valid range [0,1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return {'image': image, 'label': label}
        
    def __len__(self):
        return self.num_examples // self.batch_size

def run_resnet_loop(model, optax_optimizer, val_dataset, train_steps, losscomparison=None, store_sigmoid_sum=True):
    """
    Main training loop for a ResNet model using the specified optimizer.
    
    This function orchestrates the training process, including loss computation, gradient calculation,
    parameter updates, and validation. It also handles optional loss comparison and sigmoid sum tracking.
    """
    metrics = nnx.MultiMetric(
    accuracy=nnx.metrics.Accuracy(),
    loss=nnx.metrics.Average('loss'),
    grad_norm=nnx.metrics.Average('grad_norm'),
    sigmoid_sum=nnx.metrics.Average('sigmoid_sum'),
    )

    metrics_history = {
        'train_loss': [],
        'train_accuracy': [],
        'train_grad_norm': [],  
        'train_sigmoid_sum': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_grad_norm': [],   
        'val_sigmoid_sum': [],
    }

    optimizer = nnx.Optimizer(model, optax_optimizer)


    def loss_fn(model: ResNetSoftSign, batch):
        logits = model(batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch['label']
        ).mean()
        return loss, logits

    @nnx.jit
    def train_step(model: ResNetSoftSign, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
        """Train for a single step."""
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        
        # Compute gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads)))

        if store_sigmoid_sum:
            sigmoid_sum = optimizer.opt_state.sigmoid_sum
        else:
            sigmoid_sum = 0.0
        metrics.update(loss=loss, logits=logits, labels=batch['label'], grad_norm=grad_norm, sigmoid_sum=sigmoid_sum)  # In-place updates.
        optimizer.update(grads)  # In-place updates.

    @nnx.jit
    def eval_step(model: ResNetSoftSign, metrics: nnx.MultiMetric, batch):
        grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(model, batch)
        
        # Compute gradient norm
        grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in tree_leaves(grads)))
        
        metrics.update(loss=loss, logits=logits, labels=batch['label'], grad_norm=grad_norm, sigmoid_sum=0.0)  # In-place updates.

    # Training loop
    step = 0
    loss_times = jnp.unique(jnp.concatenate(
        [jnp.array([0]),
        jnp.int32(
            1.1**jnp.arange(1,jnp.ceil(jnp.log(train_steps)/jnp.log(1.1)))
        ),
        jnp.array([train_steps])]
        ))

    losscomp = None
    if losscomparison is not None:
        losscomp = iter(losscomparison['train_loss'])
    
    #with tqdm(total=train_steps, initial=step, desc='Training') as pbar:
    while step < train_steps:
    # Cycle through training files
        for train_file in TRAIN_FILES:
            train_dataset = ChunkedNpzDataset(train_file, BATCH_SIZE, CHUNK_SIZE)
            
            for batch in train_dataset:
                if step >= train_steps:
                    break
                    
                train_step(model, optimizer, metrics, batch)
                
                #f step > 0 and (step in loss_times):
                if step in loss_times:
                    # Log training metrics
                    for metric, value in metrics.compute().items():
                        metrics_history[f'train_{metric}'].append(value)
                    metrics.reset()
                    
                    for val_batch in val_dataset:
                        eval_step(model, metrics, val_batch)
                    
                    for metric, value in metrics.compute().items():
                        metrics_history[f'val_{metric}'].append(value)
                    metrics.reset()
                    
                    if losscomparison is not None:
                        loss_diff = metrics_history['train_loss'][-1] / next(losscomp)
                        color = "\033[31m" if loss_diff >= 1.0 else "\033[32m"  # Red if sgd-favoured, green else
                        print(
                            f"[train] step: {step}, "
                            f"loss-ratio: {color}{loss_diff}\033[0m, "
                            f"sigmoid_sum: {metrics_history['train_sigmoid_sum'][-1]}, "
                        )
                        # print(
                        #     f"[val] step: {step}, "
                        #     f"loss: {metrics_history['val_loss'][-1]}, "
                        #     f"accuracy: {metrics_history['val_accuracy'][-1] * 100}%, "
                        #     f"grad_norm: {metrics_history['val_grad_norm'][-1]:.6f}"    # Add this line
                        # )
                    else:
                        print(
                            f"[train] step: {step}, "
                            f"loss: {metrics_history['train_loss'][-1]}, "
                            f"accuracy: {metrics_history['train_accuracy'][-1] * 100}%, "
                            f"grad_norm: {metrics_history['train_grad_norm'][-1]:.6f}"  # Add this line
                        )
                        print(
                            f"[val] step: {step}, "
                            f"loss: {metrics_history['val_loss'][-1]}, "
                            f"accuracy: {metrics_history['val_accuracy'][-1] * 100}%, "
                            f"grad_norm: {metrics_history['val_grad_norm'][-1]:.6f}"    # Add this line
                        )
                    
                step += 1
                if step >= train_steps:
                    break
                del batch
            del train_dataset
    return loss_times, metrics_history, count_parameters(model)

# # Compute validation metrics on a chunk of validation data
# val_dataset = ChunkedNpzDataset(VAL_FILE, BATCH_SIZE, CHUNK_SIZE, apply_augmentation=True)
# subval_dataset = list(val_dataset)[:10]  # Only use first 10 batches for validation
# del val_dataset

# population_trace = 4.0 #this number was based on earlier simulations but is essentially arbitrary
# est_alpha = 0.66
# est_beta = 1.0
# g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
# g2 = optimizers.powerlaw_schedule(0.4/population_trace, 0.0, 0.0, 1)
# delta = 8
# Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta)

# g3_iv = 2.0/population_trace
# g3_sv = 0.0
# g3_p = -2.0
# g3_ts = 1.0
# g3 = optimizers.powerlaw_schedule(g3_iv, g3_sv, g3_p, g3_ts)
# g4 = optimizers.powerlaw_schedule(jnp.sqrt(1.0/delta), 0.0, 0.5, 1.0)
# thresholder = lambda d: jnp.sqrt(1.0) #UNUSED

# dananormalizedopt = optimizers.dana_optimizer_normalized_mk3(g1=g1,g2=g2,g3=g3,g4=g4,Delta=Delta)


# # Run SGD or load SGD results
# sgd_metrics_filename = f"sgd_metrics_history_steps_{TRAIN_STEPS}_filters_{NUM_FILTERS}.pkl"
# #The below generates SGD results
# try:
#     sgd_metrics_history = pickle.load(open(sgd_metrics_filename, "rb"))
# except (FileNotFoundError, IOError):
#     # If the file doesn't exist, the SGD code below will run and create it
#     sgd_metrics_history = None
#     # Reset model with same initialization
#     rngs = nnx.Rngs(0)  # Use same seed for fair comparison
#     model = ResNetSoftSign(stage_sizes=[2, 2, 2, 2], num_classes=10, num_filters=NUM_FILTERS, rngs=rngs)

#     learning_rate = 0.1
#     momentum = 0.0

#     # Create learning rate schedule that decays over TRAIN_STEPS
#     schedule = optax.linear_schedule(
#         init_value=learning_rate,
#         end_value=0.0,
#         transition_steps=TRAIN_STEPS
#     )
#     sgd = optax.sgd(schedule, momentum)


#     sgd_loss_times, sgd_metrics_history, _ = run_resnet_loop(model, sgd, subval_dataset, TRAIN_STEPS,store_sigmoid_sum=False)
#     # Save the metrics history dictionary to a file
#     with open(sgd_metrics_filename, 'wb') as f:
#         pickle.dump(sgd_metrics_history, f)
#     print(f"SGD metrics history saved to {sgd_metrics_filename}")

# # RUN DANA
# # Create model with rngs
# rngs = nnx.Rngs(0)  # Initialize with a seed
# model = ResNetSoftSign(stage_sizes=[2, 2, 2, 2], num_classes=10, num_filters=NUM_FILTERS, rngs=rngs)
# #model = ResNetSoftSign(stage_sizes=[3, 4, 6, 3], num_classes=10, num_filters=64, rngs=rngs)

# # Run with DANA optimizer
# dana_loss_times, dana_metrics_history, dimension = run_resnet_loop(model, dananormalizedopt, subval_dataset, TRAIN_STEPS,losscomparison=sgd_metrics_history)

# # Save DANA results
# # Note that total parameters = 44668662*(NUM_FILTERS**2)/(128**2)
# dana_mk4_metrics_filename = f"dana_normalized_mk3_metrics_history_steps_{TRAIN_STEPS}_filters_{NUM_FILTERS}_g3_iv={g3_iv}_sv={g3_sv}_p={g3_p}_ts={g3_ts}_delta={delta}.pkl"
# with open(dana_mk4_metrics_filename, 'wb') as f:
#     pickle.dump(dana_metrics_history, f)

# print(f"DANA metrics history saved to {dana_mk4_metrics_filename}")

# # Plot loss and accuracy in subplots
# fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
# ax1.set_title('Loss')
# # Create a second y-axis for sigmoid-sum
# ax1_right = ax1.twinx()
# ax1_right.set_yscale('log')
# ax1_right.set_ylabel('Sigmoid Sum', color='grey')

# # Plot DANA results
# for dataset in ('train', 'val'):
#     flopscalar = BATCH_SIZE * dimension
#     x_values = flopscalar * np.float32(dana_loss_times[:-1])
#     ax1.loglog(x_values, dana_metrics_history[f'{dataset}_loss'], label=f'DANA {dataset}_loss')
#     if dataset == 'train':
#         ax1_right.loglog(x_values, dana_metrics_history['train_sigmoid_sum'], color='grey', linestyle='--', label='Sigmoid Sum')
#         ax1_right.tick_params(axis='y', labelcolor='grey')

# # Plot SGD results
# for dataset in ('train', 'val'):
#     flopscalar = BATCH_SIZE * dimension
#     length = min(len(sgd_metrics_history['train_loss']),len(dana_loss_times)-1)
#     x_values = flopscalar * np.float32(dana_loss_times[:length])
#     sgdtrain = sgd_metrics_history[f'{dataset}_loss'][:length]
#     sgdacc = 1.0-np.array(sgd_metrics_history[f'{dataset}_accuracy'][:length])
#     ax1.loglog(x_values, sgdtrain, label=f'SGD {dataset}_loss')

# # Add horizontal and vertical grid lines
# ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)

# # Set more y-axis ticks (equally spaced in log-space)
# import matplotlib.ticker as ticker
# ax1.yaxis.set_major_locator(ticker.LinearLocator(numticks=15))

# ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
# ax1.yaxis.set_minor_locator(ticker.NullLocator())

# ax1.set_xlabel('FLOPs')
# ax1.legend(loc='upper left')
# ax1_right.legend(loc='lower left')

# fig.suptitle("DANA Normalized MK3, filters={}".format(NUM_FILTERS))
# plt.savefig("cifar5m_resnet18_filters_{}_dana_normalized_mk3_delta={}_g3_iv={}_sv={}_p={}_ts={}_D={}_ITER={}.pdf".format(NUM_FILTERS, delta, g3_iv, g3_sv, g3_p, g3_ts, dimension, TRAIN_STEPS*BATCH_SIZE))

def run_experiment(subval_dataset, NUM_FILTERS, g3_p, g3_iv, g3_sv, g3_ts, delta=8):
    """
    Run a single experiment with given parameters.
    """
    # Run SGD or load SGD results
    sgd_metrics_filename = f"Results/sgd/sgd_metrics_history_steps_{TRAIN_STEPS}_filters_{NUM_FILTERS}.pkl"
    
    try:
        sgd_metrics_history = pickle.load(open(sgd_metrics_filename, "rb"))
    except (FileNotFoundError, IOError):
        # If the file doesn't exist, run SGD and create it
        sgd_metrics_history = None
        # Reset model with same initialization
        rngs = nnx.Rngs(0)  # Use same seed for fair comparison
        model = ResNetSoftSign(stage_sizes=[2, 2, 2, 2], num_classes=10, num_filters=NUM_FILTERS, rngs=rngs)

        learning_rate = 0.1
        momentum = 0.0

        schedule = optax.linear_schedule(
            init_value=learning_rate,
            end_value=0.0,
            transition_steps=TRAIN_STEPS
        )
        sgd = optax.sgd(schedule, momentum)

        sgd_loss_times, sgd_metrics_history, _ = run_resnet_loop(model, sgd, subval_dataset, TRAIN_STEPS, store_sigmoid_sum=False)
        with open(sgd_metrics_filename, 'wb') as f:
            pickle.dump(sgd_metrics_history, f)
        print(f"SGD metrics history saved to {sgd_metrics_filename}")

    # Configure DANA optimizer
    g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
    g2 = optimizers.powerlaw_schedule(0.4/POPULATION_TRACE, 0.0, 0.0, 1)
    Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, delta)
    g3 = optimizers.powerlaw_schedule(g3_iv, g3_sv, g3_p, g3_ts)
    
    # Run DANA Classic
    dana_classic = optimizers.dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)
    rngs = nnx.Rngs(0)
    model = ResNetSoftSign(stage_sizes=[2, 2, 2, 2], num_classes=10, num_filters=NUM_FILTERS, rngs=rngs)
    dana_loss_times, dana_metrics_history, dimension = run_resnet_loop(
        model, dana_classic, subval_dataset, TRAIN_STEPS, losscomparison=sgd_metrics_history
    )

    # Save DANA Classic results
    dana_classic_metrics_filename = f"Results/dana_classic/dana_classic_metrics_history_steps_{TRAIN_STEPS}_filters_{NUM_FILTERS}_g3_iv={g3_iv}_sv={g3_sv}_p={g3_p}_ts={g3_ts}_delta={delta}.pkl"
    with open(dana_classic_metrics_filename, 'wb') as f:
        pickle.dump(dana_metrics_history, f)
    print(f"DANA Classic metrics history saved to {dana_classic_metrics_filename}")

    # Create and save DANA Classic plot
    fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    ax1.set_title('Loss')
    ax1_right = ax1.twinx()
    ax1_right.set_yscale('log')
    ax1_right.set_ylabel('Sigmoid Sum', color='grey')

    # Plot DANA results
    for dataset in ('train', 'val'):
        flopscalar = BATCH_SIZE * dimension
        x_values = flopscalar * np.float32(dana_loss_times[:-1])
        ax1.loglog(x_values, dana_metrics_history[f'{dataset}_loss'], label=f'DANA {dataset}_loss')
        if dataset == 'train':
            ax1_right.loglog(x_values, dana_metrics_history['train_sigmoid_sum'], color='grey', linestyle='--', label='Sigmoid Sum')
            ax1_right.tick_params(axis='y', labelcolor='grey')

    # Plot SGD results
    for dataset in ('train', 'val'):
        flopscalar = BATCH_SIZE * dimension
        length = min(len(sgd_metrics_history['train_loss']), len(dana_loss_times)-1)
        x_values = flopscalar * np.float32(dana_loss_times[:length])
        sgdtrain = sgd_metrics_history[f'{dataset}_loss'][:length]
        sgdacc = 1.0-np.array(sgd_metrics_history[f'{dataset}_accuracy'][:length])
        ax1.loglog(x_values, sgdtrain, label=f'SGD {dataset}_loss')

    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    ax1.yaxis.set_major_locator(ticker.LinearLocator(numticks=15))
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    ax1.yaxis.set_minor_locator(ticker.NullLocator())
    ax1.set_xlabel('FLOPs')
    ax1.legend(loc='upper left')
    ax1_right.legend(loc='lower left')

    fig.suptitle(f"DANA Classic, filters={NUM_FILTERS}")
    plt.savefig(f"Results/dana_classic/cifar5m_resnet18_filters_{NUM_FILTERS}_dana_classic_delta={delta}_g3_iv={g3_iv}_sv={g3_sv}_p={g3_p}_ts={g3_ts}_D={dimension}_ITER={TRAIN_STEPS*BATCH_SIZE}.pdf")
    plt.close()

def main():
    """
    Run experiments over a grid of parameters.
    """
    # Define parameter grids
    filter_values = list(range(32, 385, 32))  # 32, 64, 96, ..., 384
    g3_iv_values = [0.00125/2.0, 0.00125/4.0, 0.00125/8.0, 0.00125/16.0, 0.00125/32.0, 0.00125/64.0, 0.00125/128.0, 0.00125/256.0, 0.00125/512.0, 0.00125/1024.0]
    g3_ts_values = [1.0]

    total_experiments = len(filter_values) * len(g3_iv_values)
    completed = 0

    val_dataset = ChunkedNpzDataset(VAL_FILE, BATCH_SIZE, CHUNK_SIZE, apply_augmentation=True)
    subval_dataset = list(val_dataset)[:10]  # Only use first 10 batches for validation
    del val_dataset

    for num_filters in filter_values:
        for g3_iv in g3_iv_values:
            for g3_ts in g3_ts_values:
                completed += 1
                print(f"\nRunning experiment {completed}/{total_experiments}")
                print(f"Parameters: NUM_FILTERS={num_filters}, g3_iv={g3_iv}, g3_ts={g3_ts}")
                
                try:
                    run_experiment(subval_dataset, num_filters, g3_p=0.0, g3_iv=g3_iv, g3_ts=g3_ts, g3_sv=0.0)
                except Exception as e:
                    print(f"Error in experiment with NUM_FILTERS={num_filters}, g3_iv={g3_iv}")
                    print(f"Error message: {str(e)}")
                    continue

if __name__ == "__main__":
    main()
