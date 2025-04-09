import os
import glob
import codecs
import random
import time
import math
import pickle
import sys
from typing import Dict, List, Tuple, Any, Optional, Callable

import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
from flax import nnx
from flax.nnx import rnglib
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats
import tensorflow as tf

# Disable GPU for TensorFlow to avoid conflicts with JAX
tf.config.set_visible_devices([], 'GPU')
print("TensorFlow GPU access disabled")

# Import optimizers for DANA
sys.path.append('../../')
import optimizers

# Configuration
import argparse

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 128,
    "num_steps": 20,  # Sequence length
    "train_steps": 10000,  # For full run; use 100 for debugging
    "vocab_size": 10000,  # Top 10k words as mentioned in the paper
    "emb_size": 32,  # Paper mentions 512
    "hidden_size": 64,  # Paper mentions 1024
    "num_layers": 2,  # Paper mentions 2-layer and 4-layer LSTMs
    "learning_rate": 0.1,
    "max_grad_norm": 10.0,
    "keep_prob": 0.9,  # Dropout keep probability
    "base_path": "/home/elliotpaquette/Documents/BillionWords",
    "max_tokens": 0,  # For debugging, set to 0 for full dataset
    # DANA optimizer parameters
    "max_val_tokens": 10000,  # For validation, set to 0 for full dataset
    "optimizer": "sgd",  # 'sgd' or 'dana'
    "population_trace": 4.0,  # For DANA optimizer
    "dana_g3_iv": 0.5,  # Initial value for g3 in DANA
    "dana_g3_sv": 0.0,  # Saturation value for g3 in DANA
    "dana_g3_p": -1.0,  # Power for g3 in DANA
    "dana_g3_ts": 1.0,  # Time scale for g3 in DANA
    "dana_delta": 8,  # Delta parameter for DANA
    "results_dir": "results",  # Directory to store results
}

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM language model on the Billion Word Dataset")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"], help="Batch size")
    parser.add_argument("--num_steps", type=int, default=DEFAULT_CONFIG["num_steps"], help="Sequence length")
    parser.add_argument("--train_steps", type=int, default=DEFAULT_CONFIG["train_steps"], help="Number of training steps")
    parser.add_argument("--vocab_size", type=int, default=DEFAULT_CONFIG["vocab_size"], help="Vocabulary size (top N words)")
    parser.add_argument("--emb_size", type=int, default=DEFAULT_CONFIG["emb_size"], help="Embedding dimension")
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_CONFIG["hidden_size"], help="Hidden layer size")
    parser.add_argument("--num_layers", type=int, default=DEFAULT_CONFIG["num_layers"], help="Number of LSTM layers")
    parser.add_argument("--learning_rate", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--max_grad_norm", type=float, default=DEFAULT_CONFIG["max_grad_norm"], help="Maximum gradient norm for clipping")
    parser.add_argument("--keep_prob", type=float, default=DEFAULT_CONFIG["keep_prob"], help="Dropout keep probability")
    parser.add_argument("--base_path", type=str, default=DEFAULT_CONFIG["base_path"], help="Base path for data")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_CONFIG["max_tokens"], help="Maximum number of tokens to use (None for all)")
    parser.add_argument("--max_val_tokens", type=int, default=DEFAULT_CONFIG["max_val_tokens"], help="Maximum number of validation tokens to use (None for all)")

    # DANA optimizer parameters
    parser.add_argument("--optimizer", type=str, default=DEFAULT_CONFIG["optimizer"], choices=["sgd", "dana"],
                        help="Optimizer to use (sgd or dana)")
    parser.add_argument("--population_trace", type=float, default=DEFAULT_CONFIG["population_trace"],
                        help="Population trace for DANA optimizer")
    parser.add_argument("--dana_g3_iv", type=float, default=DEFAULT_CONFIG["dana_g3_iv"],
                        help="Initial value for g3 in DANA")
    parser.add_argument("--dana_g3_sv", type=float, default=DEFAULT_CONFIG["dana_g3_sv"],
                        help="Saturation value for g3 in DANA")
    parser.add_argument("--dana_g3_p", type=float, default=DEFAULT_CONFIG["dana_g3_p"],
                        help="Power for g3 in DANA")
    parser.add_argument("--dana_g3_ts", type=float, default=DEFAULT_CONFIG["dana_g3_ts"],
                        help="Time scale for g3 in DANA")
    parser.add_argument("--dana_delta", type=float, default=DEFAULT_CONFIG["dana_delta"],
                        help="Delta parameter for DANA")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_CONFIG["results_dir"],
                        help="Directory to store results")
    
    return parser.parse_args()

# Set global configuration variables
args = parse_args()
BATCH_SIZE = args.batch_size
NUM_STEPS = args.num_steps
TRAIN_STEPS = args.train_steps
VOCAB_SIZE = args.vocab_size
EMB_SIZE = args.emb_size
HIDDEN_SIZE = args.hidden_size
NUM_LAYERS = args.num_layers
LEARNING_RATE = args.learning_rate
MAX_GRAD_NORM = args.max_grad_norm
KEEP_PROB = args.keep_prob
BASE_PATH = args.base_path
MAX_TOKENS = args.max_tokens if args.max_tokens > 0 else None
MAX_VAL_TOKENS = args.max_val_tokens if args.max_val_tokens > 0 else None
# DANA optimizer parameters
OPTIMIZER = args.optimizer
POPULATION_TRACE = args.population_trace
DANA_G3_IV = args.dana_g3_iv
DANA_G3_SV = args.dana_g3_sv
DANA_G3_P = args.dana_g3_p
DANA_G3_TS = args.dana_g3_ts
DANA_DELTA = args.dana_delta
RESULTS_DIR = args.results_dir

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)

# Derived paths
VOCAB_FILE = f"{BASE_PATH}/1b_word_vocab.txt"
DATA_PATH = f"{BASE_PATH}/training-monolingual"

# Set up logging
EVAL_INTERVAL = 100
LOG_STEPS = jnp.unique(jnp.concatenate([
    jnp.array([0]),
    jnp.int32(1.1**jnp.arange(1, jnp.ceil(jnp.log(TRAIN_STEPS)/jnp.log(1.1)))),
    jnp.array([TRAIN_STEPS])
]))

# Function for printing log messages that should appear in both console and log files
def log_print(message):
    """Print a message to stdout and ensure it appears in log files too."""
    # Use print() instead of tqdm.write() when tqdm is disabled
    if os.environ.get('TQDM_DISABLE') == '1':
        print(message)
    else:
        tqdm.write(message)

class Vocabulary:
    """Vocabulary class for handling word-to-id mapping."""
    
    def __init__(self):
        self._token_to_id = {}
        self._id_to_token = []
        self._token_to_count = {}
        self._num_tokens = 0
        self._s_id = None
        self._unk_id = None
    
    @property
    def num_tokens(self):
        return self._num_tokens
    
    @property
    def unk(self):
        return "<UNK>"
    
    @property
    def unk_id(self):
        return self._unk_id
    
    @property
    def s(self):
        return "<S>"
    
    @property
    def s_id(self):
        return self._s_id
    
    def add(self, token, count):
        self._token_to_id[token] = self._num_tokens
        self._token_to_count[token] = count
        self._id_to_token.append(token)
        self._num_tokens += 1
    
    def finalize(self):
        self._s_id = self.get_id(self.s)
        self._unk_id = self.get_id(self.unk)
    
    def get_id(self, token):
        return self._token_to_id.get(token, self._unk_id)
    
    def get_token(self, id_):
        return self._id_to_token[id_]
    
    @staticmethod
    def from_file(filename, max_vocab_size=None):
        vocab = Vocabulary()
        with codecs.open(filename, "r", "utf-8") as f:
            for i, line in enumerate(f):
                if max_vocab_size is not None and i >= max_vocab_size:
                    break
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab

class Dataset:
    """
    Dataset class for handling language model data.
    UNUSED - see TFDataset below
    """
    
    def __init__(self, vocab, file_pattern, deterministic=False, max_tokens=None):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic
        self._max_tokens = max_tokens
        self._total_tokens = 0
        
        # Load all sentences at initialization to ensure we have data
        self._all_sentences = []
        for file_name in glob.glob(self._file_pattern):
            with codecs.open(file_name, "r", "utf-8") as f:
                lines = [line.strip() for line in f]
                if not self._deterministic:
                    random.shuffle(lines)
                for line in lines:
                    if line:
                        sentence = self._parse_sentence(line)
                        self._all_sentences.append(sentence)
                        self._total_tokens += len(sentence)
                        if self._max_tokens is not None and self._total_tokens >= self._max_tokens:
                            break
            if self._max_tokens is not None and self._total_tokens >= self._max_tokens:
                break
        
        print(f"Loaded {len(self._all_sentences)} sentences with {self._total_tokens} tokens")
    
    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word) for word in line.strip().split()] + [s_id]
    
    def _iterate(self, batch_size, num_steps):
        # Create a copy of the sentences to avoid modifying the original
        sentences = self._all_sentences.copy()
        if not self._deterministic:
            random.shuffle(sentences)
        
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.uint8)
        
        sentence_idx = 0
        
        while True:
            x[:] = 0
            y[:] = 0
            w[:] = 0
            
            for i in range(batch_size):
                tokens_filled = 0
                while tokens_filled < num_steps:
                    if streams[i] is None or len(streams[i]) <= 1:
                        if sentence_idx >= len(sentences):
                            # No more sentences
                            break
                        streams[i] = sentences[sentence_idx]
                        sentence_idx += 1
                    
                    num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                    x[i, tokens_filled:tokens_filled+num_tokens] = streams[i][:num_tokens]
                    y[i, tokens_filled:tokens_filled+num_tokens] = streams[i][1:num_tokens+1]
                    w[i, tokens_filled:tokens_filled+num_tokens] = 1
                    streams[i] = streams[i][num_tokens:]
                    tokens_filled += num_tokens
            
            if not np.any(w):
                # No more data
                return
            
            yield jnp.array(x), jnp.array(y), jnp.array(w)
    
    def iterate_once(self, batch_size, num_steps):
        for value in self._iterate(batch_size, num_steps):
            yield value

class TFDataset:
    """TensorFlow Dataset implementation for handling language model data.
    
    This implementation uses tf.data.TextLineDataset to load data efficiently, without
    loading all files in memory. It allows streaming of training data and caching of
    validation data.
    """
    
    def __init__(self, vocab, file_pattern, batch_size, num_steps, deterministic=False, max_tokens=None, is_validation=False):
        """Initialize TensorFlow Dataset for language model.
        
        Args:
            vocab: Vocabulary object
            file_pattern: File pattern or specific file to load
            batch_size: Batch size for the dataset
            num_steps: Number of steps (sequence length)
            deterministic: Whether to use deterministic processing (no shuffling)
            max_tokens: Maximum number of tokens to use
            is_validation: Whether this is a validation dataset (for caching)
        """
        print(f"{'Validation' if is_validation else 'Training'} Dataset: Initializing with max_tokens={max_tokens or 'unlimited'}")
        start_time = time.time()
        
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._deterministic = deterministic
        self._max_tokens = max_tokens
        self._is_validation = is_validation
        self._total_tokens = 0
        
        # Resolve file pattern to actual files
        if isinstance(file_pattern, str) and "*" in file_pattern:
            print(f"Resolving glob pattern: {file_pattern}")
            self._files = sorted(glob.glob(file_pattern))
        elif isinstance(file_pattern, str):
            self._files = [file_pattern]
        else:
            self._files = list(file_pattern)
            
        if not self._files:
            raise ValueError(f"No files found matching pattern: {file_pattern}")
        
        print(f"Found {len(self._files)} files for {'validation' if self._is_validation else 'training'}")
        
        # For validation, just use one file to keep consistent with original implementation
        if self._is_validation and len(self._files) > 1:
            self._files = [self._files[-1]]  # Use last file for validation
            print(f"Using only 1 file for validation: {self._files[0]}")
            
        # Print file sizes to give an idea of how much data we're loading
        total_size_mb = 0
        for file in self._files:
            size_mb = os.path.getsize(file) / (1024 * 1024)
            total_size_mb += size_mb
            print(f"  - {file}: {size_mb:.2f} MB")
        print(f"Total size: {total_size_mb:.2f} MB")
        
        # Create the TensorFlow dataset pipeline
        print(f"Creating TensorFlow dataset pipeline...")
        self._tf_dataset = self._create_tf_dataset()
        
        # For validation, cache the dataset after preprocessing
        if self._is_validation:
            print("Caching validation dataset...")
            self._tf_dataset = self._tf_dataset.cache()
            
            # Calculate total tokens in validation set for reporting
            print("Calculating validation token count...")
            token_count = 0
            batch_count = 0
            for batch in self._tf_dataset:
                x, y, w = batch
                token_count += tf.reduce_sum(w)
                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processed {batch_count} validation batches, {token_count.numpy()} tokens so far")
                if self._max_tokens is not None and token_count >= self._max_tokens:
                    break
            
            self._total_validation_tokens = int(min(token_count.numpy(), self._max_tokens or float('inf')))
            print(f"Validation dataset contains approximately {self._total_validation_tokens} tokens in {batch_count} batches")
        
        # Report initialization time
        elapsed = time.time() - start_time
        print(f"{'Validation' if is_validation else 'Training'} dataset initialization completed in {elapsed:.2f} seconds")
    
    def _parse_line(self, line):
        """Parse a text line into token IDs.
        
        Args:
            line: A string tensor containing a line of text
            
        Returns:
            A list of token IDs
        """
        s_id = self._vocab.s_id
        
        # Skip empty lines
        if tf.strings.length(line) <= 0:
            return tf.constant([s_id, s_id], dtype=tf.int32)
        
        # Split line into words
        words = tf.strings.split(line)
        
        # Convert each word to its token ID using a py_function
        # (needed because vocabulary lookup is not a TensorFlow operation)
        token_ids = tf.py_function(
            lambda x: np.array([self._vocab.get_id(word.decode('utf-8')) for word in x.numpy()], dtype=np.int32),
            [words],
            tf.int32
        )
        
        # Add start and end tokens
        token_ids = tf.concat([[s_id], token_ids, [s_id]], axis=0)
        
        return token_ids
    
    def _create_tf_dataset(self):
        """Create a TensorFlow dataset pipeline for efficient processing.
        
        Returns:
            A tf.data.Dataset that yields (inputs, targets, weights) batches
        """
        start_time = time.time()
        
        # Create dataset from file list
        print("Creating dataset from files...")
        files_ds = tf.data.Dataset.from_tensor_slices(self._files)
        
        # Shuffle file order for training (but not for validation)
        if not self._deterministic:
            print("Shuffling files...")
            files_ds = files_ds.shuffle(buffer_size=len(self._files))
        
        # Load text lines from files, interleaving for better performance
        print("Setting up interleaved file reading...")
        lines_ds = files_ds.interleave(
            lambda file: tf.data.TextLineDataset(file).filter(lambda line: tf.strings.length(line) > 0),
            cycle_length=4,
            block_length=16,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Parse lines into token sequences
        print("Setting up tokenization...")
        tokens_ds = lines_ds.map(
            self._parse_line,
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # For training, shuffle the sequences
        if not self._deterministic:
            print("Setting up sequence shuffling...")
            tokens_ds = tokens_ds.shuffle(buffer_size=10000)
        
        # Create a dataset of token streams for batching
        print("Flattening tokens into a continuous stream...")
        # This approach concatenates all tokens and then creates sliding windows
        token_stream_ds = tokens_ds.flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(x)
        )
        
        # Apply max_tokens limit if specified
        if self._max_tokens is not None:
            print(f"Setting token limit to {self._max_tokens}")
            token_stream_ds = token_stream_ds.take(self._max_tokens + self._batch_size * self._num_steps)
        else:
            print("No token limit set - will use all available data")
        
        # Create windows of size num_steps + 1 (for input and target)
        print(f"Creating windows of size {self._num_steps + 1}...")
        # with stride of 1 to get all possible windows
        windows_ds = token_stream_ds.window(
            size=self._num_steps + 1,
            shift=self._num_steps,
            drop_remainder=True
        )
        
        # Convert windows to tensors
        print("Converting windows to tensors...")
        windows_ds = windows_ds.map(
            lambda window: tf.data.experimental.get_single_element(
                window.batch(self._num_steps + 1)
            ),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Split into inputs and targets
        print("Creating input-target pairs...")
        examples_ds = windows_ds.map(
            lambda window: (window[:-1], window[1:], tf.ones_like(window[:-1], dtype=tf.uint8)),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Batch the examples
        print(f"Batching examples with batch size {self._batch_size}...")
        batches_ds = examples_ds.batch(
            self._batch_size,
            drop_remainder=True
        )
        
        # Prefetch for better performance
        print("Setting up prefetching...")
        batches_ds = batches_ds.prefetch(tf.data.AUTOTUNE)
        
        # Report pipeline setup time
        elapsed = time.time() - start_time
        print(f"Dataset pipeline setup completed in {elapsed:.2f} seconds")
        print("Note: Actual data loading will begin when the first batch is requested")
        
        return batches_ds
    
    def iterate_once(self, batch_size, num_steps):
        """Iterate through the dataset once, yielding batches.
        
        Args:
            batch_size: Batch size for the dataset (must match initialization)
            num_steps: Number of steps for the dataset (must match initialization)
            
        Yields:
            Tuples of (inputs, targets, weights) as JAX arrays
        """
        assert batch_size == self._batch_size, f"Batch size mismatch: requested {batch_size}, dataset uses {self._batch_size}"
        assert num_steps == self._num_steps, f"Steps mismatch: requested {num_steps}, dataset uses {self._num_steps}"
        
        # Iterate through the TensorFlow dataset
        start_time = time.time()
        print(f"Beginning dataset iteration with {'unlimited' if self._max_tokens is None else self._max_tokens} max tokens")
        
        token_count = 0
        batch_count = 0
        for batch in self._tf_dataset:
            if batch_count == 0:
                print(f"First batch loaded after {time.time() - start_time:.2f} seconds")
                
            x, y, w = batch
            
            # Track token count and batches for reporting
            batch_tokens = tf.reduce_sum(w).numpy()
            token_count += batch_tokens
            batch_count += 1
            
            # Print progress every 10 batches
            if batch_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"Processed {batch_count} batches, {token_count} tokens so far ({token_count/elapsed:.1f} tokens/sec)")
            
            # Convert TensorFlow tensors to JAX arrays
            yield jnp.array(x), jnp.array(y), jnp.array(w)
            
            # Track token count for max_tokens limit
            if self._max_tokens is not None and token_count >= self._max_tokens:
                print(f"Reached max_tokens limit ({self._max_tokens}) after {batch_count} batches")
                break
                
        elapsed = time.time() - start_time
        print(f"Dataset iteration completed: {batch_count} batches, {token_count} tokens in {elapsed:.2f} seconds")

class LSTM(nnx.Module):
    """LSTM Language Model implementation using Flax NNX."""
    
    def __init__(self, vocab_size, emb_size, hidden_size, num_layers, keep_prob=1.0, rngs=None):
        """Initialize LSTM Language Model.
        
        Args:
            vocab_size: Size of vocabulary
            emb_size: Embedding dimension
            hidden_size: Hidden dimension of LSTM
            num_layers: Number of LSTM layers
            keep_prob: Dropout keep probability
            rngs: Random number generator state
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        
        if rngs is None:
            rngs = nnx.Rngs(0)
        
        # Embedding layer
        self.embedding = nnx.Embed(
            num_embeddings=vocab_size,
            features=emb_size,
            embedding_init=nnx.initializers.normal(stddev=0.1),
            rngs=rngs
        )
        
        # LSTM layers
        self.lstm_cells = []
        for i in range(num_layers):
            in_features = emb_size if i == 0 else hidden_size
            cell = nnx.OptimizedLSTMCell(
                in_features=in_features,
                hidden_features=hidden_size,
                kernel_init=nnx.initializers.normal(stddev=0.1),
                recurrent_kernel_init=nnx.initializers.normal(stddev=0.1),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs
            )
            self.lstm_cells.append(cell)
        
        # Output layer
        self.output = nnx.Linear(
            in_features=hidden_size,
            out_features=vocab_size,
            kernel_init=nnx.initializers.normal(stddev=0.1),
            bias_init=nnx.initializers.zeros_init(),
            rngs=rngs
        )

    def initialize_carry(self, batch_size):
        """Initialize LSTM cell states for each layer."""
        carries = []
        for cell in self.lstm_cells:
            carry = cell.initialize_carry((batch_size, self.emb_size))
            carries.append(carry)
        return carries
    
    def __call__(self, inputs, carries=None, training=True):
        """Forward pass of the LSTM Language Model.
        
        Args:
            inputs: Input token ids of shape [batch_size, seq_len]
            carries: Initial LSTM cell states for each layer
            training: Whether to apply dropout
        
        Returns:
            logits: Output logits of shape [batch_size, seq_len, vocab_size]
            new_carries: Updated LSTM cell states for each layer
        """
        batch_size, seq_len = inputs.shape
        
        # Initialize carries if not provided
        if carries is None:
            carries = self.initialize_carry(batch_size)
        
        # Embed inputs
        embedded = self.embedding(inputs)  # [batch_size, seq_len, emb_size]
        
        # Apply dropout to embeddings
        if training and self.keep_prob < 1.0:
            dropout = nnx.Dropout(rate=1.0-self.keep_prob, rngs=nnx.Rngs(0))
            embedded = dropout(embedded, deterministic=not training)
        
        # Process inputs through LSTM layers
        outputs = jnp.zeros((batch_size, seq_len, self.hidden_size))
        new_carries = []
        
        # Process each timestep
        for t in range(seq_len):
            x = embedded[:, t, :]
            h = x
            current_carries = []
            
            # Process through each LSTM layer
            for i, cell in enumerate(self.lstm_cells):
                carry, h = cell(carries[i], h)
                current_carries.append(carry)
                
                # Apply dropout between layers (but not on the output of the last layer)
                if training and self.keep_prob < 1.0 and i < len(self.lstm_cells) - 1:
                    dropout = nnx.Dropout(rate=1.0-self.keep_prob, rngs=nnx.Rngs(1))
                    h = dropout(h, deterministic=not training)
            
            # Store output and update carries
            outputs = outputs.at[:, t, :].set(h)
            carries = current_carries
        
        # Apply output layer to get logits
        outputs = outputs.reshape(-1, self.hidden_size)
        logits = self.output(outputs)
        logits = logits.reshape(batch_size, seq_len, self.vocab_size)
        
        new_carries = carries
        return logits, new_carries

def compute_loss(logits, targets, weights=None):
    """Compute cross entropy loss with optional masking.
    
    Args:
        logits: Model logits of shape [batch_size, seq_len, vocab_size]
        targets: Target token ids of shape [batch_size, seq_len]
        weights: Optional weights for masking of shape [batch_size, seq_len]
    
    Returns:
        loss: Scalar loss value
    """
    # Reshape for computing loss
    batch_size, seq_len = targets.shape
    logits = logits.reshape(-1, logits.shape[-1])
    targets = targets.reshape(-1)
    
    # Compute cross entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    
    # Apply weights if provided
    if weights is not None:
        weights = weights.reshape(-1)
        loss = loss * weights
        # Normalize by sum of weights
        return jnp.sum(loss) / (jnp.sum(weights) + 1e-8)
    
    return jnp.mean(loss)

def compute_perplexity(loss):
    """Convert loss to perplexity."""
    return jnp.exp(loss)

def train_step(optimizer, batch):
    """Perform a single training step.
    
    Args:
        optimizer: NNX optimizer containing the model
        batch: Batch of data containing (inputs, targets, weights)
    
    Returns:
        loss: Loss value for this batch
        grad_norm: Norm of the gradients
        optimizer: Updated optimizer
    """
    inputs, targets, weights = batch
    
    def loss_fn(model):
        logits, _ = model(inputs, training=True)
        loss = compute_loss(logits, targets, weights)
        return loss
    
    # Compute gradients
    model = optimizer.model
    loss = loss_fn(model)
    grads = nnx.grad(loss_fn)(model)
    
    # Compute gradient norm
    grads_flat = jax.tree_util.tree_leaves(grads)
    grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads_flat))
    
    # Update parameters
    optimizer.update(grads)
    
    return loss, grad_norm, optimizer

def evaluate(model, data_iterator):
    """Evaluate the model on the provided data.
    
    Args:
        model: LSTM model
        data_iterator: Iterator yielding evaluation batches
    
    Returns:
        avg_loss: Average loss over all batches
        perplexity: Perplexity calculated from avg_loss
    """
    total_loss = 0.0
    total_weights = 0.0
    
    for batch in data_iterator:
        inputs, targets, weights = batch
        
        # Forward pass
        logits, _ = model(inputs, training=False)
        
        # Compute loss
        loss = compute_loss(logits, targets, weights)
        
        # Accumulate loss weighted by batch size
        batch_weight = jnp.sum(weights)
        total_loss += loss * batch_weight
        total_weights += batch_weight
    
    # Compute average loss and perplexity
    if total_weights > 0:
        avg_loss = total_loss / total_weights
        perplexity = compute_perplexity(avg_loss)
    else:
        # Return default values if no valid data
        avg_loss = jnp.array(0.0)
        perplexity = jnp.array(1.0)
    
    return avg_loss, perplexity

def count_parameters(model):
    """Count the number of parameters in the model."""
    state = nnx.state(model)
    return sum(p.size for p in jax.tree_util.tree_leaves(state))

def create_learning_curve_plot(tokens, perplexities, num_params):
    """Create a log-log plot of the learning curve.
    
    Args:
        tokens: List of token counts
        perplexities: List of perplexity values
        num_params: Number of model parameters
    """
    # Convert to numpy arrays
    tokens = np.array(tokens)
    perplexities = np.array(perplexities)
    
    # Fit power law: perplexity = a * (tokens)^b
    log_tokens = np.log(tokens)
    log_perplexity = np.log(perplexities)
    
    # Only fit if we have enough data points
    if len(tokens) > 2:
        # Linear fit in log-log space
        slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_perplexity)
        
        # Power law parameters: perplexity = A * tokens^beta
        A = np.exp(intercept)
        beta = slope
        fit_perplexity = A * (tokens ** beta)
        fit_label = f"ε(m) = {A:.1f} m^({beta:.3f})"
    else:
        fit_perplexity = None
        fit_label = "Insufficient data for fit"
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the learning curve
    ax.loglog(tokens, perplexities, 'o-', color='blue', label=f'{NUM_LAYERS}-Layer LSTM (actual)')
    
    # Plot the fit if available
    if fit_perplexity is not None:
        ax.loglog(tokens, fit_perplexity, '--', color='red', label=fit_label)
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens (millions)')
    ax.set_ylabel('Validation Perplexity')
    ax.set_title(f'LSTM Language Model Learning Curve\n{num_params:,} parameters, {HIDDEN_SIZE} hidden units')
    
    # Convert x-axis to millions
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}'))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend()
    
    # Generate a more descriptive filename that includes key parameters
    output_filename = (
        f"{RESULTS_DIR}/sgd_learning_curve_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}.pdf"
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename)
    
    print(f"\nLearning curve plot saved as {output_filename}")

def run_lstm_with_optimizer(model, optax_optimizer, train_dataset, valid_dataset, train_steps, store_sigmoid_sum=False):
    """
    Train an LSTM model using the specified optimizer.
    
    Args:
        model: LSTM model instance
        optax_optimizer: Optax optimizer
        train_dataset: Training dataset
        valid_dataset: Validation dataset
        train_steps: Number of training steps
        store_sigmoid_sum: Whether to store sigmoid_sum (for DANA optimizer)
        
    Returns:
        loss_times: Steps at which metrics were recorded
        metrics_history: Dictionary of metrics
        num_params: Number of model parameters
    """
    # Create NNX optimizer
    optimizer = nnx.Optimizer(model, optax_optimizer)
    
    # Prepare for training
    metrics_history = {
        'step': [],
        'train_loss': [],
        'train_perplexity': [],
        'train_grad_norm': [],
        'val_loss': [],
        'val_perplexity': []
    }
    
    # Add sigmoid_sum if using DANA
    if store_sigmoid_sum:
        metrics_history['sigmoid_sum'] = []
    
    # Training loop
    print(f"Starting training for {train_steps} steps")
    step = 0
    train_iterator = train_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
    
    start_time = time.time()
    
    # Create a tqdm progress bar
    with tqdm(total=train_steps, initial=step, desc="Training", dynamic_ncols=True, file=sys.stderr) as pbar:
        for batch in train_iterator:
            if step >= train_steps:
                break
            
            # Perform training step
            loss, grad_norm, optimizer = train_step(optimizer, batch)
            
            # Update the progress bar description with current loss
            pbar.set_postfix({"loss": f"{loss:.4f}", "perplexity": f"{compute_perplexity(loss):.2f}"})
            
            # Log metrics at specified intervals
            if step in LOG_STEPS:
                perplexity = compute_perplexity(loss)
                
                metrics_history['step'].append(step)
                metrics_history['train_loss'].append(loss)
                metrics_history['train_perplexity'].append(perplexity)
                metrics_history['train_grad_norm'].append(grad_norm)
                
                # Store sigmoid_sum if requested (just use a constant value for compatibility)
                if store_sigmoid_sum:
                    sigmoid_sum = 0.0
                    metrics_history['sigmoid_sum'].append(sigmoid_sum)
                
                # Evaluate on validation set
                valid_iterator = valid_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
                val_loss, val_perplexity = evaluate(optimizer.model, valid_iterator)
                
                metrics_history['val_loss'].append(val_loss)
                metrics_history['val_perplexity'].append(val_perplexity)
                
                # Print metrics (outside of the progress bar)
                elapsed = time.time() - start_time
                log_print(f"Step: {step}/{train_steps} ({100.0 * step / train_steps:.1f}%) - Time: {elapsed:.2f}s")
                log_print(f"  Train Loss: {loss:.4f}, Perplexity: {perplexity:.2f}, Grad Norm: {grad_norm:.2f}")
                if store_sigmoid_sum:
                    log_print(f"  Sigmoid Sum: {sigmoid_sum:.4f}")
                log_print(f"  Valid Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
            
            # Update the progress bar
            pbar.update(1)
            step += 1
    
    # Final evaluation
    valid_iterator = valid_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
    final_loss, final_perplexity = evaluate(optimizer.model, valid_iterator)
    
    print("\nTraining completed!")
    print(f"Final validation loss: {final_loss:.4f}, perplexity: {final_perplexity:.2f}")
    
    # Count model parameters
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params}")
    
    return LOG_STEPS, metrics_history, num_params

def create_comparison_plot(sgd_metrics, dana_metrics, loss_times, num_params):
    """Create a comparison plot between SGD and DANA optimizers.
    
    Args:
        sgd_metrics: Metrics history from SGD training
        dana_metrics: Metrics history from DANA training 
        loss_times: Steps at which metrics were recorded
        num_params: Number of model parameters
    """
    # Create the figure
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title('Loss Comparison')
    
    # Determine the minimum length of data to plot
    min_length = min(len(sgd_metrics['train_loss']), 
                    len(dana_metrics['train_loss']) if dana_metrics else float('inf'))
    
    # Create a second y-axis for sigmoid-sum if available
    if dana_metrics and 'sigmoid_sum' in dana_metrics:
        ax1_right = ax1.twinx()
        ax1_right.set_yscale('log')
        ax1_right.set_ylabel('Sigmoid Sum', color='grey')
    
    # Skip step 0 and ensure we don't go beyond min_length
    skip = 1  # Skip the first entry (step 0)
    tokens = np.array(loss_times[skip:min_length]) * BATCH_SIZE * NUM_STEPS
    
    # Plot SGD results
    for dataset in ('train', 'val'):
        data = sgd_metrics[f'{dataset}_loss'][skip:min_length]
        ax1.loglog(tokens, data, label=f'SGD {dataset}_loss')
    
    # Plot DANA results if available
    if dana_metrics:
        for dataset in ('train', 'val'):
            data = dana_metrics[f'{dataset}_loss'][skip:min_length]
            ax1.loglog(tokens, data, label=f'DANA {dataset}_loss')
        
        # Plot sigmoid sum if available
        if 'sigmoid_sum' in dana_metrics:
            sigmoid_data = dana_metrics['sigmoid_sum'][skip:min_length]
            if np.any(sigmoid_data):  # Only plot if values are non-zero
                ax1_right.loglog(tokens, sigmoid_data, color='grey', linestyle='--', label='Sigmoid Sum')
                ax1_right.tick_params(axis='y', labelcolor='grey')
    
    # Add horizontal and vertical grid lines
    ax1.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    
    # Set more y-axis ticks
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    
    # Set axis labels
    ax1.set_xlabel('Training Tokens')
    ax1.set_ylabel('Loss')
    
    # Add legend
    ax1.legend(loc='upper left')
    if dana_metrics and 'sigmoid_sum' in dana_metrics:
        ax1_right.legend(loc='lower left')
    
    # Add title
    fig.suptitle(f"LSTM Language Model Training ({num_params:,} parameters)")
    
    # Generate a more descriptive filename that includes key parameters
    optimizer_str = "sgd_vs_dana" if dana_metrics else "sgd"
    output_filename = (
        f"{RESULTS_DIR}/comparison_{optimizer_str}_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}"
    )
    
    # Add DANA parameters if applicable
    if dana_metrics:
        output_filename += (f"_g3_iv={DANA_G3_IV}_sv={DANA_G3_SV}_"
                          f"p={DANA_G3_P}_ts={DANA_G3_TS}_delta={DANA_DELTA}")
    
    # Add file extension
    output_filename += ".pdf"
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename)
    
    print(f"\nComparison plot saved to {output_filename}")

def main():
    # Initialize random seed for reproducibility
    rng = jrandom.PRNGKey(42)
    
    # Load vocabulary (limit to top 10k words)
    print(f"Loading vocabulary from {VOCAB_FILE}")
    start_time = time.time()
    vocab = Vocabulary.from_file(VOCAB_FILE, max_vocab_size=VOCAB_SIZE)
    print(f"Vocabulary size: {vocab.num_tokens}, loaded in {time.time() - start_time:.2f} seconds")
    
    # Create dataset using all shuffled files for training
    train_pattern = f"{DATA_PATH}/*.en.shuffled"
    train_files = glob.glob(train_pattern)
    print(f"Found {len(train_files)} training files")
    
    # Get validation files (using *.en files)
    valid_pattern = f"{DATA_PATH}/*.en"
    valid_files = glob.glob(valid_pattern)
    print(f"Found {len(valid_files)} validation files")
    
    # Use all shuffled files for training and one .en file for validation
    if len(valid_files) > 0:
        valid_pattern = valid_files[-1]  # Use last .en file for validation
    else:
        print("Warning: No .en files found for validation, using training file")
        valid_pattern = train_files[-1]
    
    print(f"Using pattern {train_pattern} for training")
    print(f"Using {valid_pattern} for validation")
    
    # Create datasets - using either the original Dataset implementation or the new TFDataset
    use_tf_dataset = True  # Set to True to use TensorFlow-based dataset
    
    if use_tf_dataset:
        print("Using TensorFlow-based dataset implementation")
        print(f"Creating validation dataset (max_tokens={MAX_VAL_TOKENS})...")
        valid_dataset = TFDataset(vocab, valid_pattern, BATCH_SIZE, NUM_STEPS, deterministic=True, max_tokens=MAX_VAL_TOKENS, is_validation=True)
        
        print(f"Creating training dataset (max_tokens={MAX_TOKENS})...")
        train_dataset = TFDataset(vocab, train_pattern, BATCH_SIZE, NUM_STEPS, deterministic=False, max_tokens=MAX_TOKENS)
    else:
        print("Using original dataset implementation")
        train_dataset = Dataset(vocab, train_pattern, max_tokens=MAX_TOKENS)
        valid_dataset = Dataset(vocab, valid_pattern, deterministic=True, max_tokens=MAX_VAL_TOKENS)
    
    # Create a config dictionary for matching with saved runs
    sgd_config = {
        "optimizer": "sgd",
        "batch_size": BATCH_SIZE,
        "num_steps": NUM_STEPS,
        "train_steps": TRAIN_STEPS,
        "vocab_size": VOCAB_SIZE,
        "emb_size": EMB_SIZE,
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "learning_rate": LEARNING_RATE,
        "max_grad_norm": MAX_GRAD_NORM,
        "keep_prob": KEEP_PROB,
        "max_tokens": MAX_TOKENS,
    }
    
    # Generate a more comprehensive filename that includes key parameters
    sgd_metrics_filename = (
        f"{RESULTS_DIR}/sgd_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}.pkl"
    )
    
    # Try to load saved SGD results first
    try:
        print(f"Attempting to load saved SGD metrics from {sgd_metrics_filename}")
        with open(sgd_metrics_filename, "rb") as f:
            saved_config, sgd_loss_times, sgd_metrics_history, num_params = pickle.load(f)
            
        # Verify that the saved configuration matches the current one
        config_match = True
        for key, value in sgd_config.items():
            if key not in saved_config or saved_config[key] != value:
                config_match = False
                break
        
        if config_match:
            print("Successfully loaded SGD metrics from file with matching configuration")
        else:
            print("Found saved SGD metrics but configuration doesn't match. Running training again.")
            raise ValueError("Configuration mismatch")
            
    except (FileNotFoundError, IOError, ValueError):
        # If the file doesn't exist or config doesn't match, run SGD training
        print("Running SGD training")
        
        # Initialize model
        rng, init_rng = jrandom.split(rng)
        rngs = nnx.Rngs(init_rng)
        
        sgd_model = LSTM(
            vocab_size=vocab.num_tokens,
            emb_size=EMB_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            keep_prob=KEEP_PROB,
            rngs=rngs
        )
        
        # Initialize model parameters
        dummy_input = jnp.zeros((BATCH_SIZE, NUM_STEPS), dtype=jnp.int32)
        sgd_model(dummy_input)
        
        # Initialize SGD optimizer with gradient clipping
        sgd_optax = optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM),
            optax.sgd(learning_rate=LEARNING_RATE)
        )
        
        # Run SGD training
        sgd_loss_times, sgd_metrics_history, num_params = run_lstm_with_optimizer(
            sgd_model,
            sgd_optax,
            train_dataset,
            valid_dataset,
            TRAIN_STEPS,
            store_sigmoid_sum=False
        )
        
        # Save SGD results to file, including the configuration
        with open(sgd_metrics_filename, 'wb') as f:
            pickle.dump((sgd_config, sgd_loss_times, sgd_metrics_history, num_params), f)
        print(f"SGD metrics saved to {sgd_metrics_filename}")
    
    # Create learning curve plot for SGD
    tokens_list = []
    perplexity_list = []
    for i, step in enumerate(sgd_metrics_history['step']):
        if step > 0:  # Skip step 0
            tokens = step * BATCH_SIZE * NUM_STEPS
            perplexity = sgd_metrics_history['val_perplexity'][i]
            tokens_list.append(tokens)
            perplexity_list.append(perplexity)
    
    # Create SGD learning curve plot
    create_learning_curve_plot(tokens_list, perplexity_list, num_params)
    
    # Run DANA if specified
    dana_metrics_history = None
    if OPTIMIZER == "dana":
        # Create a config dictionary for matching with saved runs
        dana_config = {
            "optimizer": "dana",
            "batch_size": BATCH_SIZE,
            "num_steps": NUM_STEPS,
            "train_steps": TRAIN_STEPS,
            "vocab_size": VOCAB_SIZE,
            "emb_size": EMB_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "learning_rate": LEARNING_RATE,
            "max_grad_norm": MAX_GRAD_NORM,
            "keep_prob": KEEP_PROB,
            "max_tokens": MAX_TOKENS,
            # DANA-specific parameters
            "population_trace": POPULATION_TRACE,
            "dana_g3_iv": DANA_G3_IV,
            "dana_g3_sv": DANA_G3_SV,
            "dana_g3_p": DANA_G3_P,
            "dana_g3_ts": DANA_G3_TS,
            "dana_delta": DANA_DELTA,
        }
        
        # Generate a more comprehensive filename that includes key parameters
        dana_metrics_filename = (
            f"{RESULTS_DIR}/dana_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
            f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
            f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_g3_iv={DANA_G3_IV}_sv={DANA_G3_SV}_"
            f"p={DANA_G3_P}_ts={DANA_G3_TS}_delta={DANA_DELTA}.pkl"
        )
        
        # Try to load saved DANA results first
        try:
            print(f"Attempting to load saved DANA metrics from {dana_metrics_filename}")
            with open(dana_metrics_filename, "rb") as f:
                saved_config, dana_loss_times, dana_metrics_history, _ = pickle.load(f)
                
            # Verify that the saved configuration matches the current one
            config_match = True
            for key, value in dana_config.items():
                if key not in saved_config or saved_config[key] != value:
                    config_match = False
                    break
            
            if config_match:
                print("Successfully loaded DANA metrics from file with matching configuration")
            else:
                print("Found saved DANA metrics but configuration doesn't match. Running training again.")
                raise ValueError("Configuration mismatch")
                
        except (FileNotFoundError, IOError, ValueError):
            # If the file doesn't exist or config doesn't match, run DANA training
            print("Running DANA training")
            
            # Initialize model with same seed
            rng, init_rng = jrandom.split(jrandom.PRNGKey(42))
            rngs = nnx.Rngs(init_rng)
            
            dana_model = LSTM(
                vocab_size=vocab.num_tokens,
                emb_size=EMB_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                keep_prob=KEEP_PROB,
                rngs=rngs
            )
            
            # Initialize model parameters
            dummy_input = jnp.zeros((BATCH_SIZE, NUM_STEPS), dtype=jnp.int32)
            dana_model(dummy_input)
            
            # Configure DANA optimizer
            g1 = optimizers.powerlaw_schedule(1.0, 0.0, 0.0, 1)
            g2 = optimizers.powerlaw_schedule(LEARNING_RATE, 0.0, 0.0, 1)
            Delta = optimizers.powerlaw_schedule(1.0, 0.0, -1.0, DANA_DELTA)
            g3 = optimizers.powerlaw_schedule(DANA_G3_IV, DANA_G3_SV, DANA_G3_P, DANA_G3_TS)
            
            # Run DANA Classic
            dana_classic = optimizers.dana_optimizer(g1=g1, g2=g2, g3=g3, Delta=Delta)

            dana_optax = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                dana_classic
            )
            
            # Run DANA training
            dana_loss_times, dana_metrics_history, _ = run_lstm_with_optimizer(
                dana_model,
                dana_optax,
                train_dataset,
                valid_dataset,
                TRAIN_STEPS,
                store_sigmoid_sum=True
            )
            
            # Save DANA results to file, including the configuration
            with open(dana_metrics_filename, 'wb') as f:
                pickle.dump((dana_config, dana_loss_times, dana_metrics_history, num_params), f)
            print(f"DANA metrics saved to {dana_metrics_filename}")
        
        # Create comparison plot
        create_comparison_plot(sgd_metrics_history, dana_metrics_history, sgd_loss_times, num_params)

if __name__ == "__main__":
    main()
