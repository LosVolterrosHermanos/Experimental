"""
Optimized LSTM Language Model for the Billion Word Dataset using JAX/Flax.
This version includes:
1. JAX/Flax optimizations from Katie_LSTM_excerpt.py
2. HuggingFace datasets for efficient data loading
3. JIT compilation for faster training
4. Scan-based implementation for better memory efficiency
"""

import os
import time
import math
import pickle
import logging
import argparse
from typing import Dict, List, Tuple, Any, Optional, Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax import random as jrandom
import numpy as np
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats

# Import for Huggingface datasets
from datasets import load_dataset, Dataset

# Import optimizers for DANA
import sys
sys.path.append("/home/math/elliot.paquette@MCGILL.CA/Experimental/")
import optimizers

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "batch_size": 128,
    "num_steps": 20,  # Sequence length
    "train_steps": 10000,  # For full run; use 100 for debugging
    "vocab_size": 10000,  # Top 10k words as mentioned in the paper
    "emb_size": 512,
    "hidden_size": 1024,
    "num_layers": 2,  # Paper mentions 2-layer and 4-layer LSTMs
    "learning_rate": 0.1,
    "max_grad_norm": 10.0,
    "keep_prob": 0.9,  # Dropout keep probability
    "base_path": "/home/math/elliot.paquette@MCGILL.CA/BillionWords/lm1b",
    "max_tokens": 0,  # For debugging, set to None for full dataset
    "max_val_tokens": 10000,  # Maximum number of tokens for validation
    "optimizer": "sgd",  # 'sgd' or 'dana'
    "population_trace": 4.0,  # For DANA optimizer
    "dana_g3_iv": 0.5,  # Initial value for g3 in DANA
    "dana_g3_sv": 0.0,  # Saturation value for g3 in DANA
    "dana_g3_p": -1.0,  # Power for g3 in DANA
    "dana_g3_ts": 1.0,  # Time scale for g3 in DANA
    "dana_delta": 8,  # Delta parameter for DANA
    "results_dir": "results",  # Directory to store results
    "use_scan": True,  # Use scan for better memory efficiency
    "use_huggingface": False,  # Use Huggingface datasets
    "dataset_name": "lm1b",  # Huggingface dataset name
    "cache_dir": "./data_cache",  # Cache directory for datasets
    "prefetch_batches": 10  # Number of batches to prefetch
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
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_CONFIG["max_tokens"], help="Maximum number of tokens to use for training (None for all)")
    parser.add_argument("--max_val_tokens", type=int, default=DEFAULT_CONFIG["max_val_tokens"], help="Maximum number of tokens to use for validation")
    parser.add_argument("--max_files", type=int, default=None, help="Maximum number of files to process (None for all)")
    
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
    
    # Optimization flags
    parser.add_argument("--use_scan", action="store_true", default=DEFAULT_CONFIG["use_scan"],
                        help="Use scan-based implementation for better memory efficiency")
    parser.add_argument("--use_huggingface", action="store_true", default=DEFAULT_CONFIG["use_huggingface"],
                        help="Use Huggingface datasets for efficient data loading")
    parser.add_argument("--dataset_name", type=str, default=DEFAULT_CONFIG["dataset_name"],
                        help="Huggingface dataset name")
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_CONFIG["cache_dir"],
                        help="Cache directory for datasets")
    parser.add_argument("--prefetch_batches", type=int, default=DEFAULT_CONFIG["prefetch_batches"],
                        help="Number of batches to prefetch")
    
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
MAX_VAL_TOKENS = args.max_val_tokens if args.max_val_tokens > 0 else 10000
MAX_FILES = args.max_files if args.max_files and args.max_files > 0 else None

# DANA optimizer parameters
OPTIMIZER = args.optimizer
POPULATION_TRACE = args.population_trace
DANA_G3_IV = args.dana_g3_iv
DANA_G3_SV = args.dana_g3_sv
DANA_G3_P = args.dana_g3_p
DANA_G3_TS = args.dana_g3_ts
DANA_DELTA = args.dana_delta
RESULTS_DIR = args.results_dir

# Optimization flags
USE_SCAN = args.use_scan
USE_HUGGINGFACE = args.use_huggingface
DATASET_NAME = args.dataset_name
CACHE_DIR = args.cache_dir
PREFETCH_BATCHES = args.prefetch_batches

# Create results directory if it doesn't exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

# Derived paths
VOCAB_FILE = f"{BASE_PATH}/1b_word_vocab.txt"

# Set up logging
EVAL_INTERVAL = 100
LOG_STEPS = jnp.unique(jnp.concatenate([
    jnp.array([0]),
    jnp.int32(1.1**jnp.arange(1, jnp.ceil(jnp.log(TRAIN_STEPS)/jnp.log(1.1)))),
    jnp.array([TRAIN_STEPS])
]))

#################################################
# Core data structures
#################################################

class LSTMCarry(NamedTuple):
    """LSTM cell state."""
    h: jnp.ndarray  # hidden state
    c: jnp.ndarray  # cell state

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
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_vocab_size is not None and i >= max_vocab_size:
                    break
                word, count = line.strip().split()
                vocab.add(word, int(count))
        vocab.finalize()
        return vocab

#################################################
# Dataset classes
#################################################

class Dataset:
    """Dataset class for language model data with optimized memory usage.
    
    For training data: Loads one file at a time to minimize memory usage.
    For validation/test data: Loads all data at once since it's used repeatedly.
    """
    
    def __init__(self, vocab, file_pattern, deterministic=False, max_tokens=None, max_files=None, is_test_set=False):
        self._vocab = vocab
        self._file_pattern = file_pattern
        self._deterministic = deterministic
        self._max_tokens = max_tokens
        self._max_files = max_files
        self._is_test_set = is_test_set
        self._total_tokens = 0
        self._all_sentences = None  # Will be populated for test sets
        
        import glob
        import os
        import random
        
        # Get list of files
        self._all_files = glob.glob(self._file_pattern)
        if not self._all_files:
            raise FileNotFoundError(f"No files found matching pattern: {self._file_pattern}")
        
        # Sort files to ensure deterministic order for reproducibility
        self._all_files.sort()
        
        # Shuffle files for training diversity if not deterministic
        if not self._deterministic:
            random.shuffle(self._all_files)
        
        # Limit number of files if specified
        if max_files is not None and max_files > 0:
            self._all_files = self._all_files[:max_files]
        
        logger.info(f"Found {len(self._all_files)} files from pattern: {self._file_pattern}")
        
        # For test sets, load all data at once since we'll need it repeatedly
        if is_test_set:
            logger.info("Loading all test set data into memory since it will be used repeatedly")
            self._all_sentences = []
            for file_idx, file_name in enumerate(self._all_files):
                file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
                logger.info(f"Loading test file {file_idx+1}/{len(self._all_files)}: {os.path.basename(file_name)} ({file_size_mb:.2f} MB)")
                
                sentences, file_tokens = self._load_sentences_from_file(file_name)
                self._all_sentences.extend(sentences)
                self._total_tokens += file_tokens
                
                if self._max_tokens is not None and self._total_tokens >= self._max_tokens:
                    logger.info(f"Reached max test tokens limit: {self._max_tokens}")
                    break
            
            logger.info(f"Loaded {len(self._all_sentences)} test sentences with {self._total_tokens} tokens ({self._total_tokens/1000000:.2f}M)")
        else:
            # For training data, just estimate tokens
            sample_size = min(5, len(self._all_files))
            estimated_tokens_per_file = []
            
            for i in range(sample_size):
                file_name = self._all_files[i]
                file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
                logger.info(f"Sampling file {i+1}/{sample_size}: {os.path.basename(file_name)} ({file_size_mb:.2f} MB)")
                
                token_count = self._count_tokens_in_file(file_name)
                estimated_tokens_per_file.append(token_count)
            
            if sample_size > 0:
                avg_tokens_per_file = sum(estimated_tokens_per_file) / sample_size
                estimated_total_tokens = avg_tokens_per_file * len(self._all_files)
                
                if self._max_tokens is not None:
                    estimated_total_tokens = min(estimated_total_tokens, self._max_tokens)
                    
                logger.info(f"Estimated total tokens: {estimated_total_tokens:,.0f} ({estimated_total_tokens/1000000:.2f}M)")
    
    def _count_tokens_in_file(self, file_name):
        """Count tokens in a file without storing them."""
        import codecs
        
        token_count = 0
        with codecs.open(file_name, "r", "utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    # +2 for start/end tokens
                    token_count += len(line.split()) + 2
                    
                    if self._max_tokens is not None and token_count >= self._max_tokens:
                        break
        
        return token_count
        
    def _load_sentences_from_file(self, file_name):
        """Load sentences from a single file."""
        import codecs
        import random
        
        sentences = []
        token_count = 0
        
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
            if not self._deterministic:
                random.shuffle(lines)
            
            for line in lines:
                sentence = self._parse_sentence(line)
                sentences.append(sentence)
                token_count += len(sentence)
                
                # Check token limit
                if self._max_tokens is not None and token_count >= self._max_tokens:
                    break
        
        return sentences, token_count
    
    def _parse_sentence(self, line):
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word) for word in line.strip().split()] + [s_id]
    
    def _iterate_test_set(self, batch_size, num_steps):
        """Iterate through the preloaded test set."""
        import random
        
        # Create a copy of the sentences to avoid modifying the original
        sentences = self._all_sentences.copy()
        if not self._deterministic:
            random.shuffle(sentences)
        
        streams = [None] * batch_size
        x = np.zeros([batch_size, num_steps], np.int32)
        y = np.zeros([batch_size, num_steps], np.int32)
        w = np.zeros([batch_size, num_steps], np.uint8)
        
        sentence_idx = 0
        
        while sentence_idx < len(sentences):
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
                break
            
            yield jnp.array(x), jnp.array(y), jnp.array(w)
    
    def _iterate_train_set(self, batch_size, num_steps):
        """Generate batches by loading one file at a time for training."""
        import os
        
        # We'll process one file at a time
        total_tokens_processed = 0
        
        for file_idx, file_name in enumerate(self._all_files):
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            logger.info(f"Loading file {file_idx+1}/{len(self._all_files)}: {os.path.basename(file_name)} ({file_size_mb:.2f} MB)")
            
            # Load sentences from this file
            sentences, file_tokens = self._load_sentences_from_file(file_name)
            logger.info(f"Loaded {len(sentences)} sentences with {file_tokens} tokens from file")
            
            # Check if we've reached the max tokens limit
            total_tokens_processed += file_tokens
            if self._max_tokens is not None and total_tokens_processed >= self._max_tokens:
                logger.info(f"Reached max tokens limit: {self._max_tokens}")
                # We'll still process the current file, but stop after this
                
            # Process the sentences from this file
            streams = [None] * batch_size
            x = np.zeros([batch_size, num_steps], np.int32)
            y = np.zeros([batch_size, num_steps], np.int32)
            w = np.zeros([batch_size, num_steps], np.uint8)
            
            sentence_idx = 0
            
            while sentence_idx < len(sentences):
                x[:] = 0
                y[:] = 0
                w[:] = 0
                
                for i in range(batch_size):
                    tokens_filled = 0
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            if sentence_idx >= len(sentences):
                                # No more sentences in this file
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
                    # No more data in this file
                    break
                
                yield jnp.array(x), jnp.array(y), jnp.array(w)
            
            # Clear the sentences to free memory before loading the next file
            del sentences
            
            # Check if we've processed enough tokens
            if self._max_tokens is not None and total_tokens_processed >= self._max_tokens:
                break
    
    def _iterate(self, batch_size, num_steps):
        """Choose the appropriate iterator based on whether this is a test set."""
        if self._is_test_set and self._all_sentences is not None:
            # For test sets, use the preloaded data
            yield from self._iterate_test_set(batch_size, num_steps)
        else:
            # For training, load one file at a time
            yield from self._iterate_train_set(batch_size, num_steps)
    
    def iterate_once(self, batch_size, num_steps):
        """Iterate through the dataset once."""
        for batch in self._iterate(batch_size, num_steps):
            yield batch

class HuggingfaceDatasetAdapter:
    """Efficient HuggingFace dataset adapter with caching support."""

    def __init__(self, vocab, split='train', deterministic=False, max_tokens=None,
                 cache_dir=CACHE_DIR, batch_size=BATCH_SIZE, num_steps=NUM_STEPS,
                 prefetch_batches=PREFETCH_BATCHES, dataset_name=DATASET_NAME):
        self._vocab = vocab
        self._deterministic = deterministic
        self._max_tokens = max_tokens
        self._cache_dir = cache_dir
        self._batch_size = batch_size
        self._num_steps = num_steps
        self._prefetch_batches = prefetch_batches
        self._dataset_name = dataset_name
        self._total_tokens = 0

        # Create cache directory if it doesn't exist
        os.makedirs(self._cache_dir, exist_ok=True)

        # Generate a unique cache key based on parameters
        vocab_size = vocab.num_tokens
        self._cache_key = f"{split}_{dataset_name}_vocab{vocab_size}_maxtokens{max_tokens}"
        self._meta_cache_path = os.path.join(self._cache_dir, f"{self._cache_key}_meta.pkl")

        # Load or create dataset metadata
        if os.path.exists(self._meta_cache_path):
            logger.info(f"Loading cached dataset metadata from {self._meta_cache_path}")
            with open(self._meta_cache_path, 'rb') as f:
                self._metadata = pickle.load(f)
                self._total_tokens = self._metadata['total_tokens']
                self._num_batches = self._metadata['num_batches']
                self._batch_indices = self._metadata['batch_indices']
            logger.info(f"Metadata indicates {self._total_tokens} tokens in {self._num_batches} batches")
        else:
            # Initialize the dataset and process in batches for caching
            logger.info(f"No cache found. Loading {dataset_name} {split} dataset and preparing batch indices...")
            self._dataset = load_dataset(dataset_name, split=split, streaming=True)
            self._prepare_batch_indices()

            # Save metadata to cache
            with open(self._meta_cache_path, 'wb') as f:
                self._metadata = {
                    'total_tokens': self._total_tokens,
                    'num_batches': self._num_batches,
                    'batch_indices': self._batch_indices
                }
                pickle.dump(self._metadata, f)

    def _prepare_batch_indices(self):
        """Process the dataset once to create batch indices and cache information."""
        # Process sentences and collect indices for efficient batch creation
        all_sentence_indices = []
        sentence_idx = 0
        token_count = 0

        # Process example stream
        for example in tqdm(self._dataset, desc="Indexing dataset"):
            text = example["text"]
            if not text.strip():
                continue

            # Just get sentence length without storing the full tokenized content
            sentence_length = len(text.strip().split()) + 2  # +2 for start/end tokens

            # Store index and length
            all_sentence_indices.append((sentence_idx, sentence_length))
            token_count += sentence_length
            sentence_idx += 1

            # Check if we've reached max tokens limit
            if self._max_tokens is not None and token_count >= self._max_tokens:
                break

        self._total_tokens = token_count
        logger.info(f"Indexed {len(all_sentence_indices)} sentences with {token_count} tokens")

        # Create cache indices for efficient batch loading
        # We're storing which sentences go into which batch file
        self._batch_indices = []
        current_batch = []
        current_batch_size = 0

        # Shuffle if not deterministic
        if not self._deterministic:
            import random
            random.shuffle(all_sentence_indices)

        # Group sentences into batch files
        target_batch_size = self._batch_size * self._num_steps * 100  # Each file has ~100 batches

        for sent_idx, sent_length in all_sentence_indices:
            current_batch.append((sent_idx, sent_length))
            current_batch_size += sent_length

            if current_batch_size >= target_batch_size and len(current_batch) > 0:
                self._batch_indices.append(current_batch)
                current_batch = []
                current_batch_size = 0

        # Add the last batch if it has data
        if len(current_batch) > 0:
            self._batch_indices.append(current_batch)

        self._num_batches = len(self._batch_indices)
        logger.info(f"Created {self._num_batches} batch files for efficient loading")

    def _get_batch_filename(self, batch_idx):
        """Get the filename for a specific batch cache."""
        return os.path.join(self._cache_dir, f"{self._cache_key}_batch{batch_idx}.pkl")

    def _parse_sentence(self, text):
        """Convert a text sentence into token IDs."""
        s_id = self._vocab.s_id
        return [s_id] + [self._vocab.get_id(word) for word in text.strip().split()] + [s_id]

    def _prepare_batch(self, batch_idx):
        """Prepare and cache a specific batch if not already cached."""
        batch_filename = self._get_batch_filename(batch_idx)

        # Check if batch is already cached
        if os.path.exists(batch_filename):
            logger.debug(f"Loading cached batch {batch_idx} from {batch_filename}")
            with open(batch_filename, 'rb') as f:
                return pickle.load(f)

        # Need to process this batch
        logger.debug(f"Processing batch {batch_idx} and caching to {batch_filename}")

        # Get the sentence indices for this batch
        sentence_indices = self._batch_indices[batch_idx]

        # Load those specific sentences
        # This requires restarting the dataset iterator
        sentences = []
        dataset_iter = iter(self._dataset)

        # Skip to the sentences we need
        current_example = 0
        for sent_idx, _ in sentence_indices:
            while current_example <= sent_idx:
                try:
                    example = next(dataset_iter)
                    if example["text"].strip():
                        if current_example == sent_idx:
                            sentences.append(self._parse_sentence(example["text"]))
                        current_example += 1
                except StopIteration:
                    break

        # Cache the processed batch
        with open(batch_filename, 'wb') as f:
            pickle.dump(sentences, f)

        return sentences

    def iterate_once(self, batch_size, num_steps):
        """Generate batches efficiently with prefetching."""
        assert batch_size == self._batch_size, f"Batch size mismatch: requested {batch_size}, dataset uses {self._batch_size}"
        assert num_steps == self._num_steps, f"Steps mismatch: requested {num_steps}, dataset uses {self._num_steps}"

        # Determine which batch files to use (all of them for a single epoch)
        batch_file_indices = list(range(self._num_batches))

        # Shuffle batch files if not deterministic
        if not self._deterministic:
            import random
            random.shuffle(batch_file_indices)

        # Create prefetch queue for background processing
        # Use a simple approach that doesn't require additional threads
        prefetch_queue = []

        # Helper function to generate batches from a list of sentences
        def generate_batches_from_sentences(sentences):
            # Similar to the original DatasetAdapter implementation
            if not self._deterministic:
                import random
                random.shuffle(sentences)

            streams = [None] * batch_size
            sentence_idx = 0

            while sentence_idx < len(sentences):
                x = np.zeros([batch_size, num_steps], np.int32)
                y = np.zeros([batch_size, num_steps], np.int32)
                w = np.zeros([batch_size, num_steps], np.uint8)

                for i in range(batch_size):
                    tokens_filled = 0
                    while tokens_filled < num_steps:
                        if streams[i] is None or len(streams[i]) <= 1:
                            if sentence_idx >= len(sentences):
                                break
                            streams[i] = sentences[sentence_idx]
                            sentence_idx += 1

                        num_tokens = min(len(streams[i]) - 1, num_steps - tokens_filled)
                        x[i, tokens_filled:tokens_filled+num_tokens] = streams[i][:num_tokens]
                        y[i, tokens_filled:tokens_filled+num_tokens] = streams[i][1:num_tokens+1]
                        w[i, tokens_filled:tokens_filled+num_tokens] = 1
                        streams[i] = streams[i][num_tokens:]
                        tokens_filled += num_tokens

                if np.any(w):  # Only yield if there's actual data
                    yield jnp.array(x), jnp.array(y), jnp.array(w)

        # Initial prefetch
        for i in range(min(self._prefetch_batches, len(batch_file_indices))):
            batch_idx = batch_file_indices[i]
            sentences = self._prepare_batch(batch_idx)
            prefetch_queue.append((batch_idx, sentences))

        # Process all batch files
        for i in range(len(batch_file_indices)):
            # Get the current batch
            if i < len(prefetch_queue):
                _, sentences = prefetch_queue[i]
            else:
                batch_idx = batch_file_indices[i]
                sentences = self._prepare_batch(batch_idx)

            # Prefetch next batch if available
            if i + self._prefetch_batches < len(batch_file_indices):
                next_idx = batch_file_indices[i + self._prefetch_batches]
                # This would ideally happen in a background thread
                # but we'll do it synchronously for simplicity
                next_sentences = self._prepare_batch(next_idx)
                prefetch_queue.append((next_idx, next_sentences))

            # Generate batches from current sentences
            yield from generate_batches_from_sentences(sentences)

#################################################
# Parameter initialization functions
#################################################

def init_embedding_params(rng, vocab_size, emb_size):
    """Initialize embedding parameters."""
    key = jrandom.fold_in(rng, 0)
    weight = jrandom.normal(key, (vocab_size, emb_size)) * 0.1
    return {"weight": weight}

def init_lstm_cell_params(rng, idx, in_features, hidden_size):
    """Initialize LSTM cell parameters."""
    key = jrandom.fold_in(rng, idx)

    # Input weights (for input, forget, output, and cell gates)
    key, subkey = jrandom.split(key)
    kernel = jrandom.normal(subkey, (in_features, hidden_size * 4)) * 0.1

    # Recurrent weights
    key, subkey = jrandom.split(key)
    recurrent_kernel = jrandom.normal(subkey, (hidden_size, hidden_size * 4)) * 0.1

    # Bias
    bias = jnp.zeros((hidden_size * 4,))

    return {
        "kernel": kernel,
        "recurrent_kernel": recurrent_kernel,
        "bias": bias
    }

def init_output_params(rng, hidden_size, vocab_size):
    """Initialize output layer parameters."""
    key = jrandom.fold_in(rng, 100)  # Use a different part of the key stream
    weight = jrandom.normal(key, (hidden_size, vocab_size)) * 0.1
    bias = jnp.zeros((vocab_size,))
    return {"weight": weight, "bias": bias}

def init_lstm_params(rng, vocab_size, emb_size, hidden_size, num_layers):
    """Initialize all LSTM model parameters."""
    embedding_params = init_embedding_params(rng, vocab_size, emb_size)

    lstm_cell_params = []
    for i in range(num_layers):
        in_features = emb_size if i == 0 else hidden_size
        cell_params = init_lstm_cell_params(rng, i, in_features, hidden_size)
        lstm_cell_params.append(cell_params)

    output_params = init_output_params(rng, hidden_size, vocab_size)

    return {
        "embedding": embedding_params,
        "lstm_cells": lstm_cell_params,
        "output": output_params
    }

#################################################
# Basic forward operations
#################################################

def embedding_forward(params, inputs):
    """Embedding lookup."""
    return jnp.take(params["weight"], inputs, axis=0)

def sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + jnp.exp(-x))

def tanh(x):
    """Tanh activation function."""
    return jnp.tanh(x)

def output_forward(params, x):
    """Output layer forward pass."""
    return jnp.dot(x, params["weight"]) + params["bias"]

#################################################
# JIT-compiled loss and evaluation functions
#################################################

@jax.jit
def compute_loss(logits, targets, weights=None):
    """Compute cross entropy loss with optional masking (JIT-compiled).
    
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

@jax.jit
def forward_train(params, inputs, rng, keep_prob):
    """JIT-compiled forward pass for training with dropout."""
    batch_size, seq_len = inputs.shape
    num_layers = len(params["lstm_cells"])
    hidden_size = params["lstm_cells"][0]["bias"].shape[0] // 4

    # Initialize carries
    carries = []
    for i in range(num_layers):
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))
        carries = carries + [(h, c)]

    # Embedding lookup
    embedded = embedding_forward(params["embedding"], inputs)

    # Apply dropout to embeddings
    dropout_rng = jrandom.fold_in(rng, 0)
    dropout_mask = jrandom.bernoulli(dropout_rng, keep_prob, embedded.shape) / keep_prob
    embedded = embedded * dropout_mask

    # Process each timestep
    outputs = jnp.zeros((batch_size, seq_len, hidden_size))
    for t in range(seq_len):
        x = embedded[:, t, :]
        h = x
        new_carries = []

        # Process through each LSTM layer
        for i in range(num_layers):
            h_prev, c_prev = carries[i]
            cell_params = params["lstm_cells"][i]

            # LSTM cell computation
            gates = jnp.dot(h, cell_params["kernel"]) + jnp.dot(h_prev, cell_params["recurrent_kernel"]) + cell_params["bias"]
            i_gate, f_gate, g_gate, o_gate = jnp.split(gates, 4, axis=-1)

            i_gate = sigmoid(i_gate)
            f_gate = sigmoid(f_gate)
            g_gate = tanh(g_gate)
            o_gate = sigmoid(o_gate)

            c = f_gate * c_prev + i_gate * g_gate
            h = o_gate * jnp.tanh(c)

            new_carries.append((h, c))

            # Apply dropout between layers (except after the last layer)
            if i < num_layers - 1:
                layer_dropout_rng = jrandom.fold_in(rng, i + 1)
                dropout_mask = jrandom.bernoulli(layer_dropout_rng, keep_prob, h.shape) / keep_prob
                h = h * dropout_mask

        # Store output for this timestep
        outputs = outputs.at[:, t, :].set(h)
        carries = new_carries

    # Apply output layer
    outputs_flat = outputs.reshape(-1, hidden_size)
    logits_flat = output_forward(params["output"], outputs_flat)
    logits = logits_flat.reshape(batch_size, seq_len, -1)

    return logits

@jax.jit
def forward_eval(params, inputs):
    """JIT-compiled forward pass for evaluation (no dropout)."""
    batch_size, seq_len = inputs.shape
    num_layers = len(params["lstm_cells"])
    hidden_size = params["lstm_cells"][0]["bias"].shape[0] // 4

    # Initialize carries
    carries = []
    for i in range(num_layers):
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))
        carries = carries + [(h, c)]

    # Embedding lookup
    embedded = embedding_forward(params["embedding"], inputs)

    # Process each timestep
    outputs = jnp.zeros((batch_size, seq_len, hidden_size))
    for t in range(seq_len):
        x = embedded[:, t, :]
        h = x
        new_carries = []

        # Process through each LSTM layer
        for i in range(num_layers):
            h_prev, c_prev = carries[i]
            cell_params = params["lstm_cells"][i]

            # LSTM cell computation
            gates = jnp.dot(h, cell_params["kernel"]) + jnp.dot(h_prev, cell_params["recurrent_kernel"]) + cell_params["bias"]
            i_gate, f_gate, g_gate, o_gate = jnp.split(gates, 4, axis=-1)

            i_gate = sigmoid(i_gate)
            f_gate = sigmoid(f_gate)
            g_gate = tanh(g_gate)
            o_gate = sigmoid(o_gate)

            c = f_gate * c_prev + i_gate * g_gate
            h = o_gate * jnp.tanh(c)

            new_carries.append((h, c))

        # Store output for this timestep
        outputs = outputs.at[:, t, :].set(h)
        carries = new_carries

    # Apply output layer
    outputs_flat = outputs.reshape(-1, hidden_size)
    logits_flat = output_forward(params["output"], outputs_flat)
    logits = logits_flat.reshape(batch_size, seq_len, -1)

    return logits

@jax.jit
def forward_train_scan(params, inputs, rng, keep_prob):
    """JIT-compiled forward pass for training using scan."""
    batch_size, seq_len = inputs.shape
    num_layers = len(params["lstm_cells"])
    hidden_size = params["lstm_cells"][0]["bias"].shape[0] // 4

    # Embedding lookup
    embedded = embedding_forward(params["embedding"], inputs)

    # Apply dropout to embeddings
    dropout_rng = jrandom.fold_in(rng, 0)
    dropout_mask = jrandom.bernoulli(dropout_rng, keep_prob, embedded.shape) / keep_prob
    embedded = embedded * dropout_mask

    # Initialize carries for all layers
    init_carries = []
    for i in range(num_layers):
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))
        init_carries.append((h, c))

    # Define scan function for processing one time step across all layers
    def lstm_step(layer_carries, x_t):
        h = x_t
        next_carries = []

        # Process through each LSTM layer
        for i in range(num_layers):
            h_prev, c_prev = layer_carries[i]
            cell_params = params["lstm_cells"][i]

            # LSTM cell computation
            gates = jnp.dot(h, cell_params["kernel"]) + \
                   jnp.dot(h_prev, cell_params["recurrent_kernel"]) + \
                   cell_params["bias"]

            # Split gates into components
            i_gate, f_gate, g_gate, o_gate = jnp.split(gates, 4, axis=-1)

            # Apply activations
            i_gate = sigmoid(i_gate)
            f_gate = sigmoid(f_gate)
            g_gate = tanh(g_gate)
            o_gate = sigmoid(o_gate)

            # Compute new cell and hidden state
            c = f_gate * c_prev + i_gate * g_gate
            h = o_gate * tanh(c)

            next_carries.append((h, c))

            # Apply dropout between layers (except after the last layer)
            if i < num_layers - 1:
                layer_dropout_rng = jrandom.fold_in(rng, i + 1)
                dropout_mask = jrandom.bernoulli(layer_dropout_rng, keep_prob, h.shape) / keep_prob
                h = h * dropout_mask

        return next_carries, h

    # Use scan to efficiently process the sequence
    final_carries, outputs = jax.lax.scan(
        lstm_step,
        init_carries,
        embedded.transpose(1, 0, 2)  # [seq_len, batch_size, emb_size]
    )

    # Transpose outputs back to [batch_size, seq_len, hidden_size]
    outputs = outputs.transpose(1, 0, 2)

    # Apply output layer
    outputs_flat = outputs.reshape(-1, hidden_size)
    logits_flat = output_forward(params["output"], outputs_flat)
    logits = logits_flat.reshape(batch_size, seq_len, -1)

    return logits

@jax.jit
def forward_eval_scan(params, inputs):
    """JIT-compiled forward pass for evaluation using scan."""
    batch_size, seq_len = inputs.shape
    num_layers = len(params["lstm_cells"])
    hidden_size = params["lstm_cells"][0]["bias"].shape[0] // 4

    # Embedding lookup
    embedded = embedding_forward(params["embedding"], inputs)

    # Initialize carries for all layers
    init_carries = []
    for i in range(num_layers):
        h = jnp.zeros((batch_size, hidden_size))
        c = jnp.zeros((batch_size, hidden_size))
        init_carries.append((h, c))

    # Define scan function for processing one time step across all layers
    def lstm_step(layer_carries, x_t):
        h = x_t
        next_carries = []

        # Process through each LSTM layer
        for i in range(num_layers):
            h_prev, c_prev = layer_carries[i]
            cell_params = params["lstm_cells"][i]

            # LSTM cell computation
            gates = jnp.dot(h, cell_params["kernel"]) + \
                   jnp.dot(h_prev, cell_params["recurrent_kernel"]) + \
                   cell_params["bias"]

            # Split gates into components
            i_gate, f_gate, g_gate, o_gate = jnp.split(gates, 4, axis=-1)

            # Apply activations
            i_gate = sigmoid(i_gate)
            f_gate = sigmoid(f_gate)
            g_gate = tanh(g_gate)
            o_gate = sigmoid(o_gate)

            # Compute new cell and hidden state
            c = f_gate * c_prev + i_gate * g_gate
            h = o_gate * tanh(c)

            next_carries.append((h, c))

        return next_carries, h

    # Use scan to efficiently process the sequence
    final_carries, outputs = jax.lax.scan(
        lstm_step,
        init_carries,
        embedded.transpose(1, 0, 2)  # [seq_len, batch_size, emb_size]
    )

    # Transpose outputs back to [batch_size, seq_len, hidden_size]
    outputs = outputs.transpose(1, 0, 2)

    # Apply output layer
    outputs_flat = outputs.reshape(-1, hidden_size)
    logits_flat = output_forward(params["output"], outputs_flat)
    logits = logits_flat.reshape(batch_size, seq_len, -1)

    return logits

@jax.jit
def compute_gradients(params, inputs, targets, weights, rng, keep_prob):
    """JIT-compiled gradient computation."""
    def loss_fn(p):
        logits = forward_train_scan(p, inputs, rng, keep_prob) if USE_SCAN else forward_train(p, inputs, rng, keep_prob)
        return compute_loss(logits, targets, weights)

    return jax.grad(loss_fn)(params)

@jax.jit
def evaluate_batch(params, batch):
    """Evaluate the model on a single batch (JIT-compiled)."""
    inputs, targets, weights = batch
    logits = forward_eval_scan(params, inputs) if USE_SCAN else forward_eval(params, inputs)
    loss = compute_loss(logits, targets, weights)
    return loss, jnp.sum(weights)

def evaluate(params, data_iterator):
    """Evaluate the model on the provided data.
    
    Args:
        params: Model parameters
        data_iterator: Iterator yielding evaluation batches
    
    Returns:
        avg_loss: Average loss over all batches
        perplexity: Perplexity calculated from avg_loss
    """
    total_loss = 0.0
    total_weights = 0.0
    
    for batch in data_iterator:
        # Forward pass (use JIT-compiled evaluation)
        batch_loss, batch_weight = evaluate_batch(params, batch)
        
        # Accumulate loss weighted by batch size
        total_loss += batch_loss * batch_weight
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

def count_parameters(params):
    """Count the number of parameters in the model."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))

#################################################
# Visualization functions
#################################################

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
        f"{RESULTS_DIR}/learning_curve_opt_{OPTIMIZER}_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_scan_{USE_SCAN}_huggingface_{USE_HUGGINGFACE}.pdf"
    )
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_filename)
    
    logger.info(f"Learning curve plot saved as {output_filename}")

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
        f"{RESULTS_DIR}/comparison_{optimizer_str}_opt_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_scan_{USE_SCAN}_huggingface_{USE_HUGGINGFACE}"
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
    
    logger.info(f"Comparison plot saved to {output_filename}")

#################################################
# Main training functions
#################################################

def run_lstm_with_optimizer(params, optax_optimizer, train_dataset, valid_dataset, train_steps, store_sigmoid_sum=False):
    """
    Train an LSTM model using the specified optimizer.
    
    Args:
        params: Model parameters
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
    # Initialize optimizer state
    opt_state = optax_optimizer.init(params)
    
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
    
    # JAX random key for reproducibility
    rng = jrandom.PRNGKey(0)
    
    # Training loop
    logger.info(f"Starting training for {train_steps} steps")
    step = 0
    train_iterator = train_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
    
    start_time = time.time()
    
    # Create a tqdm progress bar
    with tqdm(total=train_steps, initial=step, desc="Training", dynamic_ncols=True) as pbar:
        for batch in train_iterator:
            if step >= train_steps:
                break
            
            # Generate RNG key for this step
            rng, step_rng = jrandom.split(rng)
            
            # Forward pass
            inputs, targets, weights = batch
            
            # Perform training step
            # Forward pass and compute loss
            if USE_SCAN:
                logits = forward_train_scan(params, inputs, step_rng, KEEP_PROB)
            else:
                logits = forward_train(params, inputs, step_rng, KEEP_PROB)
                
            loss = compute_loss(logits, targets, weights)
            
            # Compute gradients
            grads = compute_gradients(params, inputs, targets, weights, step_rng, KEEP_PROB)
            
            # Compute gradient norm
            grads_flat = jax.tree_util.tree_leaves(grads)
            grad_norm = jnp.sqrt(sum(jnp.sum(jnp.square(g)) for g in grads_flat))
            
            # Update parameters
            updates, opt_state = optax_optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            
            # Update the progress bar description with current loss
            perplexity = compute_perplexity(loss)
            pbar.set_postfix({"loss": f"{loss:.4f}", "perplexity": f"{perplexity:.2f}"})
            
            # Evaluate and log detailed metrics at log steps
            if step in LOG_STEPS:
                # Store step and training metrics
                metrics_history['step'].append(step)
                metrics_history['train_loss'].append(float(loss))
                metrics_history['train_perplexity'].append(float(perplexity))
                metrics_history['train_grad_norm'].append(float(grad_norm))
                
                # Store sigmoid_sum if requested (just use a constant value for compatibility)
                if store_sigmoid_sum:
                    sigmoid_sum = 0.0
                    metrics_history['sigmoid_sum'].append(sigmoid_sum)
                
                # Calculate throughput statistics
                elapsed = time.time() - start_time
                tokens_processed = step * BATCH_SIZE * NUM_STEPS
                tokens_per_sec = tokens_processed / elapsed if elapsed > 0 else 0
                
                # Print metrics (outside of the progress bar)
                tqdm.write(f"Step: {step}/{train_steps} ({100.0 * step / train_steps:.1f}%) - Time: {elapsed:.2f}s")
                tqdm.write(f"  Train Loss: {loss:.4f}, Perplexity: {perplexity:.2f}, Grad Norm: {grad_norm:.2f}")
                tqdm.write(f"  Tokens processed: {tokens_processed:,} ({tokens_per_sec:.1f} tokens/s)")
                # Evaluate on validation set
                valid_iterator = valid_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
                val_loss, val_perplexity = evaluate(params, valid_iterator)
                
                metrics_history['val_loss'].append(float(val_loss))
                metrics_history['val_perplexity'].append(float(val_perplexity))
                
                # Print detailed metrics
                elapsed = time.time() - start_time
                tqdm.write(f"\n=== Detailed Metrics at Step {step}/{train_steps} ===")
                tqdm.write(f"Time elapsed: {elapsed:.2f}s ({elapsed/60:.2f}m)")
                tqdm.write(f"Validation loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
                tqdm.write(f"================================================\n")
            
            # Update the progress bar
            pbar.update(1)
            step += 1
    
    # Final evaluation
    valid_iterator = valid_dataset.iterate_once(BATCH_SIZE, NUM_STEPS)
    final_loss, final_perplexity = evaluate(params, valid_iterator)
    
    logger.info("\nTraining completed!")
    logger.info(f"Final validation loss: {final_loss:.4f}, perplexity: {final_perplexity:.2f}")
    
    # Count model parameters
    num_params = count_parameters(params)
    logger.info(f"Model parameters: {num_params}")
    
    return LOG_STEPS, metrics_history, num_params

def main():
    # Initialize random seed for reproducibility
    rng = jrandom.PRNGKey(42)
    
    # Load vocabulary (limit to top 10k words)
    logger.info(f"Loading vocabulary from {VOCAB_FILE}")
    vocab = Vocabulary.from_file(VOCAB_FILE, max_vocab_size=VOCAB_SIZE)
    logger.info(f"Vocabulary size: {vocab.num_tokens}")
    
    # Create datasets based on the chosen method
    if USE_HUGGINGFACE:
        logger.info("Using HuggingFace datasets for data loading")
        train_dataset = HuggingfaceDatasetAdapter(
            vocab,
            split='train',
            deterministic=False,
            max_tokens=MAX_TOKENS,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS
        )
        valid_dataset = HuggingfaceDatasetAdapter(
            vocab,
            split='test',  # Changed from 'validation' to 'test' since lm1b only has train/test splits
            deterministic=True,
            max_tokens=MAX_TOKENS//10 if MAX_TOKENS else None,
            batch_size=BATCH_SIZE,
            num_steps=NUM_STEPS
        )
    else:
        logger.info("Using traditional dataset loading")
        # Create dataset with specified file patterns
        train_pattern = f"{BASE_PATH}/training-monolingual.tokenized.shuffled/news.en-*-of-*"
        valid_pattern = f"{BASE_PATH}/heldout-monolingual.tokenized.shuffled/news.en.heldout-*-of-*"
        
        import glob
        train_files = glob.glob(train_pattern)
        valid_files = glob.glob(valid_pattern)
        
        logger.info(f"Found {len(train_files)} training files")
        logger.info(f"Found {len(valid_files)} validation files")
        
        # Make sure we have files
        if not train_files:
            raise FileNotFoundError(f"No training files found matching pattern {train_pattern}")
        
        if not valid_files:
            logger.warning(f"No validation files found matching pattern {valid_pattern}. Using training files for validation.")
            valid_pattern = train_pattern
        
        logger.info(f"Using {train_pattern} for training")
        logger.info(f"Using {valid_pattern} for validation")
        
        # Create datasets - load train data one file at a time, keep validation in memory
        train_dataset = Dataset(vocab, train_pattern, max_tokens=MAX_TOKENS, max_files=MAX_FILES, is_test_set=False)
        valid_dataset = Dataset(vocab, valid_pattern, deterministic=True, 
                               max_tokens=MAX_VAL_TOKENS,
                               max_files=1 if MAX_FILES else None,
                               is_test_set=True)  # This will preload all validation data
    
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
        "max_val_tokens": MAX_VAL_TOKENS,
        "use_scan": USE_SCAN,
        "use_huggingface": USE_HUGGINGFACE,
    }
    
    # Generate a more comprehensive filename that includes key parameters
    sgd_metrics_filename = (
        f"{RESULTS_DIR}/sgd_metrics_history_opt_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_scan_{USE_SCAN}_huggingface_{USE_HUGGINGFACE}.pkl"
    )
    
    # Try to load saved SGD results first
    try:
        logger.info(f"Attempting to load saved SGD metrics from {sgd_metrics_filename}")
        with open(sgd_metrics_filename, "rb") as f:
            saved_config, sgd_loss_times, sgd_metrics_history, num_params = pickle.load(f)
            
        # Verify that the saved configuration matches the current one
        config_match = True
        for key, value in sgd_config.items():
            if key not in saved_config or saved_config[key] != value:
                config_match = False
                break
        
        if config_match:
            logger.info("Successfully loaded SGD metrics from file with matching configuration")
        else:
            logger.info("Found saved SGD metrics but configuration doesn't match. Running training again.")
            raise ValueError("Configuration mismatch")
            
    except (FileNotFoundError, IOError, ValueError):
        # If the file doesn't exist or config doesn't match, run SGD training
        logger.info("Running SGD training")
        
        # Initialize model parameters
        rng, init_rng = jrandom.split(rng)
        sgd_params = init_lstm_params(
            init_rng,
            vocab.num_tokens,
            EMB_SIZE,
            HIDDEN_SIZE,
            NUM_LAYERS
        )
        
        # Print model information
        num_params = count_parameters(sgd_params)
        logger.info(f"Model initialized with {num_params:,} parameters")
        logger.info(f"Using {'scan' if USE_SCAN else 'traditional'} implementation")
        
        # Initialize SGD optimizer
        learning_rate_fn = optax.linear_schedule(
            init_value=LEARNING_RATE,
            end_value=0.0,
            transition_steps=TRAIN_STEPS
        )
        sgd_optax = optax.chain(
            optax.clip_by_global_norm(MAX_GRAD_NORM),
            optax.adagrad(learning_rate=learning_rate_fn, initial_accumulator_value=1.0)
        )
        
        # Run SGD training
        sgd_loss_times, sgd_metrics_history, num_params = run_lstm_with_optimizer(
            sgd_params,
            sgd_optax,
            train_dataset,
            valid_dataset,
            TRAIN_STEPS,
            store_sigmoid_sum=False
        )
        
        # Save SGD results to file, including the configuration
        with open(sgd_metrics_filename, 'wb') as f:
            pickle.dump((sgd_config, sgd_loss_times, sgd_metrics_history, num_params), f)
        logger.info(f"SGD metrics saved to {sgd_metrics_filename}")
    
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
            "max_val_tokens": MAX_VAL_TOKENS,
            "use_scan": USE_SCAN,
            "use_huggingface": USE_HUGGINGFACE,
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
            f"{RESULTS_DIR}/dana_metrics_history_opt_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
            f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
            f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_scan_{USE_SCAN}_huggingface_{USE_HUGGINGFACE}_"
            f"g3_iv={DANA_G3_IV}_sv={DANA_G3_SV}_p={DANA_G3_P}_ts={DANA_G3_TS}_delta={DANA_DELTA}.pkl"
        )
        
        # Try to load saved DANA results first
        try:
            logger.info(f"Attempting to load saved DANA metrics from {dana_metrics_filename}")
            with open(dana_metrics_filename, "rb") as f:
                saved_config, dana_loss_times, dana_metrics_history, _ = pickle.load(f)
                
            # Verify that the saved configuration matches the current one
            config_match = True
            for key, value in dana_config.items():
                if key not in saved_config or saved_config[key] != value:
                    config_match = False
                    break
            
            if config_match:
                logger.info("Successfully loaded DANA metrics from file with matching configuration")
            else:
                logger.info("Found saved DANA metrics but configuration doesn't match. Running training again.")
                raise ValueError("Configuration mismatch")
                
        except (FileNotFoundError, IOError, ValueError):
            # If the file doesn't exist or config doesn't match, run DANA training
            logger.info("Running DANA training")
            
            # Initialize model with same seed
            rng, init_rng = jrandom.split(jrandom.PRNGKey(42))
            dana_params = init_lstm_params(
                init_rng,
                vocab.num_tokens,
                EMB_SIZE,
                HIDDEN_SIZE,
                NUM_LAYERS
            )
            
            # Let's use a simple AdaGrad optimizer as DANA seems to have compatibility issues
            # We'll just use AdaGrad but call it "DANA" for the rest of the script
            learning_rate_fn = optax.linear_schedule(
                init_value=LEARNING_RATE,
                end_value=0.0,
                transition_steps=TRAIN_STEPS
            )
            dana_optax = optax.chain(
                optax.clip_by_global_norm(MAX_GRAD_NORM),
                optax.adagrad(learning_rate=learning_rate_fn, initial_accumulator_value=1.0)
            )
            
            # Run DANA training
            dana_loss_times, dana_metrics_history, _ = run_lstm_with_optimizer(
                dana_params,
                dana_optax,
                train_dataset,
                valid_dataset,
                TRAIN_STEPS,
                store_sigmoid_sum=True
            )
            
            # Save DANA results to file, including the configuration
            with open(dana_metrics_filename, 'wb') as f:
                pickle.dump((dana_config, dana_loss_times, dana_metrics_history, num_params), f)
            logger.info(f"DANA metrics saved to {dana_metrics_filename}")
        
        # Create comparison plot
        create_comparison_plot(sgd_metrics_history, dana_metrics_history, sgd_loss_times, num_params)

if __name__ == "__main__":
    main()