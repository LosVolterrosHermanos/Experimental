#!/usr/bin/env python
"""
FineWeb Dataset implementation for efficient large-scale language model training.

This implementation provides an efficient dataset class for processing FineWeb
parquet files, supporting both training and validation modes with memory-efficient
streaming and caching strategies.

Key features:
- Streaming data processing for large datasets that don't fit in memory
- Efficient validation data caching for consistent evaluation
- On-the-fly tokenization using tiktoken GPT-2 encoding
- Memory-efficient numpy arrays with appropriate dtypes
- Support for token limits and batch size configuration

The dataset follows the same interface as TextDataset but optimized for
parquet file format commonly used in large language model datasets.
"""

import os
import logging
import numpy as np
import pandas as pd
import tiktoken
import jax.numpy as jnp
from typing import List, Optional, Iterator, Tuple

# Set up logging
logger = logging.getLogger(__name__)


class FineWebDataset:
    """Dataset class for processing FineWeb parquet files efficiently.
    
    This dataset implementation is designed for large-scale language model training
    with the following optimizations:
    
    1. Streaming Processing: Processes one parquet file at a time to minimize memory usage
    2. Validation Caching: Preloads validation data into memory for consistent evaluation
    3. On-the-fly Tokenization: Tokenizes text as needed to save storage space
    4. Memory Efficiency: Uses appropriate numpy dtypes and cleanup strategies
    
    The dataset yields batches of (x, y, w) where:
    - x: Input token sequences of shape (batch_size, seq_len)
    - y: Target token sequences of shape (batch_size, seq_len) (x shifted by 1)
    - w: Weight masks of shape (batch_size, seq_len) (currently all ones)
    
    Args:
        parquet_files: List of paths to parquet files
        max_tokens: Maximum number of tokens to process (None for unlimited)
        is_validation: Whether this is a validation dataset (enables caching)
    """
    
    def __init__(self, parquet_files: List[str], max_tokens: Optional[int] = None, is_validation: bool = False):
        """Initialize the FineWeb dataset.
        
        Args:
            parquet_files: List of paths to parquet files to process
            max_tokens: Maximum number of tokens to load/process (None for unlimited)
            is_validation: If True, preload all data into memory for consistent validation
        """
        self.parquet_files = parquet_files
        self.max_tokens = max_tokens
        self.is_validation = is_validation
        
        # Initialize GPT-2 tokenizer
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens['<|endoftext|>']  # End-of-text token
        
        logger.info(f"FineWebDataset initialized with {len(parquet_files)} parquet files")
        if max_tokens:
            logger.info(f"Token limit: {max_tokens:,} tokens")
        
        # For validation datasets, preload all data into memory for reuse
        if is_validation:
            logger.info("Loading all validation data into memory for consistent evaluation")
            self._all_tokens = None
            self._load_all_validation_data()
    
    def _load_all_validation_data(self):
        """Load all validation data into memory efficiently.
        
        This method processes all parquet files and concatenates the tokenized
        text into a single numpy array stored in memory. This enables:
        1. Consistent validation evaluation across training steps
        2. Faster validation iterations (no file I/O during training)
        3. Deterministic validation results
        """
        # Use numpy arrays for memory efficiency
        token_chunks = []
        total_tokens = 0
        
        for file_idx, parquet_file in enumerate(self.parquet_files):
            logger.info(f"Loading validation file {file_idx+1}/{len(self.parquet_files)}: {os.path.basename(parquet_file)}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_file)
            
            # Process each text document
            for text in df['text']:
                # Tokenize text with end-of-text token prefix
                tokens = [self.eot]  # Start with end-of-text token
                tokens.extend(self.enc.encode_ordinary(text))
                
                # Convert to numpy array with efficient dtype
                token_array = np.array(tokens, dtype=np.int32)
                token_chunks.append(token_array)
                total_tokens += len(token_array)
                
                # Check max_tokens limit
                if self.max_tokens and total_tokens >= self.max_tokens:
                    logger.info(f"Reached validation max tokens limit: {self.max_tokens:,}")
                    # Concatenate what we have so far
                    self._all_tokens = np.concatenate(token_chunks, dtype=np.int32)
                    # Trim to exact limit
                    if len(self._all_tokens) > self.max_tokens:
                        self._all_tokens = self._all_tokens[:self.max_tokens]
                    del df  # Free memory
                    return
            
            del df  # Free memory after processing each file
        
        # Concatenate all token chunks into single array
        if token_chunks:
            self._all_tokens = np.concatenate(token_chunks, dtype=np.int32)
        else:
            self._all_tokens = np.array([], dtype=np.int32)
        
        logger.info(f"Loaded {len(self._all_tokens):,} validation tokens into memory "
                   f"({self._all_tokens.nbytes / 1024 / 1024:.2f} MB)")
        
    def iterate_once(self, batch_size: int, seq_len: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Create an iterator that yields batches of (x, y, w).
        
        This method returns an iterator that processes the dataset once,
        yielding batches in the format expected by training loops.
        
        Args:
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            
        Yields:
            Tuple of (x, y, w) where:
            - x: Input sequences of shape (batch_size, seq_len)
            - y: Target sequences of shape (batch_size, seq_len)
            - w: Weight masks of shape (batch_size, seq_len)
        """
        if self.is_validation:
            # For validation, use preloaded data
            return self._iterate_validation_data(batch_size, seq_len)
        else:
            # For training, process files one at a time
            return self._iterate_training_data(batch_size, seq_len)
    
    def _iterate_validation_data(self, batch_size: int, seq_len: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Iterate through preloaded validation data.
        
        Args:
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            
        Yields:
            Batches of validation data
        """
        if not hasattr(self, '_all_tokens') or len(self._all_tokens) == 0:
            logger.warning("No validation data available")
            return
        
        # Create batches from preloaded tokens
        n = len(self._all_tokens)
        num_batches = n // (batch_size * seq_len)
        
        if num_batches == 0:
            logger.warning("Validation set too small for given batch size and sequence length")
            return
        
        # Trim to ensure even division into batches
        tokens = self._all_tokens[:num_batches * batch_size * seq_len]
        
        # Reshape into batches: (batch_size, num_sequences_per_batch)
        token_array = np.array(tokens).reshape(batch_size, -1)
        
        # Create input/target batches
        for i in range(0, token_array.shape[1] - seq_len, seq_len):
            x = token_array[:, i:i+seq_len]          # Input sequences
            y = token_array[:, i+1:i+seq_len+1]      # Target sequences (shifted by 1)
            w = np.ones_like(x, dtype=np.uint8)      # All tokens are valid
            
            yield jnp.array(x), jnp.array(y), jnp.array(w)
    
    def _iterate_training_data(self, batch_size: int, seq_len: int) -> Iterator[Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Process training files one at a time for memory efficiency.
        
        This method streams through the parquet files, processing one file at a time
        to minimize memory usage while maintaining efficient batching.
        
        Args:
            batch_size: Number of sequences per batch
            seq_len: Length of each sequence
            
        Yields:
            Batches of training data
        """
        current_tokens = np.array([], dtype=np.int32)
        tokens_yielded = 0
        batch_size_tokens = batch_size * seq_len
        
        # Process one parquet file at a time
        for file_idx, parquet_file in enumerate(self.parquet_files):
            logger.info(f"Processing parquet file {file_idx+1}/{len(self.parquet_files)}: "
                       f"{os.path.basename(parquet_file)}")
            
            # Load the current parquet file
            df = pd.read_parquet(parquet_file)
            logger.info(f"Loaded {len(df)} text documents from {os.path.basename(parquet_file)}")
            
            # Process each text document in the file
            for text in df['text']:
                # Tokenize text on-the-fly
                tokens = [self.eot]  # Start with end-of-text token
                tokens.extend(self.enc.encode_ordinary(text))
                
                # Convert to numpy array for efficiency
                new_tokens = np.array(tokens, dtype=np.int32)
                current_tokens = np.concatenate([current_tokens, new_tokens])
                
                # Yield batches when we have enough tokens
                while len(current_tokens) >= batch_size_tokens + 1:
                    # Extract batch (need +1 for target sequence)
                    batch_tokens = current_tokens[:batch_size_tokens + 1]
                    current_tokens = current_tokens[batch_size_tokens:]
                    
                    # Convert to JAX arrays and reshape
                    x = jnp.array(batch_tokens[:-1]).reshape(batch_size, seq_len)  # Input
                    y = jnp.array(batch_tokens[1:]).reshape(batch_size, seq_len)   # Target
                    w = jnp.ones_like(x)  # Weight mask (all tokens valid)
                    
                    tokens_yielded += batch_size_tokens
                    
                    # Check max_tokens limit
                    if self.max_tokens and tokens_yielded >= self.max_tokens:
                        return
                        
                    yield x, y, w
                
                # Check max_tokens limit after processing each text
                if self.max_tokens and tokens_yielded >= self.max_tokens:
                    return
            
            # Free memory after processing each file
            del df


def create_fineweb_datasets(data_root: str, 
                           val_max_tokens: Optional[int] = None,
                           val_files_count: int = 1) -> Tuple[FineWebDataset, FineWebDataset]:
    """Create training and validation FineWeb datasets from a data directory.
    
    This utility function sets up the standard train/validation split used
    in most experiments, where the first few files are reserved for validation.
    
    Args:
        data_root: Root directory containing parquet files
        val_max_tokens: Maximum tokens to load for validation (None for unlimited)
        val_files_count: Number of files to reserve for validation
        
    Returns:
        Tuple of (train_dataset, val_dataset)
        
    Raises:
        ValueError: If no parquet files are found in the data directory
    """
    import glob
    
    # Find all parquet files
    parquet_files = sorted(glob.glob(os.path.join(data_root, "*_00000.parquet")))
    
    if not parquet_files:
        raise ValueError(f"No parquet files found in {data_root}")
    
    logger.info(f"Found {len(parquet_files)} parquet files in {data_root}")
    
    # Split files for train/validation
    val_files = parquet_files[:val_files_count]
    train_files = parquet_files[val_files_count:]
    
    logger.info(f"Using {len(train_files)} files for training")
    logger.info(f"Using {len(val_files)} files for validation")
    
    # Create datasets
    train_dataset = FineWebDataset(train_files)
    val_dataset = FineWebDataset(val_files, max_tokens=val_max_tokens, is_validation=True)
    
    if val_max_tokens:
        logger.info(f"Validation dataset limited to {val_max_tokens:,} tokens")
    
    return train_dataset, val_dataset