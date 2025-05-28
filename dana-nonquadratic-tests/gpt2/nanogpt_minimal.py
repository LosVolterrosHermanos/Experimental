from dataclasses import dataclass

import jax 
import time 
import optax
import tiktoken
from flax import linen as nn
from flax.core import FrozenDict
from flax.training.train_state import TrainState
from typing import Tuple
import jax.numpy as jnp
from tqdm import tqdm
import os
import pickle
import logging

BASE_PATH = "/home/elliotpaquette/Documents/BillionWords/lm1b"
TRAIN_PATTERN = f"{BASE_PATH}/training-monolingual.tokenized.shuffled/news.en-*-of-*"

TRAIN_STEPS = 10000
BATCH_SIZE = 32
SEQ_LEN = 32
GRAD_CLIP = 2.0
LR = 0.0001
INIT_STD = 0.02
MAX_TOKENS=None
MAX_FILES=None
TIKTOKEN_MODEL="gpt2"

LOG_STEPS = jnp.unique(jnp.concatenate([
    jnp.array([0]),
    jnp.int32(1.1**jnp.arange(1, jnp.ceil(jnp.log(TRAIN_STEPS)/jnp.log(1.1)))),
    jnp.array([TRAIN_STEPS])
]))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
  vocab_size: int = 50257
  n_head: int = 12
  n_embd: int = 768
  block_size: int = 1024
  n_layer: int = 12
  dropout_rate: float = 0.1

class CausalSelfAttention(nn.Module):
    config: ModelConfig

    def setup(self):
        # Initialize dense layers with smaller weights using kernel_init
        self.q_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )
        self.k_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )
        self.v_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )
        self.out_proj = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        assert len(x.shape) == 3
        B, T, C = x.shape  # batch size, sequence length, embedding dimensionality

        # calculate query, key, values for all heads in batch
        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # reshape to separate heads
        q = jnp.reshape(q, (B, T, self.config.n_head, C // self.config.n_head))
        k = jnp.reshape(k, (B, T, self.config.n_head, C // self.config.n_head))
        v = jnp.reshape(v, (B, T, self.config.n_head, C // self.config.n_head))

        # transpose to get (B, nh, T, hs)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # causal self-attention
        att = jnp.matmul(q, jnp.transpose(k, (0, 1, 3, 2))) * (1.0 / jnp.sqrt(k.shape[-1]))
        
        # create causal mask
        mask = jnp.tril(jnp.ones((T, T)))[None, None, :, :]  # (1, 1, T, T)
        att = jnp.where(mask, att, float('-inf'))
        att = jax.nn.softmax(att, axis=-1)
        y = jnp.matmul(att, v)  # (B, nh, T, hs)

        # re-assemble all head outputs side by side
        y = jnp.transpose(y, (0, 2, 1, 3))  # (B, T, nh, hs)
        y = jnp.reshape(y, (B, T, C))  # (B, T, C)

        # output projection
        y = self.out_proj(y)
        return y
  

class MLP(nn.Module):
    config: ModelConfig

    def setup(self):
        # Initialize with smaller weights using kernel_init
        self.fc1 = nn.Dense(
            self.config.n_embd*4,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )
        self.fc2 = nn.Dense(
            self.config.n_embd,
            kernel_init=nn.initializers.normal(stddev=INIT_STD)
        )

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = self.fc1(x)
        x = nn.gelu(x, approximate=True)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        x = self.fc2(x)
        x = nn.Dropout(rate=self.config.dropout_rate)(x, deterministic=deterministic)
        return x

class Block(nn.Module):
  config: ModelConfig

  @nn.checkpoint  # Add gradient checkpointing to save memory
  @nn.compact
  def __call__(self, x):
    x = nn.LayerNorm()(x)
    x = x + CausalSelfAttention(self.config)(x)
    x = nn.LayerNorm()(x)
    x = x + MLP(self.config)(x)
    return x
  

class GPT(nn.Module):
  config: ModelConfig
  def setup(self):
    self.wte = nn.Embed(self.config.vocab_size, self.config.n_embd)
                        #embedding_init=nn.initializers.normal(stddev=INIT_STD*0.5))
    self.wpe = nn.Embed(self.config.block_size, self.config.n_embd)
                        #embedding_init=nn.initializers.normal(stddev=INIT_STD))
    self.ln_f = nn.LayerNorm()
    self.head = nn.Dense(self.config.vocab_size, 
                         kernel_init=nn.initializers.normal(stddev=INIT_STD*0.5))

  @nn.compact
  def __call__(self, x, deterministic=False):
    
    B, T = x.shape
    assert T <= self.config.block_size

    pos     = jnp.arange(0, T)[None]
    pos_emb = self.wpe(pos)
    tok_emb = self.wte(x)
    x       = tok_emb + pos_emb

    for _ in range(self.config.n_layer):
      x = Block(self.config)(x)
    x = self.ln_f(x)
    logits = self.head(x)
    # logits = wte.attend(x) # parameter sharing
    return logits
  
  def init(self, rng):
    tokens = jnp.zeros((1, self.config.block_size), dtype=jnp.uint16)
    params = super().init(rng, tokens, True)

    return params


#################################################
# Dataset classes
#################################################

class TextDataset:
    """Dataset class for language model data with tiktoken tokenization."""
    
    def __init__(self, file_pattern, deterministic=False, max_tokens=None, max_files=None, is_test_set=False):
        self._tokenizer = tiktoken.get_encoding("gpt2")
        self._file_pattern = file_pattern
        self._deterministic = deterministic
        self._max_tokens = max_tokens
        self._max_files = max_files
        self._is_test_set = is_test_set
        self._total_tokens = 0
        self._all_tokenized_texts = None  # Will be populated for test sets
        
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
            logger.info("Loading all test set data into memory")
            self._all_tokenized_texts = []
            for file_idx, file_name in enumerate(self._all_files):
                file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
                logger.info(f"Loading test file {file_idx+1}/{len(self._all_files)}: {os.path.basename(file_name)} ({file_size_mb:.2f} MB)")
                
                tokenized_texts, file_tokens = self._load_tokenized_texts_from_file(file_name)
                self._all_tokenized_texts.extend(tokenized_texts)
                self._total_tokens += file_tokens
                
                if self._max_tokens is not None and self._total_tokens >= self._max_tokens:
                    logger.info(f"Reached max test tokens limit: {self._max_tokens}")
                    break
            
            logger.info(f"Loaded {len(self._all_tokenized_texts)} test samples with {self._total_tokens} tokens ({self._total_tokens/1000000:.2f}M)")
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
                    # Tokenize and count
                    token_ids = self._tokenizer.encode(line)
                    token_count += len(token_ids)
                    
                    if self._max_tokens is not None and token_count >= self._max_tokens:
                        break
        
        return token_count
        
    def _load_tokenized_texts_from_file(self, file_name):
        """Load and tokenize texts from a single file."""
        import codecs
        import random
        
        tokenized_texts = []
        token_count = 0
        
        with codecs.open(file_name, "r", "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
            if not self._deterministic:
                random.shuffle(lines)
            
            for line in lines:
                # Tokenize the line
                token_ids = self._tokenizer.encode(line)
                tokenized_texts.append(token_ids)
                token_count += len(token_ids)
                
                # Check token limit
                if self._max_tokens is not None and token_count >= self._max_tokens:
                    break
        
        return tokenized_texts, token_count
    
    def _iterate_test_set(self, batch_size, num_steps):
        """Iterate through the preloaded test set."""
        import random
        import numpy as np
        
        # Create a copy of the tokenized texts to avoid modifying the original
        tokenized_texts = self._all_tokenized_texts.copy()
        if not self._deterministic:
            random.shuffle(tokenized_texts)
        
        # Concatenate all tokenized texts with separator
        all_tokens = []
        for tokens in tokenized_texts:
            all_tokens.extend(tokens)
            # No need for separator as each tokenized text already has EOS
        
        # Create batches
        n = len(all_tokens)
        num_batches = n // (batch_size * num_steps)
        
        if num_batches == 0:
            logger.warning("Test set too small for batch size and sequence length")
            return
        
        # Trim to ensure even division into batches
        all_tokens = all_tokens[:num_batches * batch_size * num_steps]
        
        # Reshape into batches
        token_array = np.array(all_tokens).reshape(batch_size, -1)
        
        # Create input/target/mask batches
        for i in range(0, token_array.shape[1] - num_steps, num_steps):
            x = token_array[:, i:i+num_steps]
            y = token_array[:, i+1:i+num_steps+1]
            # Mask is all 1s (all tokens are valid)
            w = np.ones_like(x, dtype=np.uint8)
            
            yield jnp.array(x), jnp.array(y), jnp.array(w)
    
    def _iterate_train_set(self, batch_size, num_steps):
        """Generate batches by loading one file at a time for training."""
        import os
        import numpy as np
        
        # We'll process one file at a time
        total_tokens_processed = 0
        
        for file_idx, file_name in enumerate(self._all_files):
            file_size_mb = os.path.getsize(file_name) / (1024 * 1024)
            logger.info(f"Loading file {file_idx+1}/{len(self._all_files)}: {os.path.basename(file_name)} ({file_size_mb:.2f} MB)")
            
            # Load tokenized texts from this file
            tokenized_texts, file_tokens = self._load_tokenized_texts_from_file(file_name)
            logger.info(f"Loaded {len(tokenized_texts)} samples with {file_tokens} tokens from file")
            
            # Check if we've reached the max tokens limit
            total_tokens_processed += file_tokens
            if self._max_tokens is not None and total_tokens_processed >= self._max_tokens:
                logger.info(f"Reached max tokens limit: {self._max_tokens}")
                # We'll still process the current file, but stop after this
            
            # Concatenate all tokenized texts
            all_tokens = []
            for tokens in tokenized_texts:
                all_tokens.extend(tokens)
                # No need for separator as each tokenized text already has EOS
            
            # Create batches
            n = len(all_tokens)
            num_batches = n // (batch_size * num_steps)
            
            if num_batches == 0:
                logger.warning(f"File too small for batch size and sequence length, skipping: {file_name}")
                continue
            
            # Trim to ensure even division into batches
            all_tokens = all_tokens[:num_batches * batch_size * num_steps]
            
            # Reshape into batches
            token_array = np.array(all_tokens).reshape(batch_size, -1)
            
            # Create input/target/mask batches
            for i in range(0, token_array.shape[1] - num_steps, num_steps):
                x = token_array[:, i:i+num_steps]
                y = token_array[:, i+1:i+num_steps+1]
                # Mask is all 1s (all tokens are valid)
                w = np.ones_like(x, dtype=np.uint8)
                
                yield jnp.array(x), jnp.array(y), jnp.array(w)
            
            # Clear the tokenized texts to free memory before loading the next file
            del tokenized_texts
            
            # Check if we've processed enough tokens
            if self._max_tokens is not None and total_tokens_processed >= self._max_tokens:
                break
    
    def _iterate(self, batch_size, num_steps):
        """Choose the appropriate iterator based on whether this is a test set."""
        if self._is_test_set and self._all_tokenized_texts is not None:
            # For test sets, use the preloaded data
            yield from self._iterate_test_set(batch_size, num_steps)
        else:
            # For training, load one file at a time
            yield from self._iterate_train_set(batch_size, num_steps)
    
    def iterate_once(self, batch_size, num_steps):
        """Iterate through the dataset once."""
        for batch in self._iterate(batch_size, num_steps):
            yield batch

# class DataLoader:
#   def __init__(self, B, T):
#     self.current_position = 0
#     self.B = B
#     self.T = T

#     with open(DATA_PATH,"r") as f:
#       text = f.read()
#     enc = tiktoken.get_encoding("gpt2")
#     self.tokens = jnp.array(enc.encode(text))
#     print(f"loaded {len(self.tokens)} tokens in the datasets" )
#     print(f" 1 epoch = {len(self.tokens)//(B*T)} batches")

#   def next_batch(self):
#     B,T = self.B, self.T
#     buf = self.tokens[self.current_position:self.current_position+B*T+1]
#     x,y = jnp.reshape(buf[:-1],(B,T)), jnp.reshape(buf[1:],(B,T))
#     self.current_position += B*T
#     if self.current_position + B*T+1 > len(self.tokens):
#       self.current_position = 0
#     return x,y
  

def init_train_state(key, config) -> TrainState:
  model = GPT(config)
  params = model.init(key)
  optimizer = optax.chain(
      optax.clip_by_global_norm(GRAD_CLIP),
      optax.adam(LR)
  )
  train_state = TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer)
  return train_state

@jax.jit
def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[jnp.ndarray, TrainState]:

  def loss_fn(params: FrozenDict) -> jnp.ndarray:
      logits = state.apply_fn(params, x, False)
      loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

      return loss

  loss, grads = jax.value_and_grad(loss_fn, has_aux=False)(state.params)
  new_state = state.apply_gradients(grads=grads)
  return loss, new_state

def count_params(params):
    p = jax.tree_util.tree_map(lambda a: a.size if isinstance(a, jnp.ndarray) else 0, params)
    return jax.tree_util.tree_reduce(lambda a, b: a+b, p)

def main():
    config = ModelConfig()
    key = jax.random.PRNGKey(0)
    model = GPT(config)
    params = model.init(key)
    print(f"number of parameters: {count_params(params)}")

    print("initializing train state")
    state = init_train_state(key, config)

    print("initializing dataloader")
    train_dataset = TextDataset(TRAIN_PATTERN, max_tokens=MAX_TOKENS, max_files=MAX_FILES, is_test_set=False)
    train_iterator = train_dataset.iterate_once(BATCH_SIZE, SEQ_LEN)

    pbar = tqdm(range(TRAIN_STEPS), desc="Training")
    for step in pbar:
        x, y, w = next(train_iterator)
        t0 = time.time()
        loss, state = train_step(state, x, y)
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = BATCH_SIZE * SEQ_LEN
        tokens_per_second = tokens_processed / dt
        pbar.set_postfix(loss=f"{loss:.4f}", dt=f"{dt:.4f}s", tokens_per_sec=f"{tokens_per_second:.4f}")

    # Create CHECKPOINTS directory if it doesn't exist
    os.makedirs("CHECKPOINTS", exist_ok=True)

    # Save model parameters
    checkpoint_path = f"CHECKPOINTS/model_step_{TRAIN_STEPS}.pkl"
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(state.params, f)
    print(f"Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    main()
