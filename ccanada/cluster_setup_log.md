# Cluster Setup Log for Timescale Experiment Grid Search

## Date: July 8, 2025
## Experiment: Tanea G3 Momentum Grid Search

### 1. Environment Setup

#### Required Modules:
- `StdEnv/2023` - Standard environment for Compute Canada/Alliance cluster
- `python/3.11.5` - Python 3.11 with pip and build tools
- `scipy-stack/2025a` - Scientific Python stack (numpy, scipy, matplotlib, etc.)

#### Module Loading Commands:
```bash
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
```

### 2. Virtual Environment Setup

Created Python virtual environment for offline package management:

```bash
python -m venv cluster_env
source cluster_env/bin/activate
```

### 3. Package Installation

#### Virtual Environment Setup Process:
The virtual environment was set up following Alliance Canada best practices for research computing environments.

#### Initial Package Installation:
```bash
# Core ML packages
pip install huggingface_hub tiktoken
pip install -U "jax[cuda12]"
pip install flax --no-deps
pip install optax msgpack rich treescope
pip install matplotlib tqdm
pip install -e .  # Install power_law_rf module
```

#### Environment Repair and Fixes (July 8, 2025):
After initial job failures, the virtual environment required several fixes:

1. **Missing Package Dependencies**:
   ```bash
   # Re-install missing ML dependencies
   pip install huggingface_hub tiktoken
   pip install -U "jax[cuda12]"
   pip install flax --no-deps
   pip install optax msgpack rich treescope
   pip install matplotlib
   ```

2. **Power Law RF Module Reinstallation**:
   ```bash
   # The power_law_rf module was missing from the virtual environment
   pip install -e .
   ```

3. **Complete Package List Verification**:
   ```bash
   # Verify all required packages are installed
   pip list | grep -E "(jax|flax|optax|tiktoken|power-law-rf)"
   ```

#### Final Package Versions Installed:
- `huggingface_hub-0.33.2+computecanada`
- `tiktoken-0.9.0+computecanada`
- `jax-0.6.0+computecanada`
- `jaxlib-0.6.0+computecanada`
- `jax-cuda12-plugin-0.6.0+computecanada`
- `jax-cuda12-pjrt-0.6.0+computecanada`
- `flax-0.10.7`
- `optax-0.2.5+computecanada`
- `numpy-1.26.4+computecanada` (from scipy-stack)
- `scipy-1.15.1+computecanada`
- `matplotlib-3.10.0+computecanada`
- `tqdm-4.67.1+computecanada`
- `power-law-rf-0.1.1` (local editable install)

#### Additional Dependencies:
- `absl-py-2.3.1`
- `chex-0.1.89+computecanada`
- `ml-dtypes-0.5.1+computecanada`
- `opt-einsum-3.4.0+computecanada`
- `toolz-1.0.0+computecanada`
- `treescope-0.1.9+computecanada`
- `filelock-3.18.0+computecanada`
- `hf-xet-1.1.3+computecanada`
- `fonttools-4.58.5`
- `contourpy-1.3.1+computecanada`
- `cycler-0.12.1+computecanada`
- `kiwisolver-1.4.8+computecanada`

#### Notes:
- `tensorstore` and `orbax-checkpoint` were not installed due to build issues
- These are not critical for the core experiments but may affect some checkpointing features
- JAX CUDA 12 plugin is installed for GPU acceleration
- All packages use Compute Canada optimized versions where available (+computecanada suffix)

### 4. Data Preparation

#### Downloaded FineWeb dataset:
```bash
cd /home/epaq/Experimental/dana-nonquadratic-tests/gpt2
source /home/epaq/Experimental/cluster_env/bin/activate
python grab_fineweb.py
```

This downloads the FineWeb-edu 10BT sample dataset to `./fineweb-edu/` for offline use.

### 5. Tiktoken Offline Setup

**CRITICAL**: Compute nodes don't have internet access, so tiktoken cannot download tokenizer files at runtime. We must set up a proper offline cache.

#### Problem:
Initial attempts to preload tiktoken encodings failed because tiktoken stores files in a temporary cache that isn't accessible across job submissions.

#### Solution:
Manually download and cache the GPT2 tokenizer files:

```bash
# Create cache directory
mkdir -p /home/epaq/Experimental/tiktoken_cache

# Generate cache keys for GPT2 tokenizer files
source /home/epaq/Experimental/cluster_env/bin/activate
python -c "
import hashlib
vocab_bpe_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe'
encoder_json_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json'
vocab_bpe_key = hashlib.sha1(vocab_bpe_url.encode()).hexdigest()
encoder_json_key = hashlib.sha1(encoder_json_url.encode()).hexdigest()
print('vocab.bpe cache key:', vocab_bpe_key)
print('encoder.json cache key:', encoder_json_key)
"

# Download and cache the files
python -c "
import requests
import os

vocab_bpe_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe'
encoder_json_url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/encoder.json'

print('Downloading vocab.bpe...')
r1 = requests.get(vocab_bpe_url)
with open('tiktoken_cache/6d1cbeee0f20b3d9449abfede4726ed8212e3aee', 'wb') as f:
    f.write(r1.content)

print('Downloading encoder.json...')
r2 = requests.get(encoder_json_url)
with open('tiktoken_cache/6c7ea1a7e38e3a7f062df639a5b80947f075ffe6', 'wb') as f:
    f.write(r2.content)

print('Files downloaded successfully!')
"
```

#### Cache file details:
- **vocab.bpe**: `6d1cbeee0f20b3d9449abfede4726ed8212e3aee`
- **encoder.json**: `6c7ea1a7e38e3a7f062df639a5b80947f075ffe6`

#### Usage in job scripts:
All job scripts must include:
```bash
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
```

#### Test offline functionality:
```bash
source /home/epaq/Experimental/cluster_env/bin/activate
python -c "
import os
import tiktoken
os.environ['TIKTOKEN_CACHE_DIR'] = '/home/epaq/Experimental/tiktoken_cache'
enc = tiktoken.get_encoding('gpt2')
print('Success! Vocab size:', enc.n_vocab)
"
```

### 6. Job Submission Script

Created `/home/epaq/Experimental/timescale-experiment/submit_grid_search_g3_momentum.sh`:

#### Script features:
- Submits individual SLURM jobs for each parameter combination
- Uses sbatch with appropriate resource allocation:
  - 24 hours time limit
  - 32GB memory
  - 4 CPU cores
  - 1 GPU (via `--gres=gpu:1`)
- Automatically loads required modules
- Activates virtual environment
- Runs experiments in parallel

#### Grid search parameters:
- `tanea_g3`: [16E-5, 8E-5, 4E-5]
- `momentum_flavor`: ["effective-clip", "mk3"]
- Adam baselines: beta1=0.9 and beta1=0.0

#### Total jobs: 8 (6 Tanea combinations + 2 Adam baselines)

### 7. Usage Instructions

#### To run the grid search:
```bash
cd /home/epaq/Experimental/timescale-experiment
chmod +x submit_grid_search_g3_momentum.sh
./submit_grid_search_g3_momentum.sh
```

#### To monitor jobs:
```bash
squeue -u $USER
```

#### To cancel all jobs:
```bash
scancel -u $USER
```

### 8. File Locations

- Virtual environment: `/home/epaq/Experimental/cluster_env/`
- FineWeb dataset: `/home/epaq/Experimental/dana-nonquadratic-tests/gpt2/fineweb-edu/`
- Tiktoken cache: `/home/epaq/Experimental/tiktoken_cache/`
- Job submission script: `/home/epaq/Experimental/timescale-experiment/submit_grid_search_g3_momentum.sh`
- Original grid search: `/home/epaq/Experimental/timescale-experiment/grid_search_tanea_g3_momentum.sh`
- Adam resubmission script: `/home/epaq/Experimental/timescale-experiment/resubmit_adam_jobs.sh`
- Tanea resubmission script: `/home/epaq/Experimental/timescale-experiment/resubmit_tanea_jobs.sh`

### 9. Results

Results will be saved to timestamped directories in the format:
`/home/epaq/Experimental/timescale-experiment/grid_search_g3_momentum_results_YYYYMMDD_HHMMSS/`

#### Current experiment status:
- **Results directory**: `grid_search_g3_momentum_results_20250708_213413`
- **Jobs submitted**: 8 total (2 Adam baselines + 6 Tanea parameter combinations)
- **Training steps**: 120,000 per job
- **Job status**: All jobs queued and running after fixing environment issues

#### Job details:
- **Adam jobs**: 46065964 (beta1=0.9), 46065966 (beta1=0.0)
- **Tanea jobs**: 46065970, 46065972, 46065984, 46065986, 46065988, 46065990
- **Resource allocation**: 32GB RAM, 4 CPUs, 1 GPU per job
- **Time limit**: 24 hours per job

#### Latest Job Submission (July 8, 2025 - 21:34 EDT):
Successfully submitted 8 jobs with fixed virtual environment:
1. **adam_b1_0.9** (46065964)
2. **adam_b1_0.0** (46065966)  
3. **tanea_g3_16E-5_effective-clip** (46065970) - g3=16E-5
4. **tanea_g3_16E-5_mk3** (46065972) - g3=2.552e-04 (scaled)
5. **tanea_g3_8E-5_effective-clip** (46065984) - g3=8E-5
6. **tanea_g3_8E-5_mk3** (46065986) - g3=1.276e-04 (scaled)
7. **tanea_g3_4E-5_effective-clip** (46065988) - g3=4E-5
8. **tanea_g3_4E-5_mk3** (46065990) - g3=6.380e-05 (scaled)

### 10. Troubleshooting

#### Issues encountered and solutions:

1. **Tanea jobs failed with "No module named 'power_law_rf'"**:
   - **Problem**: The local `power_law_rf` module wasn't installed in the virtual environment
   - **Solution**: `pip install -e .` from the Experimental directory

2. **Adam jobs failed with tiktoken connection timeout**:
   - **Problem**: tiktoken trying to download files from `openaipublic.blob.core.windows.net` on compute nodes without internet
   - **Solution**: Manually download and cache tokenizer files, set `TIKTOKEN_CACHE_DIR` environment variable

3. **Missing matplotlib/tqdm imports**:
   - **Problem**: Additional dependencies required by training scripts
   - **Solution**: `pip install matplotlib tqdm`

4. **Scaling syntax errors in mk3 momentum**:
   - **Problem**: f-string syntax error in bash script
   - **Solution**: Fixed string quoting in Python evaluation

5. **Virtual Environment Corruption (July 8, 2025)**:
   - **Problem**: After initial setup, the virtual environment was missing critical packages
   - **Root Cause**: Incomplete package installation during initial setup
   - **Solution**: Complete environment rebuild with proper dependency installation
   - **Commands used**:
     ```bash
     source cluster_env/bin/activate
     pip install huggingface_hub tiktoken
     pip install -U "jax[cuda12]"
     pip install flax --no-deps
     pip install optax msgpack rich treescope
     pip install matplotlib
     pip install -e .
     ```

6. **Submission Script Scaling Errors**:
   - **Problem**: Python f-string syntax errors in bash script for mk3 momentum scaling
   - **Solution**: Fixed quote escaping in Python evaluation: `float('$tanea_g3')` → `float(\"$tanea_g3\")`

7. **Environment Testing and Verification**:
   - **Problem**: Need to verify environment setup before job submission
   - **Solution**: Created comprehensive test scripts:
     - `test_environment.py` - Full 7-test suite (~55 seconds)
     - `quick_test.py` - Essential 4-test verification (~2 seconds)

#### Common issues:
1. **Module loading errors**: Ensure you're on a compute node with access to the module system
2. **GPU access**: Make sure jobs are submitted with `--gres=gpu:1`
3. **Virtual environment**: Always activate the virtual environment before running experiments
4. **Network access**: Compute nodes don't have internet access, so all data and packages must be pre-loaded
5. **Tiktoken cache**: Always set `TIKTOKEN_CACHE_DIR` environment variable in job scripts

#### Environment activation sequence:
```bash
module load StdEnv/2023
module load python/3.11.5
module load scipy-stack/2025a
source /home/epaq/Experimental/cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
```

### 11. Reproducibility Notes

To reproduce this setup on a similar cluster:
1. Load equivalent Python and scientific computing modules
2. Create virtual environment and install listed packages
3. Download FineWeb dataset using the provided script
4. Preload tiktoken encodings
5. Use the sbatch submission script with appropriate resource requests for your cluster

### 12. Environment Testing

#### Test Scripts Created:
Two comprehensive test scripts were created to verify the environment setup:

1. **Full Environment Test** (`test_environment.py`):
   - **Duration**: ~55 seconds
   - **Tests**: 7 comprehensive tests
   - **Coverage**: JAX functionality, tiktoken cache, GPT2 model, FineWeb dataset, optimizers, ML libraries
   - **Usage**: `python test_environment.py`

2. **Quick Verification Test** (`quick_test.py`):
   - **Duration**: ~2 seconds  
   - **Tests**: 4 essential tests
   - **Coverage**: Core imports, tiktoken cache, optimizer imports, model/dataset classes
   - **Usage**: `python quick_test.py`

#### Test Results (July 8, 2025):
- ✅ **All 7 tests passed** in comprehensive suite
- ✅ **All 4 tests passed** in quick verification
- ✅ **Environment fully functional** for grid search experiments

#### Key Components Verified:
- **JAX**: Working with CPU (GPU available on compute nodes)
- **Tiktoken**: Properly configured with offline cache
- **Power Law RF**: Module correctly installed and accessible
- **GPT2 Model**: 162M parameter model instantiation successful
- **FineWeb Dataset**: Parquet file loading and batch generation working
- **ML Libraries**: Flax, Optax, NumPy, Matplotlib all functional

#### Testing Workflow:
```bash
# Before running experiments, verify environment
source cluster_env/bin/activate
export TIKTOKEN_CACHE_DIR=/home/epaq/Experimental/tiktoken_cache
python quick_test.py  # Fast verification
# OR
python test_environment.py  # Comprehensive testing
```

Last updated: July 8, 2025