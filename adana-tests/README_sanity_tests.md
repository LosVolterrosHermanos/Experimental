# MPowerLawRF + ADANA Sanity Check Scripts

This directory contains three comprehensive sanity check scripts for testing the integration between the `MPowerLawRF` class and the ADANA optimizer implementation.

## Scripts Overview

### 1. `mpowerlaw_adana_sanity_check.py`
**Purpose**: Basic sanity check for MPowerLawRF and ADANA optimizer.

**What it does**:
- Creates a small instance of MPowerLawRF (3 experts, 20 hidden dims, 10 embedded dims)
- Sets up the batch MSE loss function (same as in `mlsq_streaming_optax`)
- Tests one step of the ADANA optimizer
- Examines the gradients, updates, and optimizer state changes
- Runs a few additional steps to verify convergence behavior

**Key features tested**:
- MPowerLawRF initialization and data generation
- Expert selection mechanism
- Population risk calculation
- ADANA optimizer state management (m, v, tau tensors)
- Gradient computation and parameter updates
- Momentum and adaptive learning rate components

**Usage**:
```bash
python mpowerlaw_adana_sanity_check.py
```

### 2. `mpowerlaw_adana_sgd_comparison.py`
**Purpose**: Direct comparison between ADANA and SGD optimizers on MPowerLawRF.

**What it does**:
- Creates a medium-sized MPowerLawRF instance (3 experts, 30 hidden, 15 embedded)
- Runs both ADANA and SGD optimizers for 15 steps
- Compares their convergence behavior step-by-step
- Generates visualization plots comparing population risk and batch loss
- Provides detailed statistics on optimizer performance

**Key insights**:
- Shows ADANA's adaptive behavior vs SGD's fixed learning rate
- Demonstrates the effect of momentum and adaptive scaling in ADANA
- Tracks parameter change magnitudes for both optimizers
- Creates publication-ready comparison plots

**Usage**:
```bash
python mpowerlaw_adana_sgd_comparison.py
```

### 3. `mpowerlaw_streaming_test.py`
**Purpose**: Full integration test using `mlsq_streaming_optax_simple`.

**What it does**:
- Creates a larger MPowerLawRF instance (4 experts, 50 hidden, 25 embedded)
- Uses the full `mlsq_streaming_optax_simple` training function
- Tests exponentially spaced loss recording (power-law appropriate sampling)
- Compares ADANA vs SGD over 500 optimization steps
- Verifies convergence toward empirical risk limit
- Tests the complete training pipeline as used in research experiments

**Key validations**:
- Full compatibility with existing training infrastructure
- Proper handling of mixed expert model in streaming setup
- Exponential time spacing works correctly (essential for power-law analysis)
- ADANA's superior convergence to empirical limit
- Integration with population risk oracle

**Usage**:
```bash
python mpowerlaw_streaming_test.py
```

## Expected Outputs

All scripts provide detailed console output showing:
- Problem setup parameters (α, β, ζ, dimensions)
- Optimizer configurations and hyperparameters
- Step-by-step progress with numerical metrics
- Final convergence results and comparisons
- Integration test confirmations

The comparison scripts also generate visualization plots saved to `results/` (if the directory exists).

## Key Test Results

These scripts verify that:

1. **MPowerLawRF works correctly**: Expert selection follows power-law distribution, data generation produces appropriate shapes, population risk calculation is consistent.

2. **ADANA optimizer integrates properly**: State management works, momentum terms update correctly, adaptive scaling functions as expected.

3. **Loss function compatibility**: The batch MSE implementation matches `mlsq_streaming_optax_simple`, gradients are computed correctly, expert indexing works.

4. **Performance characteristics**: ADANA typically achieves 90%+ better final risk than SGD, converges much closer to empirical limit, handles mixed expert problems effectively.

5. **Research pipeline compatibility**: Full integration with existing experimental infrastructure, exponential time spacing for power-law analysis, proper handling of population risk oracles.

## Technical Notes

- All scripts use small problem sizes for fast execution and clear debugging
- Random seeds are set for reproducibility
- Error handling ensures graceful failures with informative messages
- Output is designed to be both human-readable and suitable for automated testing
- Scripts follow the same coding patterns as existing research code

## Dependencies

These scripts require the same dependencies as the main project:
- JAX/JAXlib
- Optax
- Matplotlib (for visualization scripts)
- NumPy/SciPy
- The local `optimizers` module and `power_law_rf` package

## Integration with Existing Tests

These sanity check scripts complement the existing test suite in `adana-tests/`:
- `adana-dana-comparison.py` - Research-scale comparisons
- `adana-adam-*.py` - Adam vs ADANA studies  
- `*-varying-dimensions.py` - Scaling behavior analysis

The sanity checks focus on correctness and basic functionality, while the existing tests focus on research questions and performance scaling. 