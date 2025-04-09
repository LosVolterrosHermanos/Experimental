import numpy as np
import matplotlib.pyplot as plt
import pickle
import glob
import os
import scipy.stats as stats

# Constants
BATCH_SIZE = 128
STEPS = 50000

def count_parameters(num_filters):
    """Calculate the number of parameters in the ResNet model for given number of filters"""
    # Using the formula from the original code
    return 44668662 * (num_filters**2) / (128**2)

def generate_loss_times(train_steps):
    """Generate the same time sequence as used in training"""
    return np.unique(np.concatenate([
        np.array([0]),
        np.int32(1.1**np.arange(1, np.ceil(np.log(train_steps)/np.log(1.1)))),
        np.array([train_steps])
    ]))

# Find all the pickle files and sort by filter number
pickle_files = glob.glob("sgd_metrics_history_steps_50000_filters_*.pkl")
pickle_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Extract all filter numbers to determine color mapping
filter_numbers = [int(f.split('_')[-1].split('.')[0]) for f in pickle_files]
min_filters = min(filter_numbers)
max_filters = max(filter_numbers)

plt.figure(figsize=(10, 6))

# Create color mapping (shifted to avoid yellow)
colors = plt.cm.plasma(np.linspace(0, 0.8, len(filter_numbers)))

# Store final losses and flops for linear fit
final_losses = []
final_flops = []

for idx, pickle_file in enumerate(pickle_files):
    # Extract number of filters from filename
    num_filters = int(pickle_file.split('_')[-1].split('.')[0])
    
    # Skip the 32 filters case (outlier)
    if num_filters == 32:
        continue
    
    # Load the data
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the loss values
    losses = data['train_loss']
    
    # Calculate the time points using the same sequence as training
    times = generate_loss_times(STEPS)
    # Make sure we have the same number of time points as loss values
    times = times[:len(losses)]
    # Scale by parameters and batch size
    flops = times * count_parameters(num_filters) * BATCH_SIZE
    
    # Store final values for linear fit
    final_losses.append(losses[-1])
    final_flops.append(flops[-1])
    
    # Plot the data with color from plasma colormap
    plt.loglog(flops, losses, label=f'Filters={num_filters}', color=colors[idx])

# Perform linear fit on log-log data
slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(final_flops), np.log10(final_losses))

# Plot the fit line
x_fit = np.array([min(final_flops), max(final_flops)])
y_fit = 10**(slope * np.log10(x_fit) + intercept)
plt.loglog(x_fit, y_fit, 'k--', label=f'Fit (slope={slope:.3f})')

plt.xlabel('Flops')
plt.ylabel('Training Loss')
plt.title('Resnet-18 Training Loss vs Flops')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig('sgd_losses_comparison.pdf')
plt.close() 