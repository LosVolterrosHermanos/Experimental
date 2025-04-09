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

def extract_filters(filename):
    """Extract the number of filters from a filename"""
    # Find the part that starts with 'filters_' and ends with '_g3' or '.pkl'
    start = filename.find('filters_') + len('filters_')
    if '_g3' in filename:
        end = filename.find('_g3')
    else:
        end = filename.find('.pkl')
    return int(filename[start:end])

def extract_iv(filename):
    """Extract the initialization value (iv) from a filename"""
    start = filename.find('iv=') + len('iv=')
    end = filename.find('_sv=')
    if start == -1 or end == -1:
        print(f"Warning: Could not extract iv from filename: {filename}")
        return None
    try:
        return float(filename[start:end])
    except ValueError:
        print(f"Warning: Could not convert iv to float in filename: {filename}")
        return None

def find_crossing_points(curve1_flops, curve1_losses, curve2_flops, curve2_losses):
    """Find the last crossing point between two curves"""
    # Interpolate curve2 onto curve1's flops points
    curve2_losses_interp = np.interp(curve1_flops, curve2_flops, curve2_losses)
    
    # Find where curve1 crosses below curve2
    crossings = np.where(curve1_losses < curve2_losses_interp)[0]
    
    if len(crossings) == 0:
        return None, None
    
    # Get the last crossing point
    last_crossing = crossings[-1]
    return curve1_flops[last_crossing], curve1_losses[last_crossing]

def plot_crossing_points_vs_iv(dana_files):
    """Plot crossing points' step counts vs initialization values"""
    plt.figure(figsize=(10, 6))
    
    # Process DANA files - group by filter size
    filter_groups = {}
    for pickle_file in dana_files:
        num_filters = extract_filters(pickle_file)
        if num_filters == 32:  # Skip outlier
            continue
            
        if num_filters not in filter_groups:
            filter_groups[num_filters] = []
        filter_groups[num_filters].append(pickle_file)
    
    print(f"Found {len(filter_groups)} filter groups")
    for num_filters, files in filter_groups.items():
        print(f"Filter size {num_filters} has {len(files)} files")
        for f in files:
            iv = extract_iv(f)
            print(f"  File: {f}, iv: {iv}")
    
    # Create color mapping for different filter sizes
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(filter_groups)))
    
    # Store all points for each filter size
    all_points = {}
    
    # Collect all points for global fit
    all_iv_values = []
    all_step_counts = []
    
    for idx, (num_filters, group_files) in enumerate(filter_groups.items()):
        # Sort files by iv value (descending)
        group_files.sort(key=extract_iv, reverse=True)
        
        # Store points for this filter size
        iv_values = []
        step_counts = []
        
        # Find crossing points between consecutive curves
        for i in range(len(group_files) - 1):
            file1 = group_files[i]
            file2 = group_files[i + 1]
            
            with open(file1, 'rb') as f:
                data1 = pickle.load(f)
            with open(file2, 'rb') as f:
                data2 = pickle.load(f)
            
            losses1 = data1['train_loss']
            losses2 = data2['train_loss']
            times1 = generate_loss_times(STEPS)[:len(losses1)]
            times2 = generate_loss_times(STEPS)[:len(losses2)]
            flops1 = times1 * count_parameters(num_filters) * BATCH_SIZE
            flops2 = times2 * count_parameters(num_filters) * BATCH_SIZE
            
            # Find the last crossing point
            crossing_flops, crossing_losses = find_crossing_points(flops1, losses1, flops2, losses2)
            if crossing_flops is not None:
                # Convert flops back to steps
                crossing_steps = crossing_flops / (count_parameters(num_filters) * BATCH_SIZE)
                # Store the lower iv value (from file2)
                lower_iv = extract_iv(file2)
                
                if lower_iv is not None:
                    iv_values.append(lower_iv)
                    step_counts.append(crossing_steps)
                    all_iv_values.append(lower_iv)
                    all_step_counts.append(crossing_steps)
                    print(f"Found crossing point for filters={num_filters}: iv={lower_iv}, steps={crossing_steps}")
        
        # Store points for this filter size
        all_points[num_filters] = (iv_values, step_counts)
        
        # Plot points for this filter size
        if iv_values:
            plt.loglog(step_counts, iv_values, 'o-', color=colors[idx], 
                      label=f'Filters={num_filters}')
            
            # Fit line through points
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log10(step_counts), 
                np.log10(iv_values)
            )
            x_fit = np.array([min(step_counts), max(step_counts)])
            y_fit = 10**(slope * np.log10(x_fit) + intercept)
            plt.loglog(x_fit, y_fit, '--', color=colors[idx], 
                      label=f'Fit (slope={slope:.3f})')
    
    # Add global fit using all points
    if all_iv_values and all_step_counts:
        global_slope, global_intercept, global_r_value, global_p_value, global_std_err = stats.linregress(
            np.log10(all_step_counts),
            np.log10(all_iv_values)
        )
        x_global = np.array([min(all_step_counts), max(all_step_counts)])
        y_global = 10**(global_slope * np.log10(x_global) + global_intercept)
        plt.loglog(x_global, y_global, 'k-', linewidth=2,
                  label=f'Global Fit (slope={global_slope:.3f}, intercept={global_intercept:.3f})')
    
    plt.ylabel('Initialization Value (iv)')
    plt.xlabel('Steps at Crossing Point')
    plt.title('Initialization Value vs Steps at Crossing Point')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('crossing_points_vs_iv.pdf', bbox_inches='tight')
    plt.close()
    
    return all_points

# Find and sort all files
dana_files = glob.glob("Results/dana_classic/dana_classic_metrics_history_steps_50000_filters_*_g3_iv=*_sv=0.0_p=0.0_ts=1.0_delta=8.pkl")
dana_files.sort(key=extract_filters)

print(f"Found {len(dana_files)} DANA files")
for f in dana_files:
    print(f"File: {f}")

# Generate plot
all_points = plot_crossing_points_vs_iv(dana_files) 