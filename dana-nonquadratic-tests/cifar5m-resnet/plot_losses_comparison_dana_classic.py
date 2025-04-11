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
    return float(filename[start:end])

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

def plot_losses(pickle_files, title, output_file, use_colors=True):
    """Generate a plot for the given files"""
    plt.figure(figsize=(10, 6))
    
    # Create color mapping if needed
    if use_colors:
        colors = plt.cm.plasma(np.linspace(0, 0.8, len(pickle_files)))
    
    # Store final losses and flops for linear fit
    final_losses = []
    final_flops = []
    
    for idx, pickle_file in enumerate(pickle_files):
        # Extract number of filters from filename
        num_filters = extract_filters(pickle_file)
        
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
        
        # Plot the data
        if use_colors:
            plt.loglog(flops, losses, label=f'Filters={num_filters}', color=colors[idx])
        else:
            plt.loglog(flops, losses, label=f'Filters={num_filters}')
    
    # Perform linear fit on log-log data
    slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(final_flops), np.log10(final_losses))
    
    # Plot the fit line
    x_fit = np.array([min(final_flops), max(final_flops)])
    y_fit = 10**(slope * np.log10(x_fit) + intercept)
    plt.loglog(x_fit, y_fit, 'k--', label=f'Fit (slope={slope:.3f})')
    
    plt.xlabel('Flops')
    plt.ylabel('Training Loss')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    
    return final_flops, final_losses

def plot_comparison(sgd_files, dana_files):
    """Generate a comparison plot with both SGD and DANA results"""
    plt.figure(figsize=(12, 8))
    
    # Store final losses and flops for both methods
    final_losses_sgd = []
    final_flops_sgd = []
    
    # Process SGD files
    for pickle_file in sgd_files:
        num_filters = extract_filters(pickle_file)
        if num_filters == 32:  # Skip outlier
            continue
            
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        losses = data['train_loss']
        times = generate_loss_times(STEPS)[:len(losses)]
        flops = times * count_parameters(num_filters) * BATCH_SIZE
        
        final_losses_sgd.append(losses[-1])
        final_flops_sgd.append(flops[-1])
        plt.loglog(flops, losses, color='blue', alpha=0.3, label=f'SGD (Filters={num_filters})' if num_filters == sgd_files[0] else None)
    
    # Process DANA files - group by filter size
    filter_groups = {}
    for pickle_file in dana_files:
        num_filters = extract_filters(pickle_file)
        if num_filters == 32:  # Skip outlier
            continue
            
        if num_filters not in filter_groups:
            filter_groups[num_filters] = []
        filter_groups[num_filters].append(pickle_file)
    
    # Plot DANA curves and find crossing points
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(filter_groups)))
    
    for idx, (num_filters, group_files) in enumerate(filter_groups.items()):
        # Sort files by iv value (descending)
        group_files.sort(key=extract_iv, reverse=True)
        
        # Plot all curves for this filter size
        for pickle_file in group_files:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            losses = data['train_loss']
            times = generate_loss_times(STEPS)[:len(losses)]
            flops = times * count_parameters(num_filters) * BATCH_SIZE
            
            # Only add label for the first curve of each filter group
            label = f'DANA (Filters={num_filters})' if pickle_file == group_files[0] else None
            plt.loglog(flops, losses, color=colors[idx], alpha=0.3, label=label)
        
        # Find crossing points between consecutive curves for this filter size
        crossing_points_flops = []
        crossing_points_losses = []
        
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
                crossing_points_flops.append(crossing_flops)
                crossing_points_losses.append(crossing_losses)
                plt.plot(crossing_flops, crossing_losses, 'ko', color=colors[idx])
        
        # Fit line through crossing points for this filter size
        if crossing_points_flops:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                np.log10(crossing_points_flops), 
                np.log10(crossing_points_losses)
            )
            x_fit = np.array([min(crossing_points_flops), max(crossing_points_flops)])
            y_fit = 10**(slope * np.log10(x_fit) + intercept)
            plt.loglog(x_fit, y_fit, color=colors[idx], linestyle='--', 
                      label=f'DANA Fit (Filters={num_filters}, slope={slope:.3f})')
    
    # Fit line for SGD
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        np.log10(final_flops_sgd), 
        np.log10(final_losses_sgd)
    )
    x_fit = np.array([min(final_flops_sgd), max(final_flops_sgd)])
    y_fit = 10**(slope * np.log10(x_fit) + intercept)
    plt.loglog(x_fit, y_fit, 'b--', label=f'SGD Fit (slope={slope:.3f})')
    
    plt.xlabel('Flops')
    plt.ylabel('Training Loss')
    plt.title('Resnet-18 Training Loss vs Flops (SGD vs DANA Classic)')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('combined_comparison_dana_classic.pdf', bbox_inches='tight')
    plt.close()

# Find and sort all files
sgd_files = glob.glob("sgd_metrics_history_steps_50000_filters_*.pkl")
dana_files = glob.glob("dana_classic_metrics_history_steps_50000_filters_*_g3_iv=*_sv=0.0_p=0.0_ts=1.0_delta=8.pkl")
sgd_files.sort(key=extract_filters)
dana_files.sort(key=extract_filters)

# Generate individual plots
plot_losses(sgd_files, 'Resnet-18 Training Loss vs Flops (SGD)', 'sgd_losses_comparison.pdf')
plot_losses(dana_files, 'Resnet-18 Training Loss vs Flops (DANA Classic)', 'dana_classic_losses_comparison.pdf')

# Generate comparison plot
plot_comparison(sgd_files, dana_files) 