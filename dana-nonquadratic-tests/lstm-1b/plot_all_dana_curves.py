import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scipy.stats as stats

# Configuration
RESULTS_DIR = "results"
BATCH_SIZE = 128
NUM_STEPS = 20
TRAIN_STEPS = 10000
NUM_LAYERS = 2
HIDDEN_SIZE = 64
EMB_SIZE = 32
LEARNING_RATE = 0.1
MAX_GRAD_NORM = 10.0
KEEP_PROB = 0.9
MAX_TOKENS = None

def fit_power_law(tokens, values):
    """Fit a power law to the data.
    
    Args:
        tokens: Array of token counts
        values: Array of values to fit
        
    Returns:
        A: Amplitude of the power law
        beta: Exponent of the power law
        fit_values: Fitted values
        r_value: R-squared value of the fit
    """
    # Convert to numpy arrays
    tokens = np.array(tokens)
    values = np.array(values)
    
    # Fit power law: value = A * (tokens)^beta
    log_tokens = np.log(tokens)
    log_values = np.log(values)
    
    # Linear fit in log-log space
    slope, intercept, r_value, p_value, std_err = stats.linregress(log_tokens, log_values)
    
    # Power law parameters: value = A * tokens^beta
    A = np.exp(intercept)
    beta = slope
    fit_values = A * (tokens ** beta)
    
    return A, beta, fit_values, r_value

def plot_all_curves():
    """Create separate plots for training and validation perplexity curves with power law fits."""
    # Find all DANA metrics files with 100000 steps
    pattern = f"{RESULTS_DIR}/dana_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_" \
             f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_" \
             f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_g3_iv=0.5_sv=0.0_p=*.pkl"
    
    dana_files = glob.glob(pattern)
    
    if not dana_files:
        print(f"No DANA metrics files found matching pattern: {pattern}")
        return
    
    # Find SGD metrics file
    sgd_file = f"{RESULTS_DIR}/sgd_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_" \
               f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_" \
               f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}.pkl"
    
    if not os.path.exists(sgd_file):
        print(f"SGD metrics file not found: {sgd_file}")
        return
    
    # Extract p values and sort files
    dana_files_with_p = [(f, float(f.split('p=')[1].split('_')[0])) for f in dana_files]
    dana_files_with_p.sort(key=lambda x: x[1])  # Sort by p value
    dana_files = [f[0] for f in dana_files_with_p]
    
    # Colors for different curves
    dana_colors = plt.cm.viridis(np.linspace(0, 1, len(dana_files)))
    sgd_color = 'red'  # Use red for SGD
    
    # Create two figures - one for training and one for validation
    fig_train, ax_train = plt.subplots(figsize=(12, 8))
    fig_val, ax_val = plt.subplots(figsize=(12, 8))
    
    # Plot SGD curve first
    print(f"Loading SGD metrics from {sgd_file}")
    with open(sgd_file, "rb") as f:
        saved_config, loss_times, metrics_history, num_params = pickle.load(f)
    
    # Skip step 0 and convert to tokens
    skip = 1
    data_length = len(metrics_history['train_perplexity'])
    tokens = np.array(loss_times[skip:skip+data_length]) * BATCH_SIZE * NUM_STEPS
    
    # Calculate the midpoint for using only the second half of the data
    midpoint = len(tokens) // 2
    
    # Plot SGD training and validation curves
    for dataset, ax in [('train_perplexity', ax_train), ('val_perplexity', ax_val)]:
        data = np.array(metrics_history[dataset])
        label_prefix = 'Train' if dataset == 'train_perplexity' else 'Val'
        
        # Plot the actual data
        ax.loglog(tokens, data, 'o-', color=sgd_color, 
                 label=f'SGD', 
                 alpha=0.7, markersize=4)
        
        # Fit power law using only the second half of the data
        if len(tokens[midpoint:]) > 2:  # Need at least 3 points for a fit
            A, beta, fit_values, r_value = fit_power_law(tokens[midpoint:], data[midpoint:])
            
            # Plot the fit
            ax.loglog(tokens[midpoint:], fit_values, '--', color=sgd_color, 
                     label=f'SGD fit: {A:.2f} * m^({beta:.3f}), R²={r_value**2:.3f}', 
                     alpha=1.0, linewidth=2)
    
    # Plot each DANA curve
    for i, dana_file in enumerate(dana_files):
        # Extract p value from filename
        p_value = float(dana_file.split('p=')[1].split('_')[0])
        
        # Load the metrics
        print(f"Loading DANA metrics from {dana_file}")
        with open(dana_file, "rb") as f:
            saved_config, loss_times, metrics_history, num_params = pickle.load(f)
        
        # Skip step 0 and convert to tokens
        skip = 1
        data_length = len(metrics_history['train_perplexity'])
        tokens = np.array(loss_times[skip:skip+data_length]) * BATCH_SIZE * NUM_STEPS
        
        # Calculate the midpoint for using only the second half of the data
        midpoint = len(tokens) // 2
        
        # Plot training and validation curves
        for dataset, ax in [('train_perplexity', ax_train), ('val_perplexity', ax_val)]:
            data = np.array(metrics_history[dataset])
            color = dana_colors[i]
            
            # Plot the actual data
            ax.loglog(tokens, data, 'o-', color=color, 
                     label=f'DANA p={p_value:.2f}', 
                     alpha=0.7, markersize=4)
            
            # Fit power law using only the second half of the data
            if len(tokens[midpoint:]) > 2:  # Need at least 3 points for a fit
                A, beta, fit_values, r_value = fit_power_law(tokens[midpoint:], data[midpoint:])
                
                # Plot the fit
                ax.loglog(tokens[midpoint:], fit_values, '--', color=color, 
                         label=f'DANA p={p_value:.2f} fit: {A:.2f} * m^({beta:.3f}), R²={r_value**2:.3f}', 
                         alpha=1.0, linewidth=2)
    
    # Configure both plots
    for ax, title in [(ax_train, "Training"), (ax_val, "Validation")]:
        # Add horizontal and vertical grid lines
        ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
        
        # Set more y-axis ticks
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
        
        # Set axis labels
        ax.set_xlabel('Training Tokens (millions)')
        ax.set_ylabel('Perplexity')
        
        # Convert x-axis to millions
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}'))
        
        # Add legend with smaller font
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, fontsize='small')
        
        # Add title
        fig = ax.figure
        fig.suptitle(f"DANA vs SGD {title} Perplexity Curves ({num_params:,} parameters)\n" + 
                     f"2-layer LSTM (4-layer network), {HIDDEN_SIZE} hidden, {EMB_SIZE} embedding",
                     y=0.95)
        
        # Adjust layout to prevent legend cutoff
        fig.subplots_adjust(right=0.85)
    
    # Generate output filenames
    output_filename_train = f"{RESULTS_DIR}/all_curves_train_perplexity_fits_steps_{TRAIN_STEPS}_hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}.pdf"
    output_filename_val = f"{RESULTS_DIR}/all_curves_val_perplexity_fits_steps_{TRAIN_STEPS}_hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}.pdf"
    
    # Save the plots
    fig_train.savefig(output_filename_train, bbox_inches='tight')
    fig_val.savefig(output_filename_val, bbox_inches='tight')
    
    print(f"\nTraining perplexity fits plot saved to {output_filename_train}")
    print(f"Validation perplexity fits plot saved to {output_filename_val}")

if __name__ == "__main__":
    plot_all_curves() 