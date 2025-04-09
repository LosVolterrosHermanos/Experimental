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
TRAIN_STEPS = 100000
NUM_LAYERS = 2
HIDDEN_SIZE = 1024
EMB_SIZE = 512
LEARNING_RATE = 0.1
MAX_GRAD_NORM = 10.0
KEEP_PROB = 0.9
MAX_TOKENS = None

# DANA optimizer parameters
DANA_G3_IV = 0.5
DANA_G3_SV = 0.0
DANA_G3_P = -0.5
DANA_G3_TS = 1.0
DANA_DELTA = 8

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

def create_dana_powerlaw_plot():
    """Create a plot of DANA and SGD perplexity curves with power law fits."""
    # Find the DANA metrics file
    dana_metrics_filename = (
        f"{RESULTS_DIR}/dana_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_g3_iv={DANA_G3_IV}_sv={DANA_G3_SV}_"
        f"p={DANA_G3_P}_ts={DANA_G3_TS}_delta={DANA_DELTA}.pkl"
    )
    
    # Find the SGD metrics file
    sgd_metrics_filename = (
        f"{RESULTS_DIR}/sgd_metrics_history_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_maxgrad_{MAX_GRAD_NORM}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}.pkl"
    )
    
    # Load the DANA metrics
    print(f"Loading DANA metrics from {dana_metrics_filename}")
    with open(dana_metrics_filename, "rb") as f:
        saved_config, dana_loss_times, dana_metrics_history, num_params = pickle.load(f)
    
    # Load the SGD metrics
    print(f"Loading SGD metrics from {sgd_metrics_filename}")
    with open(sgd_metrics_filename, "rb") as f:
        saved_config, sgd_loss_times, sgd_metrics_history, _ = pickle.load(f)
    
    # Print debug information
    print("\nArray shapes:")
    print(f"dana_loss_times: {len(dana_loss_times)}")
    print(f"sgd_loss_times: {len(sgd_loss_times)}")
    for key in dana_metrics_history:
        print(f"{key}: {len(dana_metrics_history[key])}")
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Skip step 0 and convert to tokens
    skip = 1
    # Get the length of the data arrays
    data_length = len(dana_metrics_history['train_perplexity'])
    # Trim tokens array to match data length
    dana_tokens = np.array(dana_loss_times[skip:skip+data_length]) * BATCH_SIZE * NUM_STEPS
    sgd_tokens = np.array(sgd_loss_times[skip:skip+data_length]) * BATCH_SIZE * NUM_STEPS
    
    print(f"\nTokens shape after trimming: {dana_tokens.shape}")
    
    # Calculate the midpoint for using only the second half of the data
    midpoint = len(dana_tokens) // 2
    
    # Add vertical line at midpoint
    midpoint_tokens = dana_tokens[midpoint]
    ax.axvline(x=midpoint_tokens, color='gray', linestyle=':', alpha=0.5, 
               label='Fit start point')
    
    # Colors and line styles for the different curves
    colors = {
        'dana_train': 'blue',
        'dana_val': 'red',
        'sgd_train': 'lightblue',
        'sgd_val': 'pink'
    }
    
    # Plot DANA and SGD perplexity curves and fit power laws
    for optimizer, metrics in [('dana', dana_metrics_history), ('sgd', sgd_metrics_history)]:
        tokens = dana_tokens if optimizer == 'dana' else sgd_tokens
        
        for dataset in ('train_perplexity', 'val_perplexity'):
            # Get data
            data = np.array(metrics[dataset])
            print(f"\n{optimizer} {dataset} shape: {data.shape}")
            
            # Plot the actual data
            color = colors[f'{optimizer}_{dataset.split("_")[0]}']
            ax.loglog(tokens, data, 'o-', color=color, 
                     label=f'{optimizer.upper()} {dataset.split("_")[0]}', 
                     alpha=0.7, markersize=4)
            
            # Fit power law using only the second half of the data
            if len(tokens[midpoint:]) > 2:  # Need at least 3 points for a fit
                A, beta, fit_values, r_value = fit_power_law(tokens[midpoint:], data[midpoint:])
                
                # Plot the fit
                ax.loglog(tokens[midpoint:], fit_values, '--', color=color, 
                         label=f'{optimizer.upper()} {dataset.split("_")[0]} fit: {A:.2f} * m^({beta:.3f}), R²={r_value**2:.3f}', 
                         alpha=1.0, linewidth=2)
    
    # Add horizontal and vertical grid lines
    ax.grid(True, which='both', axis='both', linestyle='--', alpha=0.7)
    
    # Set more y-axis ticks
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3g'))
    
    # Set axis labels
    ax.set_xlabel('Training Tokens (millions)')
    ax.set_ylabel('Perplexity')
    
    # Convert x-axis to millions
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e6:.1f}'))
    
    # Add legend with two columns and smaller font
    ax.legend(loc='upper right', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
    
    # Add title
    fig.suptitle(f"DANA vs SGD Perplexity Curves ({num_params:,} parameters)\n" + 
                 f"DANA decay with exponent -0.5 (alpha=1.0)\n" + 
                 f"2-layer LSTM (4-layer network), 1024 hidden, 512 embedding",
                 y=0.95)
    
    # Generate output filename
    output_filename = (
        f"{RESULTS_DIR}/dana_sgd_perplexity_fits_steps_{TRAIN_STEPS}_layers_{NUM_LAYERS}_"
        f"hidden_{HIDDEN_SIZE}_emb_{EMB_SIZE}_lr_{LEARNING_RATE}_"
        f"keep_{KEEP_PROB}_maxtokens_{MAX_TOKENS}_g3_iv={DANA_G3_IV}_sv={DANA_G3_SV}_"
        f"p={DANA_G3_P}_ts={DANA_G3_TS}_delta={DANA_DELTA}.pdf"
    )
    
    # Adjust layout to prevent legend cutoff
    plt.subplots_adjust(right=0.85)
    plt.savefig(output_filename, bbox_inches='tight')
    
    print(f"\nDANA vs SGD perplexity fits plot saved to {output_filename}")

if __name__ == "__main__":
    create_dana_powerlaw_plot() 