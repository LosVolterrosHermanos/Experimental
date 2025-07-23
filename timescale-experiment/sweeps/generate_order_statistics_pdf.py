#!/usr/bin/env python3
"""
Generate comprehensive PDF report of order statistics for all layers in the model.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from pathlib import Path

def compute_tau_order_statistics_synthetic(tau_vector):
    """Compute order statistics for tau vector - adapted from nanogpt_tanea.py
    
    Args:
        tau_vector: A 1D array of non-negative tau values
        
    Returns:
        Tuple of (largest_order_stats, smallest_order_stats)
    """
    n = len(tau_vector)
    if n == 0:
        return np.array([]), np.array([])
    
    # Sort in ascending order
    sorted_tau_asc = np.sort(tau_vector)
    
    # Compute powers of 1.1 up to n
    max_k = int(np.ceil(np.log(n) / np.log(1.1)))
    indices = np.int32(1.1 ** np.arange(max_k + 1)) - 1  # 0-indexed
    
    # Remove duplicates and clamp to valid range
    indices = np.unique(indices)
    indices = np.minimum(indices, n - 1)
    
    # Get smallest and largest order statistics
    smallest_order_stats = sorted_tau_asc[indices]
    reversed_indices = (n - 1) - indices
    largest_order_stats = sorted_tau_asc[reversed_indices]
    
    return largest_order_stats, smallest_order_stats

def generate_synthetic_distributions(n_samples=10**6):
    """Generate synthetic distributions for comparison"""
    
    # Beta(1,1) - Uniform distribution
    beta_1_1 = np.random.beta(1, 1, n_samples)
    
    # Beta(2,1) - Linear distribution
    beta_2_1 = np.random.beta(2, 1, n_samples)
    
    # Zipf-like distribution: beta(100,1) * 1/(1+X) where X has probability proportional to 1/j
    max_x = 10000
    j_values = np.arange(1, max_x + 1)
    probabilities = 1.0 / j_values
    probabilities = probabilities / np.sum(probabilities)
    
    X_samples = np.random.choice(j_values, size=n_samples, p=probabilities)
    beta_samples = np.random.beta(100, 1, n_samples)
    zipf_like = beta_samples / (1.0 + X_samples)
    
    return {
        'Beta(1,1) - Uniform': beta_1_1,
        'Beta(2,1) - Linear': beta_2_1,
        'Zipf-like: Beta(100,1) * 1/(1+X)': zipf_like
    }

def create_synthetic_order_statistics_plot(tau_values, distribution_name, ax):
    """Create order statistics plot for synthetic data"""
    
    # Compute order statistics
    largest_order_stats, smallest_order_stats = compute_tau_order_statistics_synthetic(tau_values)
    
    if len(largest_order_stats) == 0:
        ax.text(0.5, 0.5, f"No data available for\n{distribution_name}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(distribution_name)
        return None
    
    # Create order indices K = 1.1^k
    order_indices = np.array([1.1 ** k for k in range(len(largest_order_stats))])
    
    # Color by order index for visualization
    color_values = np.log(order_indices + 1)
    
    # Plot both largest and smallest order statistics
    sc1 = ax.scatter(order_indices, largest_order_stats, c=color_values, 
                     cmap='plasma', alpha=0.8, label='Largest Order Statistics', marker='o', s=15)
    sc2 = ax.scatter(order_indices, smallest_order_stats, c=color_values, 
                     cmap='plasma', alpha=0.8, label='Smallest Order Statistics', marker='s', s=15)
    
    ax.set_xlabel('Order Statistic Index')
    ax.set_ylabel('Order Statistic Value')
    ax.set_title(f'{distribution_name}')
    ax.legend(fontsize=8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    return sc1

def get_nested_value(data, path):
    """Helper function to access nested dictionary with path"""
    for key in path:
        data = data[key]
    return data

def create_order_statistics_plot(data, layer_path, ax, title_prefix=""):
    """Create order statistics plot for a given layer path"""
    timestamps = data['tau_statistics']['timestamps']
    all_order_statistics = data['tau_statistics']['tau_statistics']
    
    # Prepare data for combined plot
    order_indices_largest = []
    order_values_largest = []
    order_timestamps_largest = []
    
    order_indices_smallest = []
    order_values_smallest = []
    order_timestamps_smallest = []
    
    for t_idx, (ts, arr) in enumerate(zip(timestamps, all_order_statistics)):
        try:
            # Get largest order statistics (index 0)
            arr_largest = np.asarray(get_nested_value(arr['params'], layer_path)[0])
            for order_idx, val in enumerate(arr_largest):
                # Transform k to K = (1.1 ** k)
                K = 1.1 ** order_idx
                order_indices_largest.append(K)
                order_values_largest.append(val)
                order_timestamps_largest.append(ts)
            
            # Get smallest order statistics (index 1)
            arr_smallest = np.asarray(get_nested_value(arr['params'], layer_path)[1])
            for order_idx, val in enumerate(arr_smallest):
                # Transform k to K = (1.1 ** k)
                K = 1.1 ** order_idx
                order_indices_smallest.append(K)
                order_values_smallest.append(val)
                order_timestamps_smallest.append(ts)
        except (KeyError, IndexError) as e:
            print(f"Warning: Could not access {' -> '.join(layer_path)}: {e}")
            continue
    
    if not order_indices_largest:
        ax.text(0.5, 0.5, f"No data available for\n{' -> '.join(layer_path)}", 
                transform=ax.transAxes, ha='center', va='center', fontsize=12)
        ax.set_title(f"{title_prefix}{' -> '.join(layer_path)}")
        return
    
    # Convert to numpy arrays
    order_indices_largest = np.array(order_indices_largest)
    order_values_largest = np.array(order_values_largest)
    order_timestamps_largest = np.array(order_timestamps_largest)
    
    order_indices_smallest = np.array(order_indices_smallest)
    order_values_smallest = np.array(order_values_smallest)
    order_timestamps_smallest = np.array(order_timestamps_smallest)
    
    # Use log of order_timestamps for color, but keep original timestamps for colorbar ticks
    # Filter out zero and negative values to avoid log warnings
    valid_mask_largest = order_timestamps_largest > 0
    valid_mask_smallest = order_timestamps_smallest > 0
    
    log_order_timestamps_largest = np.full_like(order_timestamps_largest, np.nan)
    log_order_timestamps_largest[valid_mask_largest] = np.log(order_timestamps_largest[valid_mask_largest])
    
    log_order_timestamps_smallest = np.full_like(order_timestamps_smallest, np.nan)
    log_order_timestamps_smallest[valid_mask_smallest] = np.log(order_timestamps_smallest[valid_mask_smallest])
    
    # Plot both largest and smallest order statistics
    sc1 = ax.scatter(order_indices_largest, order_values_largest, c=log_order_timestamps_largest, 
                     cmap='plasma', alpha=0.8, label='Largest Order Statistics', marker='o', s=10)
    sc2 = ax.scatter(order_indices_smallest, order_values_smallest, c=log_order_timestamps_smallest, 
                     cmap='plasma', alpha=0.8, label='Smallest Order Statistics', marker='s', s=10)
    
    ax.set_xlabel('Order Statistic Index')
    ax.set_ylabel('Order Statistic Value')
    layer_name = ' -> '.join(layer_path)
    ax.set_title(f'{title_prefix}{layer_name}')
    ax.legend(fontsize=8)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    return sc1

def generate_comprehensive_pdf(pickle_file_path, output_path="order_statistics_report.pdf", include_synthetic=True):
    """Generate comprehensive PDF report of order statistics"""
    
    # Load data
    print(f"Loading data from {pickle_file_path}...")
    with open(pickle_file_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get number of transformer blocks
    params = data['tau_statistics']['tau_statistics'][0]['params']
    transformer_blocks = [k for k in params.keys() if k.startswith('TransformerBlock_')]
    transformer_blocks.sort(key=lambda x: int(x.split('_')[1]))
    
    print(f"Found {len(transformer_blocks)} transformer blocks")
    
    with PdfPages(output_path) as pdf:
        # Page 0: Synthetic examples to understand the plots (optional)
        if include_synthetic:
            print("Creating synthetic examples page...")
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle('Synthetic Order Statistics Examples', fontsize=16, fontweight='bold')
            
            # Create axes with space for colorbar on the right
            gs = fig.add_gridspec(2, 2, left=0.08, right=0.82, top=0.92, bottom=0.08, hspace=0.35, wspace=0.3)
            
            # Generate synthetic data
            print("Generating synthetic distributions...")
            synthetic_data = generate_synthetic_distributions(n_samples=10**6)
            
            axes = []
            for i in range(2):
                for j in range(2):
                    axes.append(fig.add_subplot(gs[i, j]))
            
            # Plot synthetic distributions
            sc = None
            for i, (dist_name, synthetic_values) in enumerate(synthetic_data.items()):
                if i < 3:  # Only plot first 3 distributions
                    sc_temp = create_synthetic_order_statistics_plot(synthetic_values, dist_name, axes[i])
                    if sc_temp is not None:
                        sc = sc_temp
            
            # Add explanatory text in the 4th panel
            if len(axes) > 3:
                ax_text = axes[3]
                explanation_text = """
                UNDERSTANDING ORDER STATISTICS PLOTS
                
                These synthetic examples show how different 
                distributions appear in order statistics plots:
                
                • Beta(1,1): Uniform distribution [0,1]
                • Beta(2,1): Linear distribution favoring 0
                • Zipf-like: Heavy-tailed distribution
                
                Each plot shows:
                • X-axis: Order statistic index K = 1.1^k (log scale)
                • Y-axis: Order statistic values (log scale)
                • Circles: Largest order statistics
                • Squares: Smallest order statistics
                
                Real neural network tau values can be 
                compared to these reference distributions.
                """
                ax_text.text(0.05, 0.95, explanation_text, transform=ax_text.transAxes, 
                            fontsize=10, verticalalignment='top', fontfamily='monospace')
                ax_text.set_title('Plot Interpretation Guide')
                ax_text.axis('off')
            
            # Add colorbar to the right
            if sc is not None:
                cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.set_label('Log(Order Index + 1)', fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        # Page 1: Overview - head and wte
        print("Creating overview page...")
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Model Overview: Head and Word Token Embeddings', fontsize=16, fontweight='bold')
        
        # Create axes with space for colorbar on the right
        gs = fig.add_gridspec(2, 1, left=0.1, right=0.82, top=0.92, bottom=0.08, hspace=0.3)
        axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])]
        
        # Head kernel
        sc1 = create_order_statistics_plot(data, ['head', 'kernel'], axes[0])
        
        # Word token embeddings
        sc2 = create_order_statistics_plot(data, ['wte', 'embedding'], axes[1])
        
        # Add colorbar to the right
        if sc1 is not None:
            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
            cbar = fig.colorbar(sc1, cax=cbar_ax)
            cbar.set_label('Log Timestamp', fontsize=10)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Pages for each transformer block
        for block_name in transformer_blocks:
            block_num = int(block_name.split('_')[1])
            print(f"Creating page for {block_name}...")
            
            fig = plt.figure(figsize=(11, 8.5))
            fig.suptitle(f'Transformer Block {block_num} - Order Statistics', fontsize=16, fontweight='bold')
            
            # Create axes with space for colorbar on the right
            gs = fig.add_gridspec(2, 3, left=0.08, right=0.82, top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)
            axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]
            
            # Define all layer paths for this block
            layer_paths = [
                # Attention layers
                [block_name, 'CausalSelfAttention_0', 'q_proj', 'kernel'],
                [block_name, 'CausalSelfAttention_0', 'k_proj', 'kernel'],
                [block_name, 'CausalSelfAttention_0', 'v_proj', 'kernel'],
                [block_name, 'CausalSelfAttention_0', 'out_proj', 'kernel'],
                # MLP layers
                [block_name, 'MLP_0', 'fc1', 'kernel'],
                [block_name, 'MLP_0', 'fc2', 'kernel']
            ]
            
            # Plot each layer
            sc = None
            for i, layer_path in enumerate(layer_paths):
                row = i // 3
                col = i % 3
                layer_short_name = f"{layer_path[-2]}/{layer_path[-1]}"
                sc_temp = create_order_statistics_plot(data, layer_path, axes[row][col], "")
                if sc_temp is not None:
                    sc = sc_temp
                # Simplify title to just show the layer name
                axes[row][col].set_title(layer_short_name, fontsize=12)
            
            # Add colorbar to the right
            if sc is not None:
                cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
                cbar = fig.colorbar(sc, cax=cbar_ax)
                cbar.set_label('Log Timestamp', fontsize=10)
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
        
        # Additional page for other layers (ln_f, etc.)
        print("Creating additional layers page...")
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle('Additional Model Layers', fontsize=16, fontweight='bold')
        
        # Create axes with space for colorbar on the right
        gs = fig.add_gridspec(2, 2, left=0.1, right=0.82, top=0.92, bottom=0.08, hspace=0.3, wspace=0.3)
        axes = [[fig.add_subplot(gs[i, j]) for j in range(2)] for i in range(2)]
        
        # Check if ln_f exists and plot it
        additional_layers = []
        if 'ln_f' in params:
            ln_f_layers = list(params['ln_f'].keys())
            for layer in ln_f_layers:
                if layer in ['scale', 'bias']:  # Common layer norm parameters
                    additional_layers.append(['ln_f', layer])
        
        # Plot additional layers if they exist
        sc = None
        for i, layer_path in enumerate(additional_layers[:4]):  # Max 4 additional layers
            row = i // 2
            col = i % 2
            sc_temp = create_order_statistics_plot(data, layer_path, axes[row][col])
            if sc_temp is not None:
                sc = sc_temp
        
        # Hide unused subplots
        for i in range(len(additional_layers), 4):
            row = i // 2
            col = i % 2
            axes[row][col].set_visible(False)
        
        # Add colorbar to the right if we have any plots
        if sc is not None:
            cbar_ax = fig.add_axes([0.85, 0.15, 0.03, 0.7])
            cbar = fig.colorbar(sc, cax=cbar_ax)
            cbar.set_label('Log Timestamp', fontsize=10)
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report saved to: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive order statistics PDF report')
    parser.add_argument('pickle_file', help='Path to the pickle file containing tau statistics')
    parser.add_argument('-o', '--output', default='order_statistics_report.pdf', 
                      help='Output PDF file name (default: order_statistics_report.pdf)')
    
    args = parser.parse_args()
    
    # Check if pickle file exists
    if not Path(args.pickle_file).exists():
        print(f"Error: Pickle file '{args.pickle_file}' not found!")
        return 1
    
    try:
        output_path = generate_comprehensive_pdf(args.pickle_file, args.output, True)
        print(f"Successfully generated PDF report: {output_path}")
        return 0
    except Exception as e:
        print(f"Error generating PDF: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    exit(main())