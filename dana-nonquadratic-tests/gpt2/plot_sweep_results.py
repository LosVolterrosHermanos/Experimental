#!/usr/bin/env python
"""
Script to load and visualize sweep results from the g3p parameter study.
Creates a plot showing training and validation losses for different g3p values.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from pathlib import Path

def load_sweep_data(results_dir="results"):
    """Load all pickle files from the results directory and extract sweep data."""
    
    # Find all pickle files containing metrics with steps_10000
    pickle_files = glob.glob(os.path.join(results_dir, "*_metrics_*steps_10000*.pkl"))
    
    if not pickle_files:
        raise ValueError(f"No metrics pickle files found in {results_dir}")
    
    print(f"Found {len(pickle_files)} metrics files")
    
    sweep_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            
            # Get g3_iv value (this appears to be the g3p parameter being swept)
            g3p_value = config.get('dana_g3_p', None)
            
            if g3p_value is not None:
                sweep_data.append({
                    'g3p': g3p_value,
                    'steps': np.array(metrics['step']),
                    'train_loss': np.array(metrics['train_loss']),
                    'val_loss': np.array(metrics['val_loss']),
                    'tokens_processed': np.array(metrics['tokens_processed']),
                    'config': config,
                    'filename': os.path.basename(pkl_file)
                })
                print(f"Loaded data for g3p={g3p_value} from {os.path.basename(pkl_file)}")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not sweep_data:
        raise ValueError("No valid sweep data found")
    
    # Sort by g3p value in reverse order (highest first)
    sweep_data.sort(key=lambda x: x['g3p'], reverse=True)
    
    return sweep_data

def create_sweep_visualization(sweep_data, output_file="g3p_sweep_results.pdf"):
    """Create a visualization of the sweep results similar to the original plotting routine."""
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use plasma colormap for different g3p values
    cmap = plt.cm.plasma
    n_runs = len(sweep_data)
    colors = [cmap((n_runs - 1 - i) / (n_runs - 1)) for i in range(n_runs)]
    legend_elements = []

    
    # Plot each run
    for i, data in enumerate(reversed(sweep_data)):
        g3p = data['g3p']
        steps = data['steps']
        train_loss = data['train_loss']
        val_loss = data['val_loss']
        
        # Calculate tokens (similar to original code)
        config = data['config']
        tokens_per_step = config["batch_size"] * config["seq_len"]
        tokens = steps * tokens_per_step
        
        color = colors[i]
        
        # Plot training loss with alpha for less visibility
        # ax.loglog(tokens, train_loss, 'o-', color=color, alpha=0.1, 
        #          markersize=1, linewidth=1.5, 
        #          label=f'g3p={g3p:.3f} (train)')
        
        # Plot validation loss with same color but full alpha
        ax.loglog(tokens, val_loss, 's-', color=color, alpha=1.0, 
                 markersize=0, linewidth=3,
                 label=f'g3p={g3p:.3f} (val)')
        
        legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, marker='s', 
                                markersize=4, label=f'g3p={g3p:.3f} (val)'))
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens')
    ax.set_ylabel('Loss')
    ax.set_title('NanoGPT Learning Curves: g3p Parameter Sweep\n(Validation: solid lines)')
    # Add subtitle with batch size and sequence length from config
    config = sweep_data[0]['config']  # Get config from first run
    subtitle = f"DANA, g2={config['dana_g2']}, g3iv={config['dana_g3_iv']}, batch_size={config['batch_size']}, seq_len={config['seq_len']}"
    ax.set_title(f'NanoGPT Learning Curves: g3p Parameter Sweep\n(Validation: solid lines)\n{subtitle}')
    
    # Format x-axis similar to original
    def format_tokens(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tokens))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Create a single legend with both training and validation
    # Group by g3p value and show both train/val for each
    # unique_g3ps = sorted(set([data['g3p'] for data in sweep_data]), reverse=True)
    
    # # Create custom legend entries
    # legend_elements = []
    # for i, g3p in enumerate(unique_g3ps):
    #     color = colors[i]
    #     # Add validation entry (solid line)
    #     legend_elements.append(plt.Line2D([0], [0], color=color, lw=2, marker='s', 
    #                                     markersize=4, label=f'g3p={g3p:.3f} (val)'))
    #     # # Add training entry (faded line)
    #     # legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, marker='o', 
    #     #                                 markersize=3, alpha=0.3, label=f'g3p={g3p:.3f} (train)'))
    
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Set y-axis limits
    all_losses = []
    for data in sweep_data:
        all_losses.extend(data['train_loss'])
        all_losses.extend(data['val_loss'])
    
    min_loss = min(all_losses)
    max_loss = max(all_losses)
    ax.set_ylim(min_loss * 0.9, min(max_loss * 1.1, 12))
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sweep visualization saved as {output_file}")
    
    # Print summary statistics
    print("\nSweep Summary:")
    print("g3p Value | Final Train Loss | Final Val Loss")
    print("-" * 45)
    for data in sweep_data:
        g3p = data['g3p']
        final_train = data['train_loss'][-1]
        final_val = data['val_loss'][-1]
        print(f"{g3p:8.3f} | {final_train:14.6f} | {final_val:12.6f}")

def main():
    """Main function to load data and create visualization."""
    try:
        # Load sweep data
        results_dir = "results"
        if not os.path.exists(results_dir):
            print(f"Results directory '{results_dir}' not found!")
            print("Please ensure you have run the sweep and saved results to the 'results' directory.")
            return
        
        sweep_data = load_sweep_data(results_dir)
        
        print(f"\nLoaded sweep data for {len(sweep_data)} different g3p values:")
        for data in sweep_data:
            print(f"  g3p = {data['g3p']:.4f}")
        
        # Create visualization
        create_sweep_visualization(sweep_data)
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()