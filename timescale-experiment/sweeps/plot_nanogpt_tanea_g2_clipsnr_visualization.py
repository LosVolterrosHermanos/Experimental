#!/usr/bin/env python
"""
Script to visualize Tanea results with g2 and clipsnr parameter sweep.
Creates learning curves with color coding for g2 values and line patterns for clipsnr values.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from pathlib import Path
import argparse
from collections import defaultdict

def load_tanea_results(results_dir="results", pattern="*tanea_results*.pkl"):
    """Load Tanea training results from pickle files."""
    
    pickle_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not pickle_files:
        raise ValueError(f"No Tanea results pickle files found in {results_dir} with pattern {pattern}")
    
    print(f"Found {len(pickle_files)} Tanea results files")
    
    results_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            tau_stats = data.get('tau_statistics', {})
            num_params = data.get('num_params', 0)
            
            # Extract clipsnr from config if available
            clipsnr = config.get('clipsnr', 2.0)  # Default to 2.0 if not specified
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'tau_statistics': tau_stats,
                'num_params': num_params,
                'clipsnr': clipsnr,
                'filename': os.path.basename(pkl_file)
            })
            print(f"Loaded data from {os.path.basename(pkl_file)}")
            print(f"  Parameters: g2={config['tanea_g2']}, clipsnr={clipsnr}, g3={config['tanea_g3']}")
            print(f"  Model params: {num_params:,}")
    
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid Tanea results data found")
    
    # Sort by g2 parameter, then by clipsnr for consistent ordering
    results_data.sort(key=lambda x: (x['config']['tanea_g2'], x['clipsnr']))
    
    return results_data

def load_adamw_baselines(results_dir="results", pattern="*adamw_baseline*.pkl"):
    """Load all AdamW baseline results from pickle files."""
    
    pickle_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not pickle_files:
        print(f"No AdamW baseline files found in {results_dir} with pattern {pattern}")
        return []
    
    baseline_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            num_params = data.get('num_params', 0)
            optimizer_type = data.get('optimizer_type', 'adamw')
            
            baseline_info = {
                'config': config,
                'metrics': metrics,
                'num_params': num_params,
                'optimizer_type': optimizer_type,
                'filename': os.path.basename(pkl_file)
            }
            
            baseline_data.append(baseline_info)
            
            print(f"Loaded AdamW baseline from {os.path.basename(pkl_file)}")
            print(f"  Parameters: lr={config['lr']}, beta1={config['beta1']}, beta2={config['beta2']}, wd={config['weight_decay']}")
            print(f"  Model params: {num_params:,}")
            
        except Exception as e:
            print(f"Error loading AdamW baseline {pkl_file}: {e}")
            continue
    
    # Sort by beta1 for consistent ordering
    baseline_data.sort(key=lambda x: x['config']['beta1'])
    
    return baseline_data

def create_g2_clipsnr_visualization(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create learning curves plot with color coding for g2 and line patterns for clipsnr."""
    
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot AdamW baselines first if available
    if adamw_baselines:
        baseline_colors = ['black', 'gray']
        baseline_markers = ['o', 's']
        
        for i, baseline in enumerate(adamw_baselines):
            config = baseline['config']
            metrics = baseline['metrics']
            
            # Calculate tokens processed
            steps = np.array(metrics['step'])
            train_losses = np.array(metrics['train_loss'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            color = baseline_colors[i % len(baseline_colors)]
            marker = baseline_markers[i % len(baseline_markers)]
            
            label_base = f"AdamW β1={config['beta1']:.1f} (lr={config['lr']:.1e}, β2={config['beta2']:.2f})".replace('e+0', 'e+').replace('e-0', 'e-')
            # Plot AdamW baseline without markers
            ax.loglog(tokens, val_losses, linestyle='-', color=color, alpha=1.0, 
                     linewidth=2, label=label_base)
    
    # Group results by g2 value
    g2_groups = defaultdict(list)
    for data in results_data:
        g2_val = data['config']['tanea_g2']
        g2_groups[g2_val].append(data)
    
    # Define colors for different g2 values using plasma colormap (0-0.8 range)
    g2_values = sorted(g2_groups.keys())
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(g2_values)))
    
    # Get unique clipsnr values and assign line styles with better spacing
    clipsnr_values = sorted(set(data['clipsnr'] for data in results_data))
    # Use more distinct line styles and wider spacing
    available_styles = ['-', '--', '-.', ':', (0, (5, 10)), (0, (3, 10, 1, 10)), (0, (1, 10)), (0, (5, 1, 3, 1, 1, 1))]
    
    # Create clipsnr to line style mapping
    clipsnr_styles = {}
    for i, clipsnr in enumerate(clipsnr_values):
        clipsnr_styles[clipsnr] = available_styles[i % len(available_styles)]
    
    # Plot each g2 group with different colors
    for i, (g2_val, g2_data) in enumerate(g2_groups.items()):
        color = colors[i]
        
        # Sort by clipsnr within each g2 group
        g2_data.sort(key=lambda x: x['clipsnr'])
        
        for data in g2_data:
            config = data['config']
            metrics = data['metrics']
            clipsnr = data['clipsnr']
            
            # Calculate tokens processed
            steps = np.array(metrics['step'])
            train_losses = np.array(metrics['train_loss'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            # Get line style based on clipsnr
            linestyle = clipsnr_styles.get(clipsnr, '-')
            
            # Create label
            label = f"g2={g2_val:.1e}, clipsnr={clipsnr:.1e}".replace('e+0', 'e+').replace('e-0', 'e-')
            
            # Plot validation curves without markers
            ax.loglog(tokens, val_losses, linestyle=linestyle, color=color, alpha=0.9, 
                     linewidth=2, label=label)
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    title = 'NanoGPT Learning Curves: Tanea g2 vs clipsnr Parameter Sweep'
    if adamw_baselines:
        if len(adamw_baselines) > 1:
            title += ' vs AdamW Baselines'
        else:
            title += ' vs AdamW Baseline'
    ax.set_title(title, fontsize=16)
    
    # Format x-axis
    def format_tokens(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tokens))
    
    # Add grid and legend
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Create custom legend with two sections
    # First collect all handles and labels
    handles, labels = ax.get_legend_handles_labels()
    
    # Create legend with smaller font and more columns
    legend = ax.legend(handles, labels, fontsize=10, loc='upper right', ncol=2, 
                      bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
    
    # Create a second legend for clipsnr line styles
    from matplotlib.lines import Line2D
    clipsnr_legend_elements = []
    for clipsnr in sorted(clipsnr_values):
        style = clipsnr_styles[clipsnr]
        clipsnr_legend_elements.append(Line2D([0], [0], color='black', linestyle=style, 
                                            label=f'clipsnr={clipsnr:.1e}'))
    
    # Add the clipsnr legend
    clipsnr_legend = ax.legend(handles=clipsnr_legend_elements, loc='lower left', 
                              title='Clipsnr (line style)', fontsize=9, title_fontsize=10)
    ax.add_artist(clipsnr_legend)
    
    # Add colorbar for g2 values
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    
    # Create a colorbar for g2 values using the same [0,0.8] range
    norm = mcolors.Normalize(vmin=min(g2_values), vmax=max(g2_values))
    # Create a custom colormap that uses only [0,0.8] of plasma
    from matplotlib.colors import LinearSegmentedColormap
    plasma_colors = plt.cm.plasma(np.linspace(0, 0.8, 256))
    plasma_custom = LinearSegmentedColormap.from_list('plasma_custom', plasma_colors)
    sm = cm.ScalarMappable(norm=norm, cmap=plasma_custom)
    sm.set_array([])
    
    # Add colorbar to the plot
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, aspect=30, pad=0.02)
    cbar.set_label('g2 value', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_g2_clipsnr_curves.pdf")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"G2-clipsnr visualization saved as {output_file}")
    plt.show()

def create_parameter_summary_table(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create a summary table of all parameter combinations and their performance."""
    
    # Group by g2 and clipsnr for easy comparison
    summary_data = []
    
    for data in results_data:
        config = data['config']
        metrics = data['metrics']
        clipsnr = data['clipsnr']
        
        # Get final validation loss
        final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
        
        summary_data.append({
            'g2': config['tanea_g2'],
            'clipsnr': clipsnr,
            'final_val_loss': final_val_loss,
            'train_steps': config['train_steps'],
            'filename': data['filename']
        })
    
    # Sort by final validation loss (best first)
    summary_data.sort(key=lambda x: x['final_val_loss'])
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_g2_clipsnr_summary.txt")
    
    # Write summary to file
    with open(output_file, 'w') as f:
        f.write("NanoGPT Tanea g2-clipsnr Parameter Sweep Summary\n")
        f.write("="*60 + "\n\n")
        
        if adamw_baselines:
            f.write("AdamW Baselines:\n")
            for i, baseline in enumerate(adamw_baselines):
                config = baseline['config']
                metrics = baseline['metrics']
                final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
                f.write(f"  {i+1}. β1={config['beta1']:.1f}, lr={config['lr']:.1e}, final_val_loss={final_val_loss:.6f}\n")
            f.write("\n")
        
        f.write("Tanea Results (sorted by final validation loss):\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Rank':<4} {'g2':<10} {'clipsnr':<8} {'Final Val Loss':<15} {'Filename'}\n")
        f.write("-"*60 + "\n")
        
        for i, data in enumerate(summary_data, 1):
            f.write(f"{i:<4} {data['g2']:<10.1e} {data['clipsnr']:<8.1f} {data['final_val_loss']:<15.6f} {data['filename']}\n")
    
    print(f"Parameter summary saved as {output_file}")
    
    # Also print top 5 results
    print("\nTop 5 parameter combinations:")
    print(f"{'Rank':<4} {'g2':<10} {'clipsnr':<8} {'Final Val Loss':<15}")
    print("-"*40)
    for i, data in enumerate(summary_data[:5], 1):
        print(f"{i:<4} {data['g2']:<10.1e} {data['clipsnr']:<8.1f} {data['final_val_loss']:<15.6f}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize NanoGPT Tanea g2-clipsnr parameter sweep")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results pickle files")
    parser.add_argument("--pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match result files")
    parser.add_argument("--output_prefix", type=str, default="nanogpt_tanea_g2_clipsnr",
                       help="Prefix for output files")
    parser.add_argument("--adamw_pattern", type=str, default="*adamw_baseline*.pkl",
                       help="Pattern to match AdamW baseline files")
    
    args = parser.parse_args()
    
    try:
        # Load results data
        if not os.path.exists(args.results_dir):
            print(f"Results directory '{args.results_dir}' not found!")
            return
        
        results_data = load_tanea_results(args.results_dir, args.pattern)
        
        print(f"\nLoaded results for {len(results_data)} Tanea configurations")
        
        # Load AdamW baselines
        adamw_baselines = load_adamw_baselines(args.results_dir, args.adamw_pattern)
        if adamw_baselines:
            print(f"\n{len(adamw_baselines)} AdamW baseline(s) loaded successfully")
        else:
            print("\nNo AdamW baselines found")
        
        # Create g2-clipsnr visualization
        viz_output = os.path.join(args.results_dir, f"{args.output_prefix}_visualization.pdf")
        create_g2_clipsnr_visualization(results_data, adamw_baselines, viz_output, args.results_dir)
        
        # Create parameter summary table
        summary_output = os.path.join(args.results_dir, f"{args.output_prefix}_summary.txt")
        create_parameter_summary_table(results_data, adamw_baselines, summary_output, args.results_dir)
        
        # Print statistics
        print(f"\nParameter sweep statistics:")
        g2_values = set(data['config']['tanea_g2'] for data in results_data)
        clipsnr_values = set(data['clipsnr'] for data in results_data)
        print(f"  g2 values tested: {sorted(g2_values)}")
        print(f"  clipsnr values tested: {sorted(clipsnr_values)}")
        print(f"  Total combinations: {len(results_data)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()