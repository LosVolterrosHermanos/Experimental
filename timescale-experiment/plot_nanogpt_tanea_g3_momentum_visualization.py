#!/usr/bin/env python
"""
Script to visualize Tanea results with g3 and momentum flavor parameter sweep.
Creates learning curves with plasma color coding for g3 values and line styles for momentum flavors.
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
            
            # Extract momentum_flavor from config
            momentum_flavor = config.get('momentum_flavor', 'mk3')  # Default to mk3 if not specified
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'tau_statistics': tau_stats,
                'num_params': num_params,
                'momentum_flavor': momentum_flavor,
                'filename': os.path.basename(pkl_file)
            })
            print(f"Loaded data from {os.path.basename(pkl_file)}")
            print(f"  Parameters: g3={config['tanea_g3']}, momentum_flavor={momentum_flavor}, g2={config['tanea_g2']}")
            print(f"  Model params: {num_params:,}")
    
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid Tanea results data found")
    
    # Sort by g3 parameter, then by momentum flavor for consistent ordering
    results_data.sort(key=lambda x: (x['config']['tanea_g3'], x['momentum_flavor']))
    
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

def create_g3_momentum_visualization(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create learning curves plot with plasma colors for g3 and line styles for momentum flavors."""
    
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
            # Plot AdamW baseline with thick lines
            ax.loglog(tokens, val_losses, marker=marker, linestyle='-', color=color, alpha=1.0, 
                     markersize=6, linewidth=4, label=label_base)
    
    # Group results by g3 value
    g3_groups = defaultdict(list)
    for data in results_data:
        g3_val = data['config']['tanea_g3']
        g3_groups[g3_val].append(data)
    
    # Define plasma colors for different g3 values
    g3_values = sorted(g3_groups.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(g3_values)))  # Use plasma colormap
    
    # Create g3 to color mapping for legend
    g3_color_map = {g3: colors[i] for i, g3 in enumerate(g3_values)}
    
    # Get unique momentum flavor values and assign line styles
    momentum_flavors = sorted(set(data['momentum_flavor'] for data in results_data))
    
    # Define line styles for momentum flavors
    flavor_styles = {
        'effective-clip': '-',      # solid line
        'mk2': '--',               # dashed line
        'mk3': '-.'                # dash-dot line
    }
    
    # Plot each g3 group with different colors
    for i, (g3_val, g3_data) in enumerate(g3_groups.items()):
        color = colors[i]
        
        # Sort by momentum flavor within each g3 group
        g3_data.sort(key=lambda x: x['momentum_flavor'])
        
        for data in g3_data:
            config = data['config']
            metrics = data['metrics']
            momentum_flavor = data['momentum_flavor']
            
            # Calculate tokens processed
            steps = np.array(metrics['step'])
            train_losses = np.array(metrics['train_loss'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            # Get line style based on momentum flavor
            linestyle = flavor_styles.get(momentum_flavor, '-')
            
            # Create label
            label = f"g3={g3_val:.1e}, {momentum_flavor}".replace('e+0', 'e+').replace('e-0', 'e-')
            
            # Plot validation curves
            ax.loglog(tokens, val_losses, linestyle=linestyle, color=color, alpha=0.8, 
                     markersize=3, linewidth=2, label=label, marker='s')
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    title = 'NanoGPT Learning Curves: Tanea g3 vs Momentum Flavor Parameter Sweep'
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
    
    # Create a second legend for momentum flavor line styles
    from matplotlib.lines import Line2D
    flavor_legend_elements = []
    for flavor in sorted(momentum_flavors):
        if flavor in flavor_styles:
            style = flavor_styles[flavor]
            flavor_legend_elements.append(Line2D([0], [0], color='black', linestyle=style, 
                                                label=f'{flavor}'))
    
    # Add the momentum flavor legend
    flavor_legend = ax.legend(handles=flavor_legend_elements, loc='lower left', 
                             title='Momentum Flavor (line style)', fontsize=9, title_fontsize=10)
    ax.add_artist(flavor_legend)
    
    # Create a third legend for g3 values with colors
    g3_legend_elements = []
    for g3 in sorted(g3_values):
        color = g3_color_map[g3]
        g3_legend_elements.append(Line2D([0], [0], color=color, linestyle='-', linewidth=3,
                                        label=f'g3={g3:.1e}'.replace('e+0', 'e+').replace('e-0', 'e-')))
    
    # Add the g3 color legend
    g3_legend = ax.legend(handles=g3_legend_elements, loc='center left', 
                         title='g3 value (color)', fontsize=9, title_fontsize=10,
                         bbox_to_anchor=(1.02, 0.5))
    ax.add_artist(g3_legend)
    
    # Add text box to explain color coding
    textstr = 'Plasma colormap: darker = smaller g3, brighter = larger g3'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_g3_momentum_curves.pdf")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"G3-momentum visualization saved as {output_file}")
    plt.show()

def create_parameter_summary_table(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create a summary table of all parameter combinations and their performance."""
    
    # Group by g3 and momentum flavor for easy comparison
    summary_data = []
    
    for data in results_data:
        config = data['config']
        metrics = data['metrics']
        momentum_flavor = data['momentum_flavor']
        
        # Get final validation loss
        final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
        
        summary_data.append({
            'g3': config['tanea_g3'],
            'momentum_flavor': momentum_flavor,
            'final_val_loss': final_val_loss,
            'train_steps': config['train_steps'],
            'filename': data['filename']
        })
    
    # Sort by final validation loss (best first)
    summary_data.sort(key=lambda x: x['final_val_loss'])
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_g3_momentum_summary.txt")
    
    # Write summary to file
    with open(output_file, 'w') as f:
        f.write("NanoGPT Tanea g3-momentum Parameter Sweep Summary\n")
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
        f.write("-"*80 + "\n")
        f.write(f"{'Rank':<4} {'g3':<10} {'Momentum':<15} {'Final Val Loss':<15} {'Filename'}\n")
        f.write("-"*80 + "\n")
        
        for i, data in enumerate(summary_data, 1):
            f.write(f"{i:<4} {data['g3']:<10.1e} {data['momentum_flavor']:<15} {data['final_val_loss']:<15.6f} {data['filename']}\n")
    
    print(f"Parameter summary saved as {output_file}")
    
    # Also print top 5 results
    print("\nTop 5 parameter combinations:")
    print(f"{'Rank':<4} {'g3':<10} {'Momentum':<15} {'Final Val Loss':<15}")
    print("-"*50)
    for i, data in enumerate(summary_data[:5], 1):
        print(f"{i:<4} {data['g3']:<10.1e} {data['momentum_flavor']:<15} {data['final_val_loss']:<15.6f}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize NanoGPT Tanea g3-momentum parameter sweep")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results pickle files")
    parser.add_argument("--pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match result files")
    parser.add_argument("--output_prefix", type=str, default="nanogpt_tanea_g3_momentum",
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
        
        # Create g3-momentum visualization
        viz_output = os.path.join(args.results_dir, f"{args.output_prefix}_visualization.pdf")
        create_g3_momentum_visualization(results_data, adamw_baselines, viz_output, args.results_dir)
        
        # Create parameter summary table
        summary_output = os.path.join(args.results_dir, f"{args.output_prefix}_summary.txt")
        create_parameter_summary_table(results_data, adamw_baselines, summary_output, args.results_dir)
        
        # Print statistics
        print(f"\nParameter sweep statistics:")
        g3_values = set(data['config']['tanea_g3'] for data in results_data)
        momentum_flavors = set(data['momentum_flavor'] for data in results_data)
        print(f"  g3 values tested: {sorted(g3_values)}")
        print(f"  momentum flavors tested: {sorted(momentum_flavors)}")
        print(f"  Total combinations: {len(results_data)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()