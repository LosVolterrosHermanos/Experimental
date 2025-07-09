#!/usr/bin/env python
"""
Script to visualize Tanea results with kappa parameter sweep from multi-GPU training.
Creates learning curves with plasma color coding for kappa values and line styles for momentum flavors.
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

def load_tanea_results(results_dir="results", pattern="*tanea_results*multi_gpu*.pkl"):
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
            multi_gpu = data.get('multi_gpu', False)
            num_devices = data.get('num_devices', 1)
            
            # Extract momentum_flavor from config
            momentum_flavor = config.get('momentum_flavor', 'mk3')  # Default to mk3 if not specified
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'tau_statistics': tau_stats,
                'num_params': num_params,
                'momentum_flavor': momentum_flavor,
                'multi_gpu': multi_gpu,
                'num_devices': num_devices,
                'filename': os.path.basename(pkl_file)
            })
            print(f"Loaded data from {os.path.basename(pkl_file)}")
            print(f"  Parameters: kappa={config['tanea_kappa']}, momentum_flavor={momentum_flavor}, g2={config['tanea_g2']}, g3={config['tanea_g3']}")
            print(f"  Model params: {num_params:,}, Multi-GPU: {multi_gpu} ({num_devices} devices)")
    
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid Tanea results data found")
    
    # Sort by kappa parameter, then by momentum flavor for consistent ordering
    results_data.sort(key=lambda x: (x['config']['tanea_kappa'], x['momentum_flavor']))
    
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

def create_kappa_momentum_visualization(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create learning curves plot with plasma colors for kappa and line styles for momentum flavors."""
    
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
    
    # Group results by kappa value
    kappa_groups = defaultdict(list)
    for data in results_data:
        kappa_val = data['config']['tanea_kappa']
        kappa_groups[kappa_val].append(data)
    
    # Define plasma colors for different kappa values
    kappa_values = sorted(kappa_groups.keys())
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(kappa_values)))  # Use plasma colormap
    
    # Create kappa to color mapping for legend
    kappa_color_map = {kappa: colors[i] for i, kappa in enumerate(kappa_values)}
    
    # Get unique momentum flavor values and assign line styles
    momentum_flavors = sorted(set(data['momentum_flavor'] for data in results_data))
    
    # Define line styles for momentum flavors
    flavor_styles = {
        'effective-clip': '-',      # solid line
        'mk2': '--',               # dashed line
        'mk3': '-.'                # dash-dot line
    }
    
    # Plot each kappa group with different colors
    for i, (kappa_val, kappa_data) in enumerate(kappa_groups.items()):
        color = colors[i]
        
        # Sort by momentum flavor within each kappa group
        kappa_data.sort(key=lambda x: x['momentum_flavor'])
        
        for data in kappa_data:
            config = data['config']
            metrics = data['metrics']
            momentum_flavor = data['momentum_flavor']
            num_devices = data.get('num_devices', 1)
            
            # Calculate tokens processed
            steps = np.array(metrics['step'])
            train_losses = np.array(metrics['train_loss'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            # Get line style based on momentum flavor
            linestyle = flavor_styles.get(momentum_flavor, '-')
            
            # Create label with multi-GPU info
            label = f"κ={kappa_val:.2f}, {momentum_flavor}".replace('e+0', 'e+').replace('e-0', 'e-')
            if num_devices > 1:
                label += f" ({num_devices}GPU)"
            
            # Plot validation curves
            ax.loglog(tokens, val_losses, linestyle=linestyle, color=color, alpha=0.8, 
                     markersize=3, linewidth=2, label=label, marker='s')
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    title = 'NanoGPT Learning Curves: Tanea κ (kappa) vs Momentum Flavor Parameter Sweep'
    if adamw_baselines:
        if len(adamw_baselines) > 1:
            title += ' vs AdamW Baselines'
        else:
            title += ' vs AdamW Baseline'
    
    # Add multi-GPU info to title if applicable
    if any(data.get('multi_gpu', False) for data in results_data):
        max_devices = max(data.get('num_devices', 1) for data in results_data)
        title += f' (Multi-GPU: {max_devices} devices)'
    
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
    
    # Create colorbar for kappa values
    import matplotlib.colors as mcolors
    
    # Create a colormap normalization based on linear scale of kappa values
    kappa_min = min(kappa_values)
    kappa_max = max(kappa_values)
    norm = mcolors.Normalize(vmin=kappa_min, vmax=kappa_max)
    
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=norm)
    sm.set_array([])
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.15)
    cbar.set_label('κ (kappa) value', rotation=270, labelpad=20, fontsize=12)
    
    # Set colorbar ticks to actual kappa values
    cbar.set_ticks(kappa_values)
    cbar.set_ticklabels([f'{kappa:.2f}' for kappa in kappa_values])
    
    # Add text box to explain the setup
    multi_gpu_info = ""
    if any(data.get('multi_gpu', False) for data in results_data):
        max_devices = max(data.get('num_devices', 1) for data in results_data)
        multi_gpu_info = f", Multi-GPU ({max_devices} devices)"
    
    textstr = f'Line style = momentum flavor{multi_gpu_info}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_kappa_multi_gpu_curves.pdf")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Kappa-momentum visualization saved as {output_file}")
    plt.show()

def create_parameter_summary_table(results_data, adamw_baselines=None, output_file=None, results_dir="results"):
    """Create a summary table of all parameter combinations and their performance."""
    
    # Group by kappa and momentum flavor for easy comparison
    summary_data = []
    
    for data in results_data:
        config = data['config']
        metrics = data['metrics']
        momentum_flavor = data['momentum_flavor']
        num_devices = data.get('num_devices', 1)
        
        # Get final validation loss
        final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
        
        # Calculate effective throughput (tokens/s accounting for multi-GPU)
        if metrics['time_elapsed'] and metrics['tokens_processed']:
            total_time = metrics['time_elapsed'][-1]
            total_tokens = metrics['tokens_processed'][-1]
            throughput = total_tokens / total_time if total_time > 0 else 0
        else:
            throughput = 0
        
        summary_data.append({
            'kappa': config['tanea_kappa'],
            'momentum_flavor': momentum_flavor,
            'final_val_loss': final_val_loss,
            'throughput': throughput,
            'num_devices': num_devices,
            'train_steps': config['train_steps'],
            'filename': data['filename']
        })
    
    # Sort by final validation loss (best first)
    summary_data.sort(key=lambda x: x['final_val_loss'])
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_kappa_multi_gpu_summary.txt")
    
    # Write summary to file
    with open(output_file, 'w') as f:
        f.write("NanoGPT Tanea κ (Kappa) Parameter Sweep Summary - Multi-GPU Training\n")
        f.write("="*80 + "\n\n")
        
        if adamw_baselines:
            f.write("AdamW Baselines:\n")
            for i, baseline in enumerate(adamw_baselines):
                config = baseline['config']
                metrics = baseline['metrics']
                final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
                f.write(f"  {i+1}. β1={config['beta1']:.1f}, lr={config['lr']:.1e}, final_val_loss={final_val_loss:.6f}\n")
            f.write("\n")
        
        f.write("Tanea Results (sorted by final validation loss):\n")
        f.write("-"*100 + "\n")
        f.write(f"{'Rank':<4} {'κ':<6} {'Momentum':<15} {'Final Val Loss':<15} {'Throughput':<12} {'GPUs':<5} {'Filename'}\n")
        f.write("-"*100 + "\n")
        
        for i, data in enumerate(summary_data, 1):
            f.write(f"{i:<4} {data['kappa']:<6.2f} {data['momentum_flavor']:<15} {data['final_val_loss']:<15.6f} "
                   f"{data['throughput']:<12.1f} {data['num_devices']:<5} {data['filename']}\n")
        
        # Add performance analysis
        f.write("\n")
        f.write("Performance Analysis:\n")
        f.write("-"*50 + "\n")
        
        # Best performance by momentum flavor
        flavor_best = {}
        for data in summary_data:
            flavor = data['momentum_flavor']
            if flavor not in flavor_best or data['final_val_loss'] < flavor_best[flavor]['final_val_loss']:
                flavor_best[flavor] = data
        
        f.write("Best performance by momentum flavor:\n")
        for flavor in sorted(flavor_best.keys()):
            data = flavor_best[flavor]
            f.write(f"  {flavor}: κ={data['kappa']:.2f}, val_loss={data['final_val_loss']:.6f}\n")
        
        # Multi-GPU efficiency
        multi_gpu_data = [d for d in summary_data if d['num_devices'] > 1]
        if multi_gpu_data:
            avg_throughput = np.mean([d['throughput'] for d in multi_gpu_data])
            f.write(f"\nMulti-GPU average throughput: {avg_throughput:.1f} tokens/s\n")
            max_devices = max(d['num_devices'] for d in multi_gpu_data)
            f.write(f"Maximum devices used: {max_devices}\n")
    
    print(f"Parameter summary saved as {output_file}")
    
    # Also print top 5 results
    print("\nTop 5 parameter combinations:")
    print(f"{'Rank':<4} {'κ':<6} {'Momentum':<15} {'Final Val Loss':<15} {'GPUs':<5}")
    print("-"*60)
    for i, data in enumerate(summary_data[:5], 1):
        print(f"{i:<4} {data['kappa']:<6.2f} {data['momentum_flavor']:<15} {data['final_val_loss']:<15.6f} {data['num_devices']:<5}")

def create_kappa_heatmap(results_data, output_file=None, results_dir="results"):
    """Create a heatmap showing kappa vs momentum flavor performance."""
    
    # Extract unique kappa values and momentum flavors
    kappa_values = sorted(set(data['config']['tanea_kappa'] for data in results_data))
    momentum_flavors = sorted(set(data['momentum_flavor'] for data in results_data))
    
    # Create performance matrix
    performance_matrix = np.full((len(momentum_flavors), len(kappa_values)), np.nan)
    
    for data in results_data:
        kappa = data['config']['tanea_kappa']
        flavor = data['momentum_flavor']
        final_loss = data['metrics']['val_loss'][-1] if data['metrics']['val_loss'] else np.nan
        
        kappa_idx = kappa_values.index(kappa)
        flavor_idx = momentum_flavors.index(flavor)
        performance_matrix[flavor_idx, kappa_idx] = final_loss
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(performance_matrix, cmap='viridis_r', aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(kappa_values)))
    ax.set_xticklabels([f'{k:.2f}' for k in kappa_values])
    ax.set_yticks(range(len(momentum_flavors)))
    ax.set_yticklabels(momentum_flavors)
    
    # Add text annotations
    for i in range(len(momentum_flavors)):
        for j in range(len(kappa_values)):
            if not np.isnan(performance_matrix[i, j]):
                text = ax.text(j, i, f'{performance_matrix[i, j]:.4f}',
                             ha="center", va="center", color="white", fontsize=10)
    
    ax.set_xlabel('κ (kappa) value', fontsize=14)
    ax.set_ylabel('Momentum Flavor', fontsize=14)
    ax.set_title('Tanea Performance Heatmap: Final Validation Loss vs κ and Momentum Flavor\n(Multi-GPU Training)', fontsize=16)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Final Validation Loss', rotation=270, labelpad=20, fontsize=12)
    
    # Set default output file if not provided
    if output_file is None:
        output_file = os.path.join(results_dir, "nanogpt_tanea_kappa_multi_gpu_heatmap.pdf")
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Kappa heatmap saved as {output_file}")
    plt.show()

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize NanoGPT Tanea kappa parameter sweep from multi-GPU training")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results pickle files")
    parser.add_argument("--pattern", type=str, default="*tanea_results*multi_gpu*.pkl",
                       help="Pattern to match result files")
    parser.add_argument("--output_prefix", type=str, default="nanogpt_tanea_kappa_multi_gpu",
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
        
        # Create kappa-momentum visualization
        viz_output = os.path.join(args.results_dir, f"{args.output_prefix}_visualization.pdf")
        create_kappa_momentum_visualization(results_data, adamw_baselines, viz_output, args.results_dir)
        
        # Create parameter summary table
        summary_output = os.path.join(args.results_dir, f"{args.output_prefix}_summary.txt")
        create_parameter_summary_table(results_data, adamw_baselines, summary_output, args.results_dir)
        
        # Create kappa heatmap
        heatmap_output = os.path.join(args.results_dir, f"{args.output_prefix}_heatmap.pdf")
        create_kappa_heatmap(results_data, heatmap_output, args.results_dir)
        
        # Print statistics
        print(f"\nParameter sweep statistics:")
        kappa_values = set(data['config']['tanea_kappa'] for data in results_data)
        momentum_flavors = set(data['momentum_flavor'] for data in results_data)
        multi_gpu_runs = [d for d in results_data if d.get('multi_gpu', False)]
        
        print(f"  κ values tested: {sorted(kappa_values)}")
        print(f"  momentum flavors tested: {sorted(momentum_flavors)}")
        print(f"  Total combinations: {len(results_data)}")
        print(f"  Multi-GPU runs: {len(multi_gpu_runs)}")
        if multi_gpu_runs:
            devices_used = set(d.get('num_devices', 1) for d in multi_gpu_runs)
            print(f"  GPU configurations: {sorted(devices_used)} devices")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()