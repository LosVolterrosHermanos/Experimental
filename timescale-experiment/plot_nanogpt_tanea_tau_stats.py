#!/usr/bin/env python
"""
Script to load and visualize Tanea tau statistics from NanoGPT training results.
Creates tau order statistics plots similar to the MoE experiment.
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from pathlib import Path
import argparse

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
            
            if tau_stats and len(tau_stats.get('timestamps', [])) > 0:
                results_data.append({
                    'config': config,
                    'metrics': metrics,
                    'tau_statistics': tau_stats,
                    'num_params': num_params,
                    'filename': os.path.basename(pkl_file)
                })
                print(f"Loaded data from {os.path.basename(pkl_file)}")
                print(f"  Parameters: g2={config['tanea_g2']}, g3={config['tanea_g3']}, delta={config['tanea_delta']}")
                print(f"  Model params: {num_params:,}")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid Tanea results data found")
    
    # Sort by g2 parameter for consistent ordering
    results_data.sort(key=lambda x: x['config']['tanea_g2'])
    
    return results_data

def load_adamw_baseline(results_dir="results", pattern="*adamw_baseline*.pkl"):
    """Load the most recent AdamW baseline results from pickle files."""
    
    pickle_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not pickle_files:
        print(f"No AdamW baseline files found in {results_dir} with pattern {pattern}")
        return None
    
    # Sort by modification time and take the most recent
    pickle_files.sort(key=os.path.getmtime, reverse=True)
    most_recent_file = pickle_files[0]
    
    try:
        with open(most_recent_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract relevant information
        config = data['config']
        metrics = data['metrics']
        num_params = data.get('num_params', 0)
        optimizer_type = data.get('optimizer_type', 'adamw')
        
        baseline_data = {
            'config': config,
            'metrics': metrics,
            'num_params': num_params,
            'optimizer_type': optimizer_type,
            'filename': os.path.basename(most_recent_file)
        }
        
        print(f"Loaded AdamW baseline from {os.path.basename(most_recent_file)}")
        print(f"  Parameters: lr={config['lr']}, beta1={config['beta1']}, beta2={config['beta2']}, wd={config['weight_decay']}")
        print(f"  Model params: {num_params:,}")
        
        return baseline_data
        
    except Exception as e:
        print(f"Error loading AdamW baseline {most_recent_file}: {e}")
        return None

def create_tau_statistics_plots(results_data, output_file="nanogpt_tanea_tau_stats.pdf"):
    """Create tau order statistics visualization similar to MoE experiment."""
    
    n_results = len(results_data)
    
    # Set up the figure - one subplot per result
    fig, axes = plt.subplots(1, n_results, figsize=(6 * n_results, 5))
    if n_results == 1:
        axes = [axes]
    
    for i, data in enumerate(results_data):
        config = data['config']
        tau_stats = data['tau_statistics']
        
        if len(tau_stats['timestamps']) > 0:
            ax = axes[i]
            
            tau_times = tau_stats['timestamps']
            tau_order_stats = tau_stats['tau_order_statistics']
            
            # Check if reversed order statistics are available
            tau_reversed_order_stats = tau_stats.get('tau_reversed_order_statistics', None)
            
            # Create color map for time evolution
            n_timestamps = len(tau_times)
            colors = plt.cm.plasma(np.linspace(0, 0.8, n_timestamps))
            
            # Find overall max and min for y-axis range (include both regular and reversed stats)
            all_order_stats = []
            for order_stats in tau_order_stats:
                if len(order_stats) > 0:
                    all_order_stats.extend(order_stats)
            
            # Also include reversed order statistics if available
            if tau_reversed_order_stats:
                for order_stats in tau_reversed_order_stats:
                    if len(order_stats) > 0:
                        all_order_stats.extend(order_stats)
            
            if all_order_stats:
                max_tau = max(all_order_stats)
                min_tau_plot = max_tau * 1e-5  # 10 orders of magnitude lower
                
                # Plot largest order statistics for each timestamp
                for t_idx, (timestamp, order_stats) in enumerate(zip(tau_times, tau_order_stats)):
                    if len(order_stats) > 0:
                        # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                        k_values = np.arange(len(order_stats))
                        
                        # Filter order stats to only show those within our range
                        valid_mask = order_stats >= min_tau_plot
                        if np.any(valid_mask):
                            filtered_k = 1.1**(k_values[valid_mask])
                            filtered_stats = order_stats[valid_mask]
                            
                            ax.scatter(filtered_k, filtered_stats, 
                                     color=colors[t_idx], alpha=0.7, s=5)
                
                # Plot smallest order statistics if available (same color scheme)
                if tau_reversed_order_stats:
                    for t_idx, (timestamp, reversed_order_stats) in enumerate(zip(tau_times, tau_reversed_order_stats)):
                        if len(reversed_order_stats) > 0:
                            # k values: 0, 1, 2, ..., max_k where (1.1)^k corresponds to order stat index
                            k_values = np.arange(len(reversed_order_stats))
                            
                            # Filter order stats to only show those within our range
                            valid_mask = reversed_order_stats >= min_tau_plot
                            if np.any(valid_mask):
                                filtered_k = 1.1**(k_values[valid_mask])
                                filtered_stats = reversed_order_stats[valid_mask]
                                
                                ax.scatter(filtered_k, filtered_stats, 
                                         color=colors[t_idx], alpha=0.7, s=5)

                        
                
                # Set y-axis limits
                ax.set_ylim(min_tau_plot, max_tau * 1.1)
            
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('k (order statistic index)')
            ax.set_ylabel('τ_k')
            
            # Create title with model and optimizer parameters
            title = (f'NanoGPT Tanea Tau Statistics\n'
                    f'Params: {data["num_params"]:,}, '
                    f'g2={config["tanea_g2"]}, kappa={config["tanea_kappa"]}, tanea_g3={config["tanea_g3"]}\n'
                    f'wd={config["weight_decay"]}, power_wd={config["power_weight_decay"]}, power_delta={config["weight_decay_ts"]*config["weight_decay"]}')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Add colorbar to show time evolution
            sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=0, vmax=0.8))
            sm.set_array([])
            
            # Map timestamp indices to [0, 0.8] range
            if n_timestamps > 1:
                actual_times = np.array(tau_times)
                cbar = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.04)
                
                # Set colorbar ticks to show actual iteration numbers
                tick_positions = np.linspace(0, 0.8, min(5, n_timestamps))
                tick_labels = []
                for pos in tick_positions:
                    # Find closest timestamp index
                    idx = int(pos / 0.8 * (n_timestamps - 1))
                    tick_labels.append(f'{actual_times[idx]:.0f}')
                
                cbar.set_ticks(tick_positions)
                cbar.set_ticklabels(tick_labels)
                cbar.set_label('Training Step')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Tau statistics plot saved as {output_file}")
    
    plt.show()

def create_learning_curves(results_data, adamw_baseline=None, output_file="nanogpt_tanea_learning_curves.pdf"):
    """Create learning curves plot showing training and validation losses."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot AdamW baseline first if available
    if adamw_baseline:
        config = adamw_baseline['config']
        metrics = adamw_baseline['metrics']
        
        # Calculate tokens processed
        steps = np.array(metrics['step'])
        train_losses = np.array(metrics['train_loss'])
        val_losses = np.array(metrics['val_loss'])
        tokens_per_step = config["batch_size"] * config["seq_len"]
        tokens = steps * tokens_per_step
        
        # Plot AdamW baseline in black with thick lines
        ax.loglog(tokens, train_losses, 'o-', color='black', alpha=0.8, 
                 markersize=5, linewidth=3, label='AdamW Baseline (train)')
        ax.loglog(tokens, val_losses, 's-', color='black', alpha=1.0, 
                 markersize=5, linewidth=3, label='AdamW Baseline (val)')
    
    # Use different colors for different Tanea configurations
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_data)))
    
    for i, data in enumerate(results_data):
        config = data['config']
        metrics = data['metrics']
        
        # Calculate tokens processed
        steps = np.array(metrics['step'])
        train_losses = np.array(metrics['train_loss'])
        val_losses = np.array(metrics['val_loss'])
        tokens_per_step = config["batch_size"] * config["seq_len"]
        tokens = steps * tokens_per_step
        
        color = colors[i]
        label_base = f"Tanea g2={config['tanea_g2']}, κ={config['tanea_kappa']}, g3={config['tanea_g3']}"
        if 'weight_decay' in config:
            label_base += f", wd={config['weight_decay']}"
            if config['power_weight_decay'] == 1.0:
                label_base += f", power_δ={config['weight_decay_ts']*config['weight_decay']}"
        
        # Plot training and validation curves
        ax.loglog(tokens, train_losses, 'o-', color=color, alpha=0.7, 
                 markersize=4, linewidth=2, label=f'{label_base} (train)')
        ax.loglog(tokens, val_losses, 's-', color=color, alpha=1.0, 
                 markersize=4, linewidth=2, label=f'{label_base} (val)')
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens')
    ax.set_ylabel('Loss')
    title = 'NanoGPT Learning Curves: Tanea vs AdamW Baseline'
    if not adamw_baseline:
        title = 'NanoGPT Tanea Learning Curves'
    ax.set_title(title)
    
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
    ax.grid(True, which='both', linestyle='--', alpha=0.7)
    ax.legend(fontsize=9, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Learning curves plot saved as {output_file}")
    plt.show()

def print_summary_statistics(results_data):
    """Print summary statistics for all results."""
    
    print("\n" + "="*80)
    print("NanoGPT Tanea Training Summary")
    print("="*80)
    
    for data in results_data:
        config = data['config']
        metrics = data['metrics']
        tau_stats = data['tau_statistics']
        
        print(f"\nConfiguration: g2={config['tanea_g2']}, g3={config['tanea_g3']}, δ={config['tanea_delta']}")
        print(f"Model parameters: {data['num_params']:,}")
        print(f"Training steps: {config['train_steps']:,}")
        print(f"Batch size: {config['batch_size']}, Sequence length: {config['seq_len']}")
        
        if metrics['train_loss']:
            print(f"Final train loss: {metrics['train_loss'][-1]:.6f}")
        if metrics['val_loss']:
            print(f"Final val loss: {metrics['val_loss'][-1]:.6f}")
        
        if len(tau_stats['timestamps']) > 0:
            final_tau_mean = tau_stats['tau_mean'][-1]
            final_tau_std = tau_stats['tau_std'][-1]
            final_tau_max = tau_stats['tau_max'][-1]
            
            print(f"Final tau statistics:")
            print(f"  Mean: {final_tau_mean:.6f}")
            print(f"  Std: {final_tau_std:.6f}")
            print(f"  Max: {final_tau_max:.6f}")
            
            if len(tau_stats['tau_order_statistics']) > 0:
                final_order_stats = tau_stats['tau_order_statistics'][-1]
                if len(final_order_stats) > 0:
                    print(f"  Largest order statistics (first 5): {final_order_stats[:5]}")
                
                # Also show smallest order statistics if available
                if 'tau_reversed_order_statistics' in tau_stats and len(tau_stats['tau_reversed_order_statistics']) > 0:
                    final_reversed_order_stats = tau_stats['tau_reversed_order_statistics'][-1]
                    if len(final_reversed_order_stats) > 0:
                        print(f"  Smallest order statistics (first 5): {final_reversed_order_stats[:5]}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Plot NanoGPT Tanea tau statistics")
    parser.add_argument("--results_dir", type=str, default="results",
                       help="Directory containing results pickle files")
    parser.add_argument("--pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match result files")
    parser.add_argument("--output_prefix", type=str, default="nanogpt_tanea",
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
        
        # Load AdamW baseline
        adamw_baseline = load_adamw_baseline(args.results_dir, args.adamw_pattern)
        if adamw_baseline:
            print("\nAdamW baseline loaded successfully")
        else:
            print("\nNo AdamW baseline found - will plot Tanea results only")
        
        # Create tau statistics plots
        tau_output = f"{args.output_prefix}_tau_stats.pdf"
        create_tau_statistics_plots(results_data, tau_output)
        
        # Create learning curves with AdamW baseline
        curves_output = f"{args.output_prefix}_learning_curves.pdf"
        create_learning_curves(results_data, adamw_baseline, curves_output)
        
        # Print summary
        print_summary_statistics(results_data)
        
        # Print AdamW baseline summary if available
        if adamw_baseline:
            print("\n" + "="*80)
            print("AdamW Baseline Summary")
            print("="*80)
            config = adamw_baseline['config']
            metrics = adamw_baseline['metrics']
            print(f"Configuration: lr={config['lr']}, β1={config['beta1']}, β2={config['beta2']}, wd={config['weight_decay']}")
            print(f"Model parameters: {adamw_baseline['num_params']:,}")
            print(f"Training steps: {config['train_steps']:,}")
            print(f"Batch size: {config['batch_size']}, Sequence length: {config['seq_len']}")
            if metrics['train_loss']:
                print(f"Final train loss: {metrics['train_loss'][-1]:.6f}")
            if metrics['val_loss']:
                print(f"Final val loss: {metrics['val_loss'][-1]:.6f}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()