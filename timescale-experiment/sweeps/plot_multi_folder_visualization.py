#!/usr/bin/env python
"""
Script to visualize results from multiple folders with enhanced legend and styling.
Accepts multiple folders, places legend below plot with G2, G3, Weight decay info,
uses line styles for momentum flavors, and special handling for Adam vs other optimizers.
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

def load_results_from_folders(folders, tanea_pattern="*tanea_results*.pkl", adamw_pattern="*adamw_baseline*.pkl"):
    """Load results from multiple folders."""
    
    all_tanea_results = []
    all_adamw_results = []
    
    for folder in folders:
        if not os.path.exists(folder):
            print(f"Warning: Folder '{folder}' not found, skipping")
            continue
            
        print(f"\nSearching in folder: {folder}")
        
        # Load Tanea results
        tanea_files = glob.glob(os.path.join(folder, tanea_pattern))
        for pkl_file in tanea_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                config = data['config']
                metrics = data['metrics']
                tau_stats = data.get('tau_statistics', {})
                num_params = data.get('num_params', 0)
                momentum_flavor = config.get('momentum_flavor', 'mk3')
                
                all_tanea_results.append({
                    'config': config,
                    'metrics': metrics,
                    'tau_statistics': tau_stats,
                    'num_params': num_params,
                    'momentum_flavor': momentum_flavor,
                    'filename': os.path.basename(pkl_file),
                    'folder': folder
                })
                print(f"  Loaded Tanea: {os.path.basename(pkl_file)}")
                
            except Exception as e:
                print(f"  Error loading {pkl_file}: {e}")
                continue
        
        # Load AdamW results
        adamw_files = glob.glob(os.path.join(folder, adamw_pattern))
        for pkl_file in adamw_files:
            try:
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                
                config = data['config']
                metrics = data['metrics']
                num_params = data.get('num_params', 0)
                optimizer_type = data.get('optimizer_type', 'adamw')
                
                all_adamw_results.append({
                    'config': config,
                    'metrics': metrics,
                    'num_params': num_params,
                    'optimizer_type': optimizer_type,
                    'filename': os.path.basename(pkl_file),
                    'folder': folder
                })
                print(f"  Loaded AdamW: {os.path.basename(pkl_file)}")
                
            except Exception as e:
                print(f"  Error loading {pkl_file}: {e}")
                continue
    
    print(f"\nTotal loaded: {len(all_tanea_results)} Tanea results, {len(all_adamw_results)} AdamW results")
    
    # Sort results
    all_tanea_results.sort(key=lambda x: (x['config'].get('tanea_g3', 0), x['momentum_flavor']))
    all_adamw_results.sort(key=lambda x: x['config'].get('lr', 0))
    
    return all_tanea_results, all_adamw_results

def create_enhanced_visualization(tanea_results, adamw_results=None, output_file=None):
    """Create visualization with enhanced legend and styling."""
    
    fig, ax = plt.subplots(figsize=(25, 12))
    
    # Get unique momentum flavors for line styles
    momentum_flavors = sorted(set(data['momentum_flavor'] for data in tanea_results))
    
    # Define line styles for momentum flavors
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]
    flavor_styles = {flavor: line_styles[i % len(line_styles)] for i, flavor in enumerate(momentum_flavors)}
    
    # Plot AdamW baselines first with black color
    adamw_handles = []
    if adamw_results:
        for i, baseline in enumerate(adamw_results):
            config = baseline['config']
            metrics = baseline['metrics']
            
            steps = np.array(metrics['step'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            lr = config.get('lr', 0)
            wd = config.get('weight_decay', 0)
            label = f"Adam LR={lr:.1e} WD={wd:.1e}"
            
            line = ax.loglog(tokens, val_losses, linestyle='-', color='black', alpha=0.8, 
                           linewidth=2, label=label)
            adamw_handles.extend(line)
    
    # Group Tanea results by parameters for consistent coloring
    param_groups = defaultdict(list)
    for data in tanea_results:
        config = data['config']
        g2 = config.get('tanea_g2', 0)
        g3 = config.get('tanea_g3', 0)
        wd = config.get('weight_decay', 0)
        key = (g2, g3, wd)
        param_groups[key].append(data)
    
    # Use tab20 colormap for non-Adam optimizers
    param_keys = sorted(param_groups.keys())
    colors = plt.cm.tab20(np.linspace(0, 1, len(param_keys)))
    
    tanea_handles = []
    for i, (param_key, group_data) in enumerate(zip(param_keys, [param_groups[key] for key in param_keys])):
        g2, g3, wd = param_key
        color = colors[i]
        
        # Sort by momentum flavor within each parameter group
        group_data.sort(key=lambda x: x['momentum_flavor'])
        
        for data in group_data:
            config = data['config']
            metrics = data['metrics']
            momentum_flavor = data['momentum_flavor']
            
            steps = np.array(metrics['step'])
            val_losses = np.array(metrics['val_loss'])
            tokens_per_step = config["batch_size"] * config["seq_len"]
            tokens = steps * tokens_per_step
            
            linestyle = flavor_styles.get(momentum_flavor, '-')
            label = f"G2={g2:.1e} G3={g3:.1e} WD={wd:.1e} {momentum_flavor}"
            
            line = ax.loglog(tokens, val_losses, linestyle=linestyle, color=color, alpha=0.8,
                           linewidth=2, label=label)
            tanea_handles.extend(line)
    
    # Set axis labels and title
    ax.set_xlabel('Training Tokens', fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    ax.set_title('Multi-Folder Training Results Comparison', fontsize=16)
    
    # Format x-axis
    def format_tokens(x, pos):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_tokens))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Get all handles and labels for the main legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a separate legend for line styles (momentum flavors) first
    from matplotlib.lines import Line2D
    flavor_legend_elements = []
    for flavor in sorted(momentum_flavors):
        if flavor in flavor_styles:
            style = flavor_styles[flavor]
            flavor_legend_elements.append(Line2D([0], [0], color='gray', linestyle=style, 
                                                label=f'{flavor}'))
    
    if flavor_legend_elements:
        flavor_legend = ax.legend(handles=flavor_legend_elements, loc='upper right', 
                                 title='Momentum Flavors (line style)', fontsize=9, title_fontsize=10)
        ax.add_artist(flavor_legend)
    
    # Create main legend below the plot
    main_legend = ax.legend(handles, labels, fontsize=15, loc='upper center', 
                           bbox_to_anchor=(0.5, -0.1), ncol=3, framealpha=0.9,
                           fancybox=True, shadow=True)
    
    # Adjust layout to accommodate legend below
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)
    
    # Save the plot
    if output_file is None:
        output_file = "multi_folder_visualization.pdf"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_file}")
    plt.show()

def create_summary_table(tanea_results, adamw_results=None, output_file=None):
    """Create a summary table of all results."""
    
    if output_file is None:
        output_file = "multi_folder_summary.txt"
    
    # Collect all results with performance metrics
    all_results = []
    
    # Process AdamW results
    if adamw_results:
        for data in adamw_results:
            config = data['config']
            metrics = data['metrics']
            final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
            
            all_results.append({
                'optimizer': 'Adam',
                'g2': None,
                'g3': None,
                'momentum_flavor': None,
                'lr': config.get('lr', 0),
                'weight_decay': config.get('weight_decay', 0),
                'final_val_loss': final_val_loss,
                'filename': data['filename'],
                'folder': data['folder']
            })
    
    # Process Tanea results
    for data in tanea_results:
        config = data['config']
        metrics = data['metrics']
        final_val_loss = metrics['val_loss'][-1] if metrics['val_loss'] else float('inf')
        
        all_results.append({
            'optimizer': 'Tanea',
            'g2': config.get('tanea_g2', 0),
            'g3': config.get('tanea_g3', 0),
            'momentum_flavor': data['momentum_flavor'],
            'lr': config.get('lr', 0),
            'weight_decay': config.get('weight_decay', 0),
            'final_val_loss': final_val_loss,
            'filename': data['filename'],
            'folder': data['folder']
        })
    
    # Sort by final validation loss
    all_results.sort(key=lambda x: x['final_val_loss'])
    
    # Write summary
    with open(output_file, 'w') as f:
        f.write("Multi-Folder Results Summary\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<4} {'Optimizer':<8} {'G2':<10} {'G3':<10} {'Momentum':<12} {'LR':<10} {'WD':<10} {'Val Loss':<10} {'Folder':<20} {'File'}\n")
        f.write("-"*120 + "\n")
        
        for i, result in enumerate(all_results, 1):
            g2_str = f"{result['g2']:.1e}" if result['g2'] is not None else "N/A"
            g3_str = f"{result['g3']:.1e}" if result['g3'] is not None else "N/A"
            momentum_str = result['momentum_flavor'] if result['momentum_flavor'] else "N/A"
            
            f.write(f"{i:<4} {result['optimizer']:<8} {g2_str:<10} {g3_str:<10} {momentum_str:<12} "
                   f"{result['lr']:<10.1e} {result['weight_decay']:<10.1e} {result['final_val_loss']:<10.6f} "
                   f"{os.path.basename(result['folder']):<20} {result['filename']}\n")
    
    print(f"Summary saved as {output_file}")
    
    # Print top 10 results
    print(f"\nTop 10 results:")
    print(f"{'Rank':<4} {'Optimizer':<8} {'G2':<10} {'G3':<10} {'Momentum':<12} {'Val Loss':<10}")
    print("-"*60)
    for i, result in enumerate(all_results[:10], 1):
        g2_str = f"{result['g2']:.1e}" if result['g2'] is not None else "N/A"
        g3_str = f"{result['g3']:.1e}" if result['g3'] is not None else "N/A"
        momentum_str = result['momentum_flavor'] if result['momentum_flavor'] else "N/A"
        print(f"{i:<4} {result['optimizer']:<8} {g2_str:<10} {g3_str:<10} {momentum_str:<12} {result['final_val_loss']:<10.6f}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize results from multiple folders")
    parser.add_argument("folders", nargs='+', help="Folders to search for results")
    parser.add_argument("--tanea_pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match Tanea result files")
    parser.add_argument("--adamw_pattern", type=str, default="*adamw_baseline*.pkl",
                       help="Pattern to match AdamW baseline files")
    parser.add_argument("--output_prefix", type=str, default="multi_folder",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    try:
        # Load results from all folders
        tanea_results, adamw_results = load_results_from_folders(
            args.folders, args.tanea_pattern, args.adamw_pattern)
        
        if not tanea_results and not adamw_results:
            print("No results found in any of the specified folders!")
            return
        
        # Create visualization
        viz_output = f"{args.output_prefix}_visualization.pdf"
        create_enhanced_visualization(tanea_results, adamw_results, viz_output)
        
        # Create summary table
        summary_output = f"{args.output_prefix}_summary.txt"
        create_summary_table(tanea_results, adamw_results, summary_output)
        
        # Print statistics
        print(f"\nStatistics:")
        print(f"  Folders searched: {len(args.folders)}")
        print(f"  Tanea results: {len(tanea_results)}")
        print(f"  AdamW results: {len(adamw_results)}")
        
        if tanea_results:
            g2_values = set(data['config'].get('tanea_g2', 0) for data in tanea_results)
            g3_values = set(data['config'].get('tanea_g3', 0) for data in tanea_results)
            momentum_flavors = set(data['momentum_flavor'] for data in tanea_results)
            print(f"  G2 values: {sorted(g2_values)}")
            print(f"  G3 values: {sorted(g3_values)}")
            print(f"  Momentum flavors: {sorted(momentum_flavors)}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()