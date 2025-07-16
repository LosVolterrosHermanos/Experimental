#!/usr/bin/env python
"""
Script to visualize results from multiple folders with FLOPS-based x-axis.
Adapted from plot_multi_folder_visualization.py with:
1. FLOPS calculation instead of tokens (6*B*D*T where B=batch_size, D=num_params, T=tokens)
2. Different colors for momentum flavors instead of line styles
3. Special labeling for tanea_kappa=1.0 as adam-star-# and tanea as dana-star-#
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

def create_enhanced_visualization(tanea_results, adamw_results=None, output_file=None, use_flops_per_batch=False):
    """Create visualization with FLOPS-based x-axis and momentum flavor coloring."""
    
    fig, ax = plt.subplots(figsize=(25, 12))
    
    # Get unique momentum flavors for colors
    momentum_flavors = sorted(set(data['momentum_flavor'] for data in tanea_results))
    
    # Define colors for momentum flavors
    flavor_colors = plt.cm.tab10(np.linspace(0, 1, len(momentum_flavors)))
    flavor_color_map = {flavor: flavor_colors[i] for i, flavor in enumerate(momentum_flavors)}
    
    # Plot AdamW baselines first with black color
    adamw_handles = []
    if adamw_results:
        for i, baseline in enumerate(adamw_results):
            config = baseline['config']
            metrics = baseline['metrics']
            num_params = baseline['num_params']
            
            steps = np.array(metrics['step'])
            val_losses = np.array(metrics['val_loss'])
            
            # Calculate FLOPS: 6*B*D*T where B=batch_size, D=num_params, T=tokens
            batch_size = np.float32(config["batch_size"])
            seq_len = np.float32(config["seq_len"])
            num_params_f32 = np.float32(num_params)
            tokens_per_step = batch_size * seq_len
            tokens = steps.astype(np.float32) * tokens_per_step
            flops = 6.0 * batch_size * num_params_f32 * tokens
            
            if use_flops_per_batch:
                x_values = flops / batch_size
                x_label = 'Training FLOPS per Batch'
            else:
                x_values = flops
                x_label = 'Training FLOPS'
            
            print(f"AdamW FLOPS debug - B={batch_size}, D={num_params_f32}, T_per_step={tokens_per_step}")
            print(f"  Steps range: {steps.min()}-{steps.max()}, Tokens range: {tokens.min():.2e}-{tokens.max():.2e}")
            print(f"  FLOPS range: {flops.min():.2e}-{flops.max():.2e}")
            if use_flops_per_batch:
                print(f"  FLOPS/batch range: {x_values.min():.2e}-{x_values.max():.2e}")
            
            lr = config.get('lr', 0)
            wd = config.get('weight_decay', 0)
            label = f"Adam LR={lr:.1e} WD={wd:.1e}"
            
            line = ax.loglog(x_values, val_losses, linestyle='-', color='black', alpha=0.8, 
                           linewidth=2, label=label)
            adamw_handles.extend(line)
    
    # Process Tanea results with special labeling
    tanea_handles = []
    for data in tanea_results:
        config = data['config']
        metrics = data['metrics']
        momentum_flavor = data['momentum_flavor']
        num_params = data['num_params']
        
        steps = np.array(metrics['step'])
        val_losses = np.array(metrics['val_loss'])
        
        # Calculate FLOPS: 6*B*D*T where B=batch_size, D=num_params, T=tokens
        batch_size = np.float32(config["batch_size"])
        seq_len = np.float32(config["seq_len"])
        num_params_f32 = np.float32(num_params)
        tokens_per_step = batch_size * seq_len
        tokens = steps.astype(np.float32) * tokens_per_step
        flops = 6.0 * batch_size * num_params_f32 * tokens
        
        if use_flops_per_batch:
            x_values = flops / batch_size
        else:
            x_values = flops
        
        print(f"Tanea FLOPS debug - B={batch_size}, D={num_params_f32}, T_per_step={tokens_per_step}")
        print(f"  Steps range: {steps.min()}-{steps.max()}, Tokens range: {tokens.min():.2e}-{tokens.max():.2e}")
        print(f"  FLOPS range: {flops.min():.2e}-{flops.max():.2e}")
        if use_flops_per_batch:
            print(f"  FLOPS/batch range: {x_values.min():.2e}-{x_values.max():.2e}")
        
        # Extract momentum flavor number (e.g., 'mk3' -> '3')
        flavor_num = momentum_flavor.replace('mk', '') if momentum_flavor.startswith('mk') else momentum_flavor
        
        # Check for tanea_kappa=1.0 to label as adam-star-#
        tanea_kappa = config.get('tanea_kappa', None)
        if tanea_kappa is not None and abs(tanea_kappa - 1.0) < 1e-6:
            label = f"adam-star-{flavor_num}"
            color = 'red'  # Special color for adam-star
        else:
            label = f"dana-star-{flavor_num}"
            color = flavor_color_map.get(momentum_flavor, 'gray')
        
        # Add parameter info to label
        g2 = config.get('tanea_g2', 0)
        g3 = config.get('tanea_g3', 0)
        wd = config.get('weight_decay', 0)
        label += f" G2={g2:.1e} G3={g3:.1e} WD={wd:.1e}"
        
        line = ax.loglog(x_values, val_losses, linestyle='-', color=color, alpha=0.8,
                       linewidth=2, label=label)
        tanea_handles.extend(line)
    
    # Set axis labels and title
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel('Validation Loss', fontsize=14)
    title_suffix = "(FLOPS per Batch)" if use_flops_per_batch else "(FLOPS-based)"
    ax.set_title(f'Multi-Folder Training Results Comparison {title_suffix}', fontsize=16)
    
    # Format x-axis for FLOPS
    def format_flops(x, pos):
        if x >= 1e18:
            return f'{x/1e18:.1f}E'
        elif x >= 1e15:
            return f'{x/1e15:.1f}P'
        elif x >= 1e12:
            return f'{x/1e12:.1f}T'
        elif x >= 1e9:
            return f'{x/1e9:.1f}G'
        elif x >= 1e6:
            return f'{x/1e6:.1f}M'
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'
        else:
            return f'{x:.0f}'
    
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_flops))
    
    # Add grid
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    
    # Get all handles and labels for the main legend
    handles, labels = ax.get_legend_handles_labels()
    
    # Create a separate legend for momentum flavors (colors)
    from matplotlib.lines import Line2D
    flavor_legend_elements = []
    for flavor in sorted(momentum_flavors):
        if flavor in flavor_color_map:
            color = flavor_color_map[flavor]
            flavor_legend_elements.append(Line2D([0], [0], color=color, linestyle='-', 
                                                label=f'{flavor}'))
    
    if flavor_legend_elements:
        flavor_legend = ax.legend(handles=flavor_legend_elements, loc='upper right', 
                                 title='Momentum Flavors (color)', fontsize=9, title_fontsize=10)
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
        output_file = "chinchilla_visualization.pdf"
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Visualization saved as {output_file}")
    plt.show()

def create_summary_table(tanea_results, adamw_results=None, output_file=None):
    """Create a summary table of all results."""
    
    if output_file is None:
        output_file = "chinchilla_summary.txt"
    
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
                'tanea_kappa': None,
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
        
        # Extract momentum flavor number
        momentum_flavor = data['momentum_flavor']
        flavor_num = momentum_flavor.replace('mk', '') if momentum_flavor.startswith('mk') else momentum_flavor
        
        # Check for tanea_kappa=1.0 to determine labeling
        tanea_kappa = config.get('tanea_kappa', None)
        if tanea_kappa is not None and abs(tanea_kappa - 1.0) < 1e-6:
            optimizer_name = f"adam-star-{flavor_num}"
        else:
            optimizer_name = f"dana-star-{flavor_num}"
        
        all_results.append({
            'optimizer': optimizer_name,
            'g2': config.get('tanea_g2', 0),
            'g3': config.get('tanea_g3', 0),
            'momentum_flavor': data['momentum_flavor'],
            'tanea_kappa': tanea_kappa,
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
        f.write("Chinchilla Results Summary (FLOPS-based)\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"{'Rank':<4} {'Optimizer':<15} {'G2':<10} {'G3':<10} {'Kappa':<8} {'Momentum':<12} {'LR':<10} {'WD':<10} {'Val Loss':<10} {'Folder':<20} {'File'}\n")
        f.write("-"*140 + "\n")
        
        for i, result in enumerate(all_results, 1):
            g2_str = f"{result['g2']:.1e}" if result['g2'] is not None else "N/A"
            g3_str = f"{result['g3']:.1e}" if result['g3'] is not None else "N/A"
            kappa_str = f"{result['tanea_kappa']:.1f}" if result['tanea_kappa'] is not None else "N/A"
            momentum_str = result['momentum_flavor'] if result['momentum_flavor'] else "N/A"
            
            f.write(f"{i:<4} {result['optimizer']:<15} {g2_str:<10} {g3_str:<10} {kappa_str:<8} {momentum_str:<12} "
                   f"{result['lr']:<10.1e} {result['weight_decay']:<10.1e} {result['final_val_loss']:<10.6f} "
                   f"{os.path.basename(result['folder']):<20} {result['filename']}\n")
    
    print(f"Summary saved as {output_file}")
    
    # Print top 10 results
    print(f"\nTop 10 results:")
    print(f"{'Rank':<4} {'Optimizer':<15} {'G2':<10} {'G3':<10} {'Kappa':<8} {'Val Loss':<10}")
    print("-"*70)
    for i, result in enumerate(all_results[:10], 1):
        g2_str = f"{result['g2']:.1e}" if result['g2'] is not None else "N/A"
        g3_str = f"{result['g3']:.1e}" if result['g3'] is not None else "N/A"
        kappa_str = f"{result['tanea_kappa']:.1f}" if result['tanea_kappa'] is not None else "N/A"
        print(f"{i:<4} {result['optimizer']:<15} {g2_str:<10} {g3_str:<10} {kappa_str:<8} {result['final_val_loss']:<10.6f}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize results from multiple folders with FLOPS-based x-axis")
    parser.add_argument("folders", nargs='+', help="Folders to search for results")
    parser.add_argument("--tanea_pattern", type=str, default="*tanea_results*.pkl",
                       help="Pattern to match Tanea result files")
    parser.add_argument("--adamw_pattern", type=str, default="*adamw_baseline*.pkl",
                       help="Pattern to match AdamW baseline files")
    parser.add_argument("--output_prefix", type=str, default="chinchilla",
                       help="Prefix for output files")
    
    args = parser.parse_args()
    
    try:
        # Load results from all folders
        tanea_results, adamw_results = load_results_from_folders(
            args.folders, args.tanea_pattern, args.adamw_pattern)
        
        if not tanea_results and not adamw_results:
            print("No results found in any of the specified folders!")
            return
        
        # Create both visualizations
        # 1. Standard FLOPS visualization
        viz_output = f"{args.output_prefix}_visualization.pdf"
        create_enhanced_visualization(tanea_results, adamw_results, viz_output, use_flops_per_batch=False)
        
        # 2. FLOPS per batch visualization
        viz_output_per_batch = f"{args.output_prefix}_visualization_per_batch.pdf"
        create_enhanced_visualization(tanea_results, adamw_results, viz_output_per_batch, use_flops_per_batch=True)
        
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
            kappa_values = set(data['config'].get('tanea_kappa', None) for data in tanea_results if data['config'].get('tanea_kappa') is not None)
            
            print(f"  G2 values: {sorted(g2_values)}")
            print(f"  G3 values: {sorted(g3_values)}")
            print(f"  Momentum flavors: {sorted(momentum_flavors)}")
            print(f"  Kappa values: {sorted(kappa_values) if kappa_values else 'None'}")
        
    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()