#!/usr/bin/env python
"""
Visualization script for tanea grid search results.
Creates:
1. Loss curves for all combinations
2. Heatmaps of last-iterate loss vs g2 and g3 for each momentum flavor
3. Multivariate linear fits of last-iterate loss vs g2 and g3 for each momentum flavor
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import glob
from pathlib import Path
import argparse
from scipy import stats
from sklearn.linear_model import LinearRegression
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

def load_grid_search_results(results_dir):
    """Load all results from a grid search directory."""
    
    # Find all pickle files in the results directory
    pickle_files = glob.glob(os.path.join(results_dir, "*.pkl"))
    
    if not pickle_files:
        raise ValueError(f"No pickle files found in {results_dir}")
    
    print(f"Found {len(pickle_files)} result files")
    
    results_data = []
    
    for pkl_file in pickle_files:
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract relevant information
            config = data['config']
            metrics = data['metrics']
            num_params = data.get('num_params', 0)
            
            # Extract parameters from filename for additional verification
            filename = os.path.basename(pkl_file)
            
            results_data.append({
                'config': config,
                'metrics': metrics,
                'num_params': num_params,
                'filename': filename,
                'tanea_g2': config['tanea_g2'],
                'tanea_g3': config['tanea_g3'],
                'momentum_flavor': config['momentum_flavor'],
                'final_train_loss': metrics['train_loss'][-1] if metrics['train_loss'] else float('nan'),
                'final_val_loss': metrics['val_loss'][-1] if metrics['val_loss'] else float('nan'),
                'steps': np.array(metrics['step']),
                'train_losses': np.array(metrics['train_loss']),
                'val_losses': np.array(metrics['val_loss'])
            })
            
            print(f"Loaded: g2={config['tanea_g2']}, g3={config['tanea_g3']}, flavor={config['momentum_flavor']}")
        
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            continue
    
    if not results_data:
        raise ValueError("No valid results data found")
    
    print(f"\nSuccessfully loaded {len(results_data)} results")
    return results_data

def plot_loss_curves(results_data, output_dir):
    """Plot all loss curves grouped by momentum flavor."""
    
    # Group by momentum flavor
    flavors = list(set([r['momentum_flavor'] for r in results_data]))
    flavors.sort()
    
    fig, axes = plt.subplots(1, len(flavors), figsize=(6*len(flavors), 5))
    if len(flavors) == 1:
        axes = [axes]
    
    for i, flavor in enumerate(flavors):
        ax = axes[i]
        flavor_results = [r for r in results_data if r['momentum_flavor'] == flavor]
        
        # Plot each combination
        for result in flavor_results:
            steps = result['steps']
            train_losses = result['train_losses']
            val_losses = result['val_losses']
            
            # Plot training loss
            ax.plot(steps, train_losses, alpha=0.7, linewidth=1, 
                   label=f"g2={result['tanea_g2']}, g3={result['tanea_g3']}")
            
            # Plot validation loss if available
            if not np.isnan(val_losses).all():
                ax.plot(steps, val_losses, alpha=0.5, linewidth=1, linestyle='--')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title(f'Momentum Flavor: {flavor}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Use log scale for y-axis if losses vary significantly
        if len(flavor_results) > 0:
            all_losses = []
            for r in flavor_results:
                all_losses.extend(r['train_losses'])
                if not np.isnan(r['val_losses']).all():
                    all_losses.extend(r['val_losses'])
            
            if max(all_losses) / min(all_losses) > 10:
                ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves_by_flavor.pdf'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'loss_curves_by_flavor.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_heatmaps_and_linear_fits(results_data, output_dir):
    """Create heatmaps and linear fits for each momentum flavor."""
    
    # Group by momentum flavor
    flavors = list(set([r['momentum_flavor'] for r in results_data]))
    flavors.sort()
    
    for flavor in flavors:
        flavor_results = [r for r in results_data if r['momentum_flavor'] == flavor]
        
        if len(flavor_results) == 0:
            continue
        
        # Extract unique g2 and g3 values
        g2_values = sorted(list(set([r['tanea_g2'] for r in flavor_results])))
        g3_values = sorted(list(set([r['tanea_g3'] for r in flavor_results])))
        
        # Create heatmap data
        heatmap_data = np.full((len(g2_values), len(g3_values)), np.nan)
        
        # Fill heatmap with final losses
        for result in flavor_results:
            g2_idx = g2_values.index(result['tanea_g2'])
            g3_idx = g3_values.index(result['tanea_g3'])
            heatmap_data[g2_idx, g3_idx] = result['final_train_loss']
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Heatmap
        im = ax1.imshow(heatmap_data, cmap='viridis', aspect='auto', origin='lower')
        ax1.set_xlabel('g3')
        ax1.set_ylabel('g2')
        ax1.set_title(f'Last-Iterate Loss Heatmap\nMomentum Flavor: {flavor}')
        
        # Set tick labels
        ax1.set_xticks(range(len(g3_values)))
        ax1.set_yticks(range(len(g2_values)))
        ax1.set_xticklabels([f'{g3:.0e}' for g3 in g3_values])
        ax1.set_yticklabels([f'{g2:.0e}' for g2 in g2_values])
        
        # Add colorbar
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        
        # Plot 2: Linear fit analysis
        # Prepare data for linear regression
        X = []
        y = []
        
        for result in flavor_results:
            X.append([np.log10(result['tanea_g2']), np.log10(result['tanea_g3'])])
            y.append(result['final_train_loss'])
        
        X = np.array(X)
        y = np.array(y)
        
        # Fit linear regression
        reg = LinearRegression()
        reg.fit(X, y)
        
        # Calculate R-squared
        y_pred = reg.predict(X)
        r_squared = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        
        # Create scatter plot with fitted surface
        ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('log10(g2)')
        ax2.set_ylabel('log10(g3)')
        ax2.set_title(f'Linear Fit Analysis\nR² = {r_squared:.3f}')
        
        # Add colorbar
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=100, alpha=0.7)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(scatter, cax=cax)
        
        # Print regression coefficients
        print(f"\nLinear fit for {flavor}:")
        print(f"  Intercept: {reg.intercept_:.6f}")
        print(f"  g2 coefficient: {reg.coef_[0]:.6f}")
        print(f"  g3 coefficient: {reg.coef_[1]:.6f}")
        print(f"  R²: {r_squared:.6f}")
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'heatmap_and_fit_{flavor}.pdf'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, f'heatmap_and_fit_{flavor}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def create_summary_statistics(results_data, output_dir):
    """Create summary statistics and tables."""
    
    # Group by momentum flavor
    flavors = list(set([r['momentum_flavor'] for r in results_data]))
    flavors.sort()
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for flavor in flavors:
        flavor_results = [r for r in results_data if r['momentum_flavor'] == flavor]
        
        if len(flavor_results) == 0:
            continue
        
        print(f"\nMomentum Flavor: {flavor}")
        print("-" * 40)
        
        # Find best performing configuration
        best_result = min(flavor_results, key=lambda x: x['final_train_loss'])
        worst_result = max(flavor_results, key=lambda x: x['final_train_loss'])
        
        print(f"Best configuration:")
        print(f"  g2={best_result['tanea_g2']}, g3={best_result['tanea_g3']}")
        print(f"  Final train loss: {best_result['final_train_loss']:.6f}")
        print(f"  Final val loss: {best_result['final_val_loss']:.6f}")
        
        print(f"\nWorst configuration:")
        print(f"  g2={worst_result['tanea_g2']}, g3={worst_result['tanea_g3']}")
        print(f"  Final train loss: {worst_result['final_train_loss']:.6f}")
        print(f"  Final val loss: {worst_result['final_val_loss']:.6f}")
        
        # Calculate statistics
        final_losses = [r['final_train_loss'] for r in flavor_results]
        print(f"\nStatistics:")
        print(f"  Mean final loss: {np.mean(final_losses):.6f}")
        print(f"  Std final loss: {np.std(final_losses):.6f}")
        print(f"  Min final loss: {np.min(final_losses):.6f}")
        print(f"  Max final loss: {np.max(final_losses):.6f}")
        
        # Create table of all results
        print(f"\nAll configurations:")
        print(f"{'g2':<10} {'g3':<10} {'Train Loss':<12} {'Val Loss':<12}")
        print("-" * 50)
        
        for result in sorted(flavor_results, key=lambda x: (x['tanea_g2'], x['tanea_g3'])):
            print(f"{result['tanea_g2']:<10.0e} {result['tanea_g3']:<10.0e} "
                  f"{result['final_train_loss']:<12.6f} {result['final_val_loss']:<12.6f}")

def main():
    """Main function to load data and create visualizations."""
    parser = argparse.ArgumentParser(description="Visualize tanea grid search results")
    parser.add_argument("--results_dir", type=str, required=True,
                       help="Directory containing grid search results")
    parser.add_argument("--output_dir", type=str, default="grid_search_visualizations",
                       help="Output directory for plots")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Load results data
        if not os.path.exists(args.results_dir):
            print(f"Results directory '{args.results_dir}' not found!")
            return
        
        results_data = load_grid_search_results(args.results_dir)
        print(f"\nLoaded results for {len(results_data)} configurations")
        
        # Create visualizations
        print("\nCreating loss curves...")
        plot_loss_curves(results_data, args.output_dir)
        
        print("Creating heatmaps and linear fits...")
        create_heatmaps_and_linear_fits(results_data, args.output_dir)
        
        print("Creating summary statistics...")
        create_summary_statistics(results_data, args.output_dir)
        
        print(f"\nAll visualizations saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 