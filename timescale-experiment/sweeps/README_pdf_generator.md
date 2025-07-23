# Order Statistics PDF Generator

This script generates a comprehensive PDF report showing order statistics plots for all layers in the transformer model.

## Usage

```bash
python generate_order_statistics_pdf.py <pickle_file> [-o output.pdf]
```

## Example

```bash
python generate_order_statistics_pdf.py gpt-medium.pkl -o order_statistics_report.pdf
```

## Output Structure

The generated PDF contains:

1. **Overview Page**: Shows order statistics for:
   - `head/kernel` - Final output layer
   - `wte/embedding` - Word token embeddings

2. **Transformer Block Pages**: One page per transformer block (0-23 for GPT2-medium), showing:
   - `q_proj/kernel` - Query projection
   - `k_proj/kernel` - Key projection  
   - `v_proj/kernel` - Value projection
   - `out_proj/kernel` - Output projection
   - `fc1/kernel` - First MLP layer
   - `fc2/kernel` - Second MLP layer

3. **Additional Layers Page**: Shows other model components like layer norms

## Plot Details

Each plot shows:
- **X-axis**: Order statistic index K = 1.1^k (log scale)
- **Y-axis**: Order statistic values (log scale)
- **Colors**: Training timestamp (plasma colormap)
- **Markers**: Circles for largest order statistics, squares for smallest
- **Grid**: Log-scale grid for easy reading

## Requirements

- matplotlib
- numpy
- pickle (built-in)
- pathlib (built-in)

## Notes

- The script automatically detects the number of transformer blocks in the model
- Each page contains 6 subplots (2x3 grid) for easy comparison
- The colorbar shows the training progression through timestamp colors
- Log scales on both axes help visualize the full range of values