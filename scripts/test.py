"""
Complete Testing Script for SARIMA-GRU Model

Usage:
    python scripts/test.py --model_path models/sarima_gru_trained.pth \\
                           --data_path data/raw/your_data.csv
    python scripts/test.py --help
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sarima_gru import SARIMAGRU, prepare_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Test SARIMA-GRU Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with saved model
  python scripts/test.py --model_path models/sarima_gru_trained.pth \\
                         --data_path data/raw/your_data.csv
  
  # Test with visualization
  python scripts/test.py --model_path models/sarima_gru_trained.pth \\
                         --data_path data/raw/your_data.csv \\
                         --plot_results
        """
    )
    
    parser.add_argument('--model_path', type=str, default='models/sarima_gru_final_model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, default='DataSet/AnKhe.csv',
                       help='Path to test data CSV file')
    parser.add_argument('--target_column', type=str, default='Lake water level (m)',
                       help='Target column name')
    parser.add_argument('--sequence_length', type=int, default=48,
                       help='Sequence length (default: 48)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--plot_results', action='store_true',
                       help='Plot predictions vs actuals')
    parser.add_argument('--output_path', type=str, default='results/test_results.txt',
                       help='Path to save test results')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Model checkpoint not found at {checkpoint_path}")
        return None, None
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = SARIMAGRU(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint


def test_model(model, test_loader, device):
    """Test model on test data"""
    predictions = []
    actuals = []
    
    print("\n🧪 Testing model...")
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            outputs, _ = model(batch_x)
            pred = outputs[:, -1, :].cpu().numpy()
            actual = batch_y.cpu().numpy()
            
            predictions.extend(pred.flatten())
            actuals.extend(actual.flatten())
            
            if (i + 1) % max(1, len(test_loader) // 5) == 0:
                print(f"  Processed {i + 1}/{len(test_loader)} batches")
    
    return np.array(predictions), np.array(actuals)


def calculate_metrics(predictions, actuals):
    """Calculate evaluation metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
    
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    # Additional metrics
    r2 = 1 - (np.sum((actuals - predictions) ** 2) / np.sum((actuals - np.mean(actuals)) ** 2))
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }


def plot_results(predictions, actuals, save_path='results/test_results.png'):
    """Plot predictions vs actuals"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Full predictions vs actuals
    ax = axes[0, 0]
    ax.plot(actuals, 'b-', label='Actual', linewidth=2, alpha=0.7)
    ax.plot(predictions, 'r--', label='Predicted', linewidth=2, alpha=0.7)
    ax.set_xlabel('Sample', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('Predictions vs Actual Values', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom into first 200 samples
    ax = axes[0, 1]
    ax.plot(actuals[:200], 'b-', label='Actual', linewidth=2, marker='o', markersize=4, alpha=0.7)
    ax.plot(predictions[:200], 'r--', label='Predicted', linewidth=2, marker='x', markersize=4, alpha=0.7)
    ax.set_xlabel('Sample', fontsize=10)
    ax.set_ylabel('Value', fontsize=10)
    ax.set_title('First 200 Samples (Detail)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals
    residuals = actuals - predictions
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.set_xlabel('Residuals', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter plot
    ax = axes[1, 1]
    ax.scatter(actuals, predictions, alpha=0.5, s=20)
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Values', fontsize=10)
    ax.set_ylabel('Predicted Values', fontsize=10)
    ax.set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"\n✓ Results plot saved to {save_path}")
    plt.close()


def main():
    """Main testing function"""
    args = parse_args()
    
    print("=" * 60)
    print("SARIMA-GRU TESTING")
    print("=" * 60)
    print(f"Model path: {args.model_path}")
    print(f"Data path: {args.data_path}")
    print(f"Device: {args.device}")
    print("=" * 60)
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Using device: {device}")
    
    # Load model
    print(f"\n🔧 Loading model from {args.model_path}...")
    model, checkpoint = load_model(args.model_path, device)
    if model is None:
        return
    print("✓ Model loaded successfully!")
    
    # Load data
    print(f"\n📊 Loading test data from {args.data_path}...")
    if not os.path.exists(args.data_path):
        print(f"❌ Error: Data file not found at {args.data_path}")
        return
    
    try:
        _, test_dataset, scalers = prepare_data(
            args.data_path,
            sequence_length=args.sequence_length,
            target_column=args.target_column
        )
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    print(f"✓ Test data loaded: {len(test_dataset)} samples")
    
    # Test model
    try:
        predictions, actuals = test_model(model, test_loader, device)
        print(f"✓ Testing complete: {len(predictions)} predictions")
        
        # Calculate metrics
        print(f"\n📊 Calculating metrics...")
        metrics = calculate_metrics(predictions, actuals)
        
        # Print results
        print("\n" + "=" * 60)
        print("TEST RESULTS")
        print("=" * 60)
        print(f"MSE:   {metrics['MSE']:.6f}")
        print(f"RMSE:  {metrics['RMSE']:.6f}")
        print(f"MAE:   {metrics['MAE']:.6f}")
        print(f"MAPE:  {metrics['MAPE']:.4f} ({metrics['MAPE']*100:.2f}%)")
        print(f"R²:    {metrics['R2']:.6f}")
        print("=" * 60)
        
        # Save results to file
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SARIMA-GRU TEST RESULTS\n")
            f.write("=" * 60 + "\n")
            f.write(f"Model: {args.model_path}\n")
            f.write(f"Data: {args.data_path}\n")
            f.write(f"Samples: {len(predictions)}\n")
            f.write("\n")
            f.write(f"MSE:   {metrics['MSE']:.6f}\n")
            f.write(f"RMSE:  {metrics['RMSE']:.6f}\n")
            f.write(f"MAE:   {metrics['MAE']:.6f}\n")
            f.write(f"MAPE:  {metrics['MAPE']:.4f} ({metrics['MAPE']*100:.2f}%)\n")
            f.write(f"R²:    {metrics['R2']:.6f}\n")
            f.write("=" * 60 + "\n")
        print(f"\n✓ Results saved to {args.output_path}")
        
        # Plot results if requested
        if args.plot_results:
            print(f"\n📈 Creating visualization...")
            plot_results(predictions, actuals, save_path='results/test_results.png')
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
