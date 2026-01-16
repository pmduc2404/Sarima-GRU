"""
Complete Evaluation Script for SARIMA-GRU Model

Comprehensive model evaluation with detailed metrics and analysis.

Usage:
    python scripts/evaluate.py --model_path models/sarima_gru_trained.pth \\
                               --data_path data/raw/your_data.csv
    python scripts/evaluate.py --help
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sarima_gru import SARIMAGRU, prepare_data


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Evaluate SARIMA-GRU Model with detailed metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate.py --model_path models/sarima_gru_trained.pth \\
                             --data_path data/raw/your_data.csv
  
  # Detailed evaluation with all plots
  python scripts/evaluate.py --model_path models/sarima_gru_trained.pth \\
                             --data_path data/raw/your_data.csv \\
                             --detailed_report
  
  # Save comparison report
  python scripts/evaluate.py --model_path models/sarima_gru_trained.pth \\
                             --data_path data/raw/your_data.csv \\
                             --save_report results/evaluation_report.csv
        """
    )
    
    parser.add_argument('--model_path', type=str, default='models/sarima_gru_final_model.pth',
                       help='Path to trained model')
    parser.add_argument('--data_path', type=str, default='DataSet/AnKhe.csv',
                       help='Path to evaluation data')
    parser.add_argument('--target_column', type=str, default='Lake water level (m)',
                       help='Target column name')
    parser.add_argument('--sequence_length', type=int, default=48,
                       help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--detailed_report', action='store_true',
                       help='Generate detailed evaluation report')
    parser.add_argument('--save_report', type=str, default=None,
                       help='Save detailed report to CSV file')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    
    return parser.parse_args()


def load_model(checkpoint_path, device):
    """Load trained model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SARIMAGRU(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """Evaluate model and get predictions"""
    predictions = []
    actuals = []
    
    print("\n Evaluating model...")
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(device)
            outputs, _ = model(batch_x)
            pred = outputs[:, -1, :].cpu().numpy()
            actual = batch_y.cpu().numpy()
            
            predictions.extend(pred.flatten())
            actuals.extend(actual.flatten())
            
            if (i + 1) % max(1, len(test_loader) // 5) == 0:
                progress = (i + 1) / len(test_loader) * 100
                print(f"  Progress: {progress:.1f}% ({i + 1}/{len(test_loader)})")
    
    return np.array(predictions), np.array(actuals)


def calculate_comprehensive_metrics(predictions, actuals):
    """Calculate comprehensive evaluation metrics"""
    from sklearn.metrics import (
        mean_squared_error, mean_absolute_error, 
        mean_absolute_percentage_error, r2_score
    )
    
    # Basic metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    r2 = r2_score(actuals, predictions)
    
    # Additional metrics
    residuals = actuals - predictions
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    # MASE (Mean Absolute Scaled Error)
    n = len(actuals)
    d = np.mean(np.abs(np.diff(actuals)))
    mase = mae / d if d != 0 else np.inf
    
    # Directional Accuracy
    actual_direction = np.diff(actuals) > 0
    pred_direction = np.diff(predictions) > 0
    directional_accuracy = np.mean(actual_direction == pred_direction)
    
    # Theil's U coefficient
    numerator = np.sum((actuals[1:] - predictions[1:]) ** 2)
    denominator = np.sum((actuals[1:] - actuals[:-1]) ** 2)
    theils_u = np.sqrt(numerator / denominator) if denominator != 0 else np.inf
    
    # Forecast error percentage
    max_error = np.max(np.abs(residuals))
    min_error = np.min(np.abs(residuals))
    median_error = np.median(np.abs(residuals))
    
    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'Mean Residual': mean_residual,
        'Std Residual': std_residual,
        'MASE': mase,
        'Directional Accuracy': directional_accuracy,
        "Theil's U": theils_u,
        'Max Error': max_error,
        'Min Error': min_error,
        'Median Error': median_error,
        'N Samples': len(actuals)
    }
    
    return metrics, residuals


def create_evaluation_report(predictions, actuals, metrics, residuals):
    """Create comprehensive evaluation report"""
    report = f"""
{'='*70}
SARIMA-GRU MODEL EVALUATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

OVERALL PERFORMANCE METRICS
{'-'*70}
MSE (Mean Squared Error):          {metrics['MSE']:.6f}
RMSE (Root Mean Squared Error):    {metrics['RMSE']:.6f}
MAE (Mean Absolute Error):         {metrics['MAE']:.6f}
MAPE (Mean Absolute %Error):       {metrics['MAPE']:.6f} ({metrics['MAPE']*100:.2f}%)
R² (Coefficient of Determination): {metrics['R²']:.6f}
MASE (Mean Absolute Scaled Error): {metrics['MASE']:.6f}
Theil's U Coefficient:             {metrics["Theil's U"]:.6f}

RESIDUAL ANALYSIS
{'-'*70}
Mean Residual:                     {metrics['Mean Residual']:.6f}
Std Dev Residual:                  {metrics['Std Residual']:.6f}
Max Absolute Error:                {metrics['Max Error']:.6f}
Min Absolute Error:                {metrics['Min Error']:.6f}
Median Absolute Error:             {metrics['Median Error']:.6f}

DIRECTIONAL METRICS
{'-'*70}
Directional Accuracy:              {metrics['Directional Accuracy']:.4f} ({metrics['Directional Accuracy']*100:.2f}%)

SAMPLE STATISTICS
{'-'*70}
Number of Samples:                 {metrics['N Samples']:,}
Actual Value Range:                [{actuals.min():.6f}, {actuals.max():.6f}]
Predicted Value Range:             [{predictions.min():.6f}, {predictions.max():.6f}]
Actual Mean:                       {actuals.mean():.6f}
Predicted Mean:                    {predictions.mean():.6f}

INTERPRETATION
{'-'*70}
"""
    
    if metrics['R²'] > 0.9:
        report += "Excellent model fit (R² > 0.9)\n"
    elif metrics['R²'] > 0.7:
        report += "Good model fit (R² > 0.7)\n"
    elif metrics['R²'] > 0.5:
        report += "Moderate model fit (R² > 0.5)\n"
    else:
        report += "Poor model fit (R² < 0.5)\n"
    
    if metrics['MAPE'] < 0.05:
        report += "Excellent accuracy (MAPE < 5%)\n"
    elif metrics['MAPE'] < 0.10:
        report += "Good accuracy (MAPE < 10%)\n"
    elif metrics['MAPE'] < 0.20:
        report += "Moderate accuracy (MAPE < 20%)\n"
    else:
        report += "Poor accuracy (MAPE > 20%)\n"
    
    if metrics['Directional Accuracy'] > 0.7:
        report += "Excellent directional accuracy (>70%)\n"
    elif metrics['Directional Accuracy'] > 0.6:
        report += "Moderate directional accuracy (>60%)\n"
    else:
        report += "Poor directional accuracy (<60%)\n"
    
    report += "\n" + "="*70 + "\n"
    
    return report


def plot_comprehensive_evaluation(predictions, actuals, residuals, output_dir):
    """Create comprehensive evaluation plots"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    
    # Figure 1: Main predictions
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('SARIMA-GRU Model Evaluation', fontsize=16, fontweight='bold', y=1.00)
    
    # Plot 1: Full time series
    ax = axes[0, 0]
    ax.plot(actuals, 'b-', label='Actual', linewidth=2, alpha=0.8)
    ax.plot(predictions, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Full Time Series Prediction', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Zoom - first 200 samples
    ax = axes[0, 1]
    ax.plot(actuals[:200], 'b-', label='Actual', linewidth=2, marker='o', markersize=3, alpha=0.8)
    ax.plot(predictions[:200], 'r--', label='Predicted', linewidth=2, marker='x', markersize=3, alpha=0.8)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('First 200 Samples (Detail View)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Residuals distribution
    ax = axes[1, 0]
    ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
    ax.axvline(x=np.mean(residuals), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(residuals):.4f}')
    ax.set_xlabel('Residuals', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Residuals Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Scatter - Actual vs Predicted
    ax = axes[1, 1]
    ax.scatter(actuals, predictions, alpha=0.5, s=20, color='blue')
    min_val = min(actuals.min(), predictions.min())
    max_val = max(actuals.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Fit')
    ax.set_xlabel('Actual Values', fontsize=11)
    ax.set_ylabel('Predicted Values', fontsize=11)
    ax.set_title('Actual vs Predicted Scatter', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_evaluation_main.png'), dpi=300, bbox_inches='tight')
    print(f"  ✓ Main evaluation plot saved")
    plt.close()
    
    # Figure 2: Error analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Error Analysis', fontsize=16, fontweight='bold', y=1.00)
    
    errors = np.abs(residuals)
    
    # Plot 1: Absolute errors over time
    ax = axes[0, 0]
    ax.plot(errors, 'o-', linewidth=2, markersize=4, color='red', alpha=0.7)
    ax.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='red')
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Absolute Error', fontsize=11)
    ax.set_title('Absolute Errors Over Time', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Cumulative error
    ax = axes[0, 1]
    cumulative_error = np.cumsum(errors)
    ax.plot(cumulative_error, 'g-', linewidth=2)
    ax.set_xlabel('Sample Index', fontsize=11)
    ax.set_ylabel('Cumulative Error', fontsize=11)
    ax.set_title('Cumulative Error', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Error Q-Q plot
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Residuals vs Normal)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Error box plot by quantiles
    ax = axes[1, 1]
    quantiles = [residuals[int(i*len(residuals)/5):(i+1)*int(len(residuals)/5)] for i in range(5)]
    bp = ax.boxplot(quantiles, labels=[f'Q{i+1}' for i in range(5)])
    ax.set_ylabel('Residuals', fontsize=11)
    ax.set_title('Residuals by Quantiles', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '02_error_analysis.png'), dpi=300, bbox_inches='tight')
    print(f"Error analysis plot saved")
    plt.close()


def main():
    """Main evaluation function"""
    args = parse_args()
    
    print("=" * 70)
    print("SARIMA-GRU MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {args.model_path}")
    print(f"Data: {args.data_path}")
    print(f"Output Dir: {args.output_dir}")
    print("=" * 70)
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model and data
    print(f"\n Loading model...")
    model, checkpoint = load_model(args.model_path, device)
    print("✓ Model loaded")
    
    print(f"\nLoading data...")
    if not os.path.exists(args.data_path):
        print(f"Data not found: {args.data_path}")
        return
    
    _, test_dataset, _ = prepare_data(
        args.data_path,
        sequence_length=args.sequence_length,
        target_column=args.target_column
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Data loaded: {len(test_dataset)} samples")
    
    # Evaluate
    predictions, actuals = evaluate_model(model, test_loader, device)
    print(f"Evaluation complete")
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics, residuals = calculate_comprehensive_metrics(predictions, actuals)
    
    # Print report
    report = create_evaluation_report(predictions, actuals, metrics, residuals)
    print(report)
    
    # Save report
    report_path = os.path.join(args.output_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Report saved: {report_path}")
    
    # Save predictions
    if args.save_report:
        df = pd.DataFrame({
            'Actual': actuals,
            'Predicted': predictions,
            'Error': residuals,
            'Abs_Error': np.abs(residuals),
            'Pct_Error': np.abs(residuals) / np.abs(actuals) * 100
        })
        df.to_csv(args.save_report, index=False)
        print(f"Detailed predictions saved: {args.save_report}")
    
    # Create plots
    if args.detailed_report:
        print(f"\nCreating detailed plots...")
        plot_comprehensive_evaluation(predictions, actuals, residuals, args.output_dir)
        print(f"Plots saved to {args.output_dir}")
    
    print(f"\nEvaluation complete!")


if __name__ == '__main__':
    main()
