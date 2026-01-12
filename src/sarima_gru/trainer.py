"""
Training and evaluation utilities for the SARIMA-GRU model.

Provides functions for model training with validation,
evaluation metrics, and checkpoint management.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, device='cuda'):
    """
    Train the model with validation.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs (int): Number of training epochs
        learning_rate (float): Learning rate for optimizer
        device (str): Device to train on ('cuda' or 'cpu')
        
    Returns:
        Dict with training and validation losses
    """
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    epoch_pbar = tqdm(range(num_epochs), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        train_loss = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", 
                         leave=False, unit="batch")
        
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            
            outputs, _ = model(batch_x)
            loss = criterion(outputs[:, -1, :], batch_y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_loss = loss.item()
            train_loss += batch_loss
            
            train_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
        # Validation
        model.eval()
        val_loss = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Val", 
                       leave=False, unit="batch")
        
        with torch.no_grad():
            for batch_x, batch_y in val_pbar:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                outputs, _ = model(batch_x)
                loss = criterion(outputs[:, -1, :], batch_y)
                
                batch_loss = loss.item()
                val_loss += batch_loss
                
                val_pbar.set_postfix({'loss': f'{batch_loss:.6f}'})
        
        # Average losses
        train_loss = train_loss / len(train_loader)
        val_loss = val_loss / len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        epoch_pbar.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'val_loss': f'{val_loss:.6f}'
        })
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'model': model
    }


def evaluate_model(model, test_loader, device='cuda', inverse_transform=None):
    """
    Evaluate model on test data.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test data
        device (str): Device to evaluate on
        inverse_transform: Function to inverse transform predictions
        
    Returns:
        Dict with evaluation metrics
    """
    model = model.to(device)
    model.eval()
    
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            outputs, _ = model(batch_x)
            pred = outputs[:, -1, :].cpu().numpy()
            actual = batch_y.cpu().numpy()
            
            predictions.extend(pred)
            actuals.extend(actual)
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    mse = mean_squared_error(actuals, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actuals, predictions)
    mape = mean_absolute_percentage_error(actuals, predictions)
    
    return {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'predictions': predictions,
        'actuals': actuals
    }


def save_checkpoint(model, optimizer, epoch, path):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        path (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'output_size': model.output_size,
            'num_layers': model.num_layers,
            'seasonal_period': model.seasonal_period
        }
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        path (str): Path to checkpoint
        model: Model to load state into
        optimizer: Optional optimizer to load state into
        
    Returns:
        Epoch number from checkpoint
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint['epoch']
