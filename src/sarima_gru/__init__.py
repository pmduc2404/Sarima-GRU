"""
SARIMA-GRU: Hybrid Deep Learning Model for Time Series Forecasting

A combination of SARIMA and GRU neural networks for accurate time series prediction,
specifically designed for lake water level forecasting.
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .model import SARIMAGRUCell, SARIMAGRU
from .data import TimeSeriesDataset, prepare_data
from .trainer import train_model, evaluate_model

__all__ = [
    'SARIMAGRUCell',
    'SARIMAGRU',
    'TimeSeriesDataset',
    'prepare_data',
    'train_model',
    'evaluate_model',
]
