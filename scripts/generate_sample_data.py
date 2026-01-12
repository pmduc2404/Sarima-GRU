"""
Sample Data Generator for Testing SARIMA-GRU Model
Generates synthetic time series data similar to lake water level data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sample_data(output_path="DataSet/sample_data.csv", n_samples=2000):
    """
    Generate synthetic time series data with seasonality and trend
    
    Args:
        output_path: Path to save the CSV file
        n_samples: Number of data points to generate
    """
    # Time index
    start_date = datetime(2019, 1, 1)
    times = [start_date + timedelta(hours=i) for i in range(n_samples)]
    
    # Base signal with trend
    t = np.arange(n_samples)
    trend = 15 + 0.005 * t + 0.00001 * (t ** 2)
    
    # Daily seasonality (24-hour cycle)
    daily_season = 2 * np.sin(2 * np.pi * t / 24)
    
    # Weekly seasonality
    weekly_season = 0.5 * np.sin(2 * np.pi * t / (24 * 7))
    
    # Noise
    noise = np.random.normal(0, 0.3, n_samples)
    
    # Lake water level
    lake_level = trend + daily_season + weekly_season + noise
    
    # Additional features (optional)
    temperature = 20 + 10 * np.sin(2 * np.pi * t / (24 * 365)) + np.random.normal(0, 2, n_samples)
    precipitation = np.maximum(0, 2 * np.sin(2 * np.pi * t / (24 * 30)) + np.random.normal(0, 1, n_samples))
    humidity = 60 + 20 * np.sin(2 * np.pi * t / 24) + np.random.normal(0, 5, n_samples)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Time': times,
        'Lake water level (m)': lake_level,
        'Temperature (C)': temperature,
        'Precipitation (mm)': precipitation,
        'Humidity (%)': humidity
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated sample data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nData statistics:")
    print(df.describe())
    
    return df

if __name__ == "__main__":
    generate_sample_data("DataSet/sample_data.csv", n_samples=2000)
