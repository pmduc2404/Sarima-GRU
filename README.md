# SARIMA-GRU: Time Series Forecasting

Hybrid deep learning model combining SARIMA + GRU for time series forecasting, optimized for water level prediction.

## 🚀 Quick Start

### Install
```bash
pip install -r requirements.txt
```

### Train
```bash
python3 scripts/train.py --num_epochs 100 --batch_size 32
```

### Test
```bash
python3 scripts/test.py --model_path sarima_gru_model.pth --data_path DataSet/AnKhe.csv
```

### Evaluate
```bash
python3 scripts/evaluate.py --model_path sarima_gru_model.pth --data_path DataSet/AnKhe.csv
```

## 📁 Structure

```
├── src/sarima_gru/          # Core model
│   ├── model.py
│   ├── data.py
│   └── trainer.py
├── scripts/                 # Scripts
│   ├── train.py
│   ├── test.py
│   └── evaluate.py
├── DataSet/                 # Data
├── models/                  # Saved models
└── results/                 # Results
```

## 💻 Usage

### Training
```python
from sarima_gru import SARIMAGRU, prepare_data, train_model
from torch.utils.data import DataLoader

train_data, test_data, scalers = prepare_data('DataSet/AnKhe.csv')
model = SARIMAGRU(input_size=10, hidden_size=64, num_layers=3)

train_loader = DataLoader(train_data, batch_size=32)
test_loader = DataLoader(test_data, batch_size=32)
results = train_model(model, train_loader, test_loader, num_epochs=100)
```

### Prediction
```python
import torch
checkpoint = torch.load('sarima_gru_model.pth')
model = SARIMAGRU(**checkpoint['model_config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    output, _ = model(input_data)
```

## 📊 Data Format

CSV: `Time, Target, Feature1, Feature2, ...`

```csv
Time,Lake water level (m),Outflow,Inflow
2019-01-01 00:00:00,15.2,25.3,10.5
```

## 🔧 Parameters

| Parameter | Default |
|-----------|---------|
| `--num_epochs` | 100 |
| `--batch_size` | 64 |
| `--learning_rate` | 0.001 |
| `--hidden_size` | 32 |
| `--num_layers` | 2 |

## 🎯 Commands

```bash
# Quick test
python3 scripts/train.py --num_epochs 10 --no_plot

# Standard
python3 scripts/train.py --num_epochs 100

# Full training
python3 scripts/train.py --num_epochs 500 --hidden_size 128

# Help
python3 scripts/train.py --help
```
---
**v1.0.0** | January 2026
