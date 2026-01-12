"""
SARIMA-GRU Model Architecture

This module contains the core model architecture combining SARIMA (Seasonal AutoRegressive 
Integrated Moving Average) components with GRU (Gated Recurrent Unit) neural networks.
"""

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("Using CPU")


class SARIMAGRUCell(nn.Module):
    """
    SARIMA-GRU Cell combining seasonal ARIMA components with GRU architecture.
    
    Args:
        input_size (int): Size of input features
        hidden_size (int): Size of hidden state
        seasonal_period (int): Seasonal period for SARIMA (default: 24)
    """
    
    def __init__(self, input_size, hidden_size, seasonal_period=24):
        super(SARIMAGRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seasonal_period = seasonal_period
        
        # GRU gates
        self.reset_gate = nn.Linear(input_size + hidden_size + hidden_size + 1, hidden_size)
        self.update_gate = nn.Linear(input_size + hidden_size + hidden_size + 1, hidden_size)
        
        # Seasonal components
        self.seasonal_gate = nn.Linear(input_size + hidden_size + hidden_size, hidden_size)
        self.trend_gate = nn.Linear(2, 1)
        
        # Transformation layers
        self.hidden_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.seasonal_transform = nn.Linear(input_size + hidden_size, hidden_size)
        self.trend_transform = nn.Linear(2, 1)
        
        # ARIMA parameters
        self.ar_order = 3
        self.ma_order = 3
        self.ar_weights = nn.Parameter(torch.randn(self.ar_order) * 0.1)
        self.ma_weights = nn.Parameter(torch.randn(self.ma_order) * 0.1)
        
        # Component weights
        self.seasonal_weight = nn.Linear(hidden_size, hidden_size)
        self.trend_weight = nn.Linear(1, hidden_size)
        self.ar_weight = nn.Linear(1, hidden_size)
        self.ma_weight = nn.Linear(1, hidden_size)
        
        self.activation = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x_t, hidden_state, seasonal_memory, trend_state, ar_buffer, ma_buffer, prev_error):
        """
        Forward pass through SARIMA-GRU cell.
        
        Args:
            x_t: Current input tensor
            hidden_state: Previous hidden state
            seasonal_memory: Historical seasonal components
            trend_state: Current trend state
            ar_buffer: AutoRegressive history buffer
            ma_buffer: Moving Average history buffer
            prev_error: Previous prediction error
            
        Returns:
            Tuple of (new_hidden_state, updated_states)
        """
        batch_size = x_t.size(0)
        
        # Extract seasonal component
        if seasonal_memory.size(1) >= self.seasonal_period:
            seasonal_component = seasonal_memory[:, -self.seasonal_period, :]
        else:
            seasonal_component = torch.zeros(batch_size, self.hidden_size, device=x_t.device)
        
        # GRU gates
        gate_input = torch.cat([x_t, hidden_state, seasonal_component, trend_state], dim=1)
        reset_gate = self.sigmoid(self.reset_gate(gate_input))
        update_gate = self.sigmoid(self.update_gate(gate_input))
        
        seasonal_gate_input = torch.cat([x_t, hidden_state, seasonal_component], dim=1)
        seasonal_gate = self.sigmoid(self.seasonal_gate(seasonal_gate_input))
        
        # Trend processing
        trend_input = x_t[:, 0:1]
        trend_gate_input = torch.cat([trend_input, trend_state], dim=1)
        trend_gate = self.sigmoid(self.trend_gate(trend_gate_input))
        
        new_trend_input = torch.cat([trend_input, trend_state], dim=1)
        new_trend = self.trend_transform(new_trend_input)
        trend_state = (1 - trend_gate) * trend_state + trend_gate * new_trend
        
        # Seasonal processing
        seasonal_input = torch.cat([x_t, reset_gate * seasonal_component], dim=1)
        new_seasonal = self.activation(self.seasonal_transform(seasonal_input))
        updated_seasonal = (1 - seasonal_gate) * seasonal_component + seasonal_gate * new_seasonal
        
        seasonal_memory = torch.cat([seasonal_memory, updated_seasonal.unsqueeze(1)], dim=1)
        if seasonal_memory.size(1) > self.seasonal_period * 2:
            seasonal_memory = seasonal_memory[:, -self.seasonal_period * 2:, :]
        
        # AR component
        ar_component = torch.zeros(batch_size, 1, device=x_t.device)
        if ar_buffer.size(1) >= self.ar_order:
            for i in range(self.ar_order):
                if ar_buffer.size(1) > i:
                    past_hidden = ar_buffer[:, -(i+1), :].mean(dim=1, keepdim=True)
                    ar_component += self.ar_weights[i] * past_hidden
        
        # MA component
        ma_component = torch.zeros(batch_size, 1, device=x_t.device)
        if ma_buffer.size(1) >= self.ma_order:
            for i in range(self.ma_order):
                if ma_buffer.size(1) > i:
                    past_error = ma_buffer[:, -(i+1), :]
                    ma_component += self.ma_weights[i] * past_error
        
        # Update hidden state
        hidden_input = torch.cat([x_t, reset_gate * hidden_state], dim=1)
        new_hidden_base = self.activation(self.hidden_transform(hidden_input))
        
        seasonal_contribution = self.seasonal_weight(updated_seasonal)
        trend_contribution = self.trend_weight(trend_state)
        ar_contribution = self.ar_weight(ar_component)
        ma_contribution = self.ma_weight(ma_component)
        
        new_hidden = new_hidden_base + seasonal_contribution + trend_contribution + ar_contribution + ma_contribution
        hidden_state = (1 - update_gate) * hidden_state + update_gate * new_hidden
        
        # Update buffers
        ar_buffer = torch.cat([ar_buffer, hidden_state.unsqueeze(1)], dim=1)
        if ar_buffer.size(1) > self.ar_order * 2:
            ar_buffer = ar_buffer[:, -self.ar_order * 2:, :]
        
        error = torch.zeros(batch_size, 1, device=x_t.device)
        ma_buffer = torch.cat([ma_buffer, error.unsqueeze(1)], dim=1)
        if ma_buffer.size(1) > self.ma_order * 2:
            ma_buffer = ma_buffer[:, -self.ma_order * 2:, :]
        
        return hidden_state, (seasonal_memory, trend_state, ar_buffer, ma_buffer)


class SARIMAGRU(nn.Module):
    """
    Full SARIMA-GRU model with multiple layers for time series forecasting.
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Hidden state dimension
        output_size (int): Output dimension (default: 1 for single target)
        num_layers (int): Number of stacked SARIMA-GRU layers
        seasonal_period (int): Seasonal period (default: 24)
    """
    
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2, seasonal_period=24):
        super(SARIMAGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.seasonal_period = seasonal_period
        
        # Create multiple SARIMA-GRU cells
        self.cells = nn.ModuleList([
            SARIMAGRUCell(input_size if i == 0 else hidden_size, hidden_size, seasonal_period)
            for i in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden_states=None):
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            hidden_states: Optional initial hidden states
            
        Returns:
            Tuple of (outputs, final_states)
        """
        batch_size, seq_len, _ = x.size()
        
        # Initialize hidden states
        if hidden_states is None:
            hidden_states = []
            for i in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                seasonal_mem = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
                trend_st = torch.zeros(batch_size, 1, device=x.device)
                ar_buf = torch.zeros(batch_size, 1, self.hidden_size, device=x.device)
                ma_buf = torch.zeros(batch_size, 1, 1, device=x.device)
                hidden_states.append((h, (seasonal_mem, trend_st, ar_buf, ma_buf)))
        
        outputs = []
        
        # Process sequence
        for t in range(seq_len):
            x_t = x[:, t, :]
            new_hidden_states = []
            
            for layer_idx, cell in enumerate(self.cells):
                h, (seasonal_mem, trend_st, ar_buf, ma_buf) = hidden_states[layer_idx]
                prev_error = torch.zeros(batch_size, 1, device=x.device)
                
                h_new, (seasonal_mem_new, trend_st_new, ar_buf_new, ma_buf_new) = cell(
                    x_t, h, seasonal_mem, trend_st, ar_buf, ma_buf, prev_error
                )
                
                new_hidden_states.append((h_new, (seasonal_mem_new, trend_st_new, ar_buf_new, ma_buf_new)))
                x_t = h_new
            
            hidden_states = new_hidden_states
            outputs.append(x_t)
        
        # Stack outputs and apply output layer
        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, hidden_size)
        predictions = self.fc(outputs)  # (batch_size, seq_len, output_size)
        
        return predictions, hidden_states
