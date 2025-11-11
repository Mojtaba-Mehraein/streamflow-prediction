# Hydrological Time Series Prediction using LSTM with Grey Wolf Optimization

A Python implementation for predicting hydrological discharge using LSTM neural networks optimized with Grey Wolf Optimizer (GWO).

## 📋 Overview

This project combines Long Short-Term Memory (LSTM) neural networks with Grey Wolf Optimization to predict hydrological discharge values from time series data. The GWO algorithm automatically tunes hyperparameters of the LSTM model for optimal performance.

## 🚀 Features

- **LSTM Neural Network**: For time series prediction of hydrological data
- **Grey Wolf Optimization**: Metaheuristic algorithm for hyperparameter tuning
- **Data Preprocessing**: Automatic normalization and sequence creation
- **Comprehensive Evaluation**: R² score and RMSE metrics with visualization
- **Modular Architecture**: Easy to modify and extend

## 📁 Project Structure

```
├── hydrological_prediction.py    # Main implementation file
├── hydrological_data.xlsx        # Input data file (required)
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 📊 Model Architecture

- **Input**: 11 hydrological features
- **LSTM Layers**: Two-layer LSTM architecture with dropout
- **Output**: Single value prediction (discharge)
- **Optimization**: GWO optimizes:
  - Number of LSTM units
  - Dropout rate
  - Number of epochs
  - Batch size

## 🛠 Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Dependencies

```bash
pip install numpy pandas scikit-learn tensorflow keras matplotlib pygwo openpyxl
```

Or create a `requirements.txt` file:

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
matplotlib>=3.5.0
pygwo>=0.1.0
openpyxl>=3.0.0
```

Install using:
```bash
pip install -r requirements.txt
```

## 📈 Usage

### 1. Prepare Your Data

- Place your hydrological data in an Excel file named `hydrological_data.xlsx`
- Format: First 11 columns as features, last column as target (discharge)
- Ensure no missing values in the dataset

### 2. Run the Model

```bash
python hydrological_prediction.py
```

### 3. Customize Parameters

You can modify these parameters in the code:

```python
# Time steps for sequence creation
time_steps = 3

# GWO optimization bounds
bounds = [
    {'name': 'units', 'type': 'discrete', 'domain': (30, 50, 100, 150)},
    {'name': 'dropout', 'type': 'continuous', 'domain': (0.1, 0.3)},
    {'name': 'epochs', 'type': 'discrete', 'domain': (30, 50, 100)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (16, 32, 64)}
]

# GWO algorithm parameters
num_wolves = 10
max_iter = 20
```

## ⚙️ Configuration

### Data Preprocessing
- MinMax scaling (0-1 normalization)
- 70-30 train-test split
- Sequence creation with configurable time steps

### Model Parameters Optimized by GWO
- **LSTM Units**: [30, 50, 100, 150]
- **Dropout Rate**: [0.1, 0.3]
- **Training Epochs**: [30, 50, 100]
- **Batch Size**: [16, 32, 64]

## 📊 Output

The script provides:
- **Console Output**: R² score and RMSE values
- **Visualization**: Comparison plot of actual vs predicted discharge
- **Optimization Results**: Best parameters found by GWO

Example output:
```
R2 Score: 0.8956
RMSE: 15.4321
```

## 🎯 Performance Metrics

- **R² Score**: Coefficient of determination
- **RMSE**: Root Mean Square Error
- **Visual Analysis**: Time series comparison plot

## 🔧 Customization

### Modify LSTM Architecture
```python
def create_lstm_model(units=50, dropout=0.2):
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, 
                  input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))
    model.add(LSTM(units=units, dropout=dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model
```

### Adjust GWO Parameters
```python
gwo_result = grey_wolf_optimizer(objective_function, bounds, 
                                num_wolves=15, max_iter=30, minimize=True)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:

- Bug fixes
- New features
- Performance improvements
- Documentation enhancements

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Keras/TensorFlow**: For LSTM implementation
- **scikit-learn**: For data preprocessing and metrics
- **pygwo**: For Grey Wolf Optimizer implementation
- **Matplotlib**: For visualization

## 📞 Contact

For questions or support, please open an issue in the repository.

---

**Note**: Make sure to place your `hydrological_data.xlsx` file in the same directory as the script before running.
