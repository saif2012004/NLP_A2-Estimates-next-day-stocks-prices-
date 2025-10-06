# CS4063 Financial Forecasting Application

A Flask-based web application for forecasting financial instruments (AAPL, MSFT, BTC-USD) using traditional (ARIMA) and deep learning (LSTM) models with ensemble predictions.

## 📋 Features

- **Multi-Model Forecasting**: ARIMA (traditional), LSTM (neural network), and Ensemble predictions
- **Interactive Visualization**: Candlestick charts with Plotly for OHLC data and forecast overlays
- **Multiple Instruments**: AAPL (Apple), MSFT (Microsoft), BTC-USD (Bitcoin)
- **Flexible Horizons**: 1, 3, 7, and 14-day forecasts
- **Database Integration**: SQLite for historical data and predictions storage
- **Performance Metrics**: RMSE, MAE, MAPE evaluation

## 🏗️ Project Structure

```
forecasting_app/
├── app.py                 # Flask application (main entry point)
├── models.py              # ARIMA, LSTM, Ensemble forecasting models
├── db.py                  # SQLite database management (SQLAlchemy)
├── utils.py               # Data loading and preprocessing utilities
├── templates/             # HTML templates
│   ├── index.html         # Main dashboard
│   ├── stats.html         # Database statistics
│   ├── 404.html           # Error page
│   └── 500.html           # Server error page
├── __init__.py
data/                      # CSV datasets directory
├── AAPL_20250915_185850.csv
├── MSFT_20250915_185853.csv
└── BTC-USD_20250915_185857.csv
tests/                     # Unit tests
docs/                      # Documentation
requirements.txt           # Python dependencies
forecasting.db            # SQLite database (created on init)
```

## 🚀 Quick Start Guide

### 1. Prerequisites

- Python 3.13 (or 3.10+)
- pip package manager
- CSV data files (place in `data/` directory)

### 2. Installation

```bash
# Navigate to project directory
cd "E:\7 semester\NLP\A2"

# Install dependencies
pip install -r requirements.txt
```

### 3. Prepare Data

**Copy your CSV files to the `data/` directory:**

```
E:\7 semester\NLP\A2\data\
├── AAPL_20250915_185850.csv
├── MSFT_20250915_185853.csv
└── BTC-USD_20250915_185857.csv
```

### 4. Initialize Database

**Option A: Via Web Interface**

1. Start the app (see step 5)
2. Click "Initialize Database" button on the home page

**Option B: Via Command Line**

```bash
python -c "from forecasting_app.db import init_db; init_db('data')"
```

### 5. Run Application

```bash
# Start Flask development server
python forecasting_app/app.py
```

**Access the application:**

- Open browser: http://127.0.0.1:5000
- Main dashboard: `/`
- Statistics: `/stats`
- Initialize DB: `/init-db`

### 6. Using the Application

1. **Select Instrument**: Choose from AAPL, MSFT, or BTC-USD
2. **Select Horizon**: Choose forecast period (1, 3, 7, or 14 days)
3. **Generate Forecast**: Click the button to run all three models
4. **View Results**:
   - Interactive candlestick chart with historical OHLC data
   - Forecast lines overlay (ARIMA, LSTM, Ensemble)
   - Prediction summary cards with percentage changes
   - Zoom, pan, and hover for detailed inspection

## 📊 Models Explanation

### ARIMA (AutoRegressive Integrated Moving Average)

- **Type**: Traditional statistical model
- **Input**: Univariate time series (Close prices only)
- **Order**: (5, 1, 0)
  - p=5: Uses last 5 days (autoregressive)
  - d=1: First-order differencing (stationary)
  - q=0: No moving average component
- **Strengths**: Good for linear trends, fast computation
- **Use Case**: Short-term predictions, baseline model

### LSTM (Long Short-Term Memory)

- **Type**: Deep learning neural network
- **Input**: Multi-feature (Close, MA_5, MA_20, Volatility, Avg_Sentiment, Daily_Return)
- **Architecture**:
  - LSTM Layer (50 units)
  - Dropout (0.2)
  - Dense Layer (25 units)
  - Output (1 unit)
- **Strengths**: Captures non-linear patterns, leverages multiple features
- **Use Case**: Complex market conditions, feature-rich predictions

### Ensemble Model

- **Type**: Hybrid approach
- **Method**: Weighted average of ARIMA and LSTM
- **Default Weights**: 50% ARIMA + 50% LSTM
- **Strengths**: Reduces variance, improves robustness
- **Use Case**: Production forecasts, balanced predictions

## 📈 Performance Metrics

The system evaluates models using:

1. **RMSE (Root Mean Squared Error)**

   - Formula: √(Σ(predicted - actual)² / n)
   - Lower is better
   - Penalizes large errors

2. **MAE (Mean Absolute Error)**

   - Formula: Σ|predicted - actual| / n
   - Lower is better
   - Robust to outliers

3. **MAPE (Mean Absolute Percentage Error)**
   - Formula: (Σ|predicted - actual| / |actual|) / n × 100%
   - Lower is better
   - Scale-independent (percentage)

## 🧪 Testing

### Run Unit Tests

```bash
# Test all modules
pytest forecasting_app/ -v

# Test specific modules
pytest forecasting_app/models.py -v
pytest forecasting_app/db.py -v

# Run with coverage
pytest forecasting_app/ -v --cov=forecasting_app
```

### Manual Testing

```bash
# Test data loading
python forecasting_app/utils.py

# Test database operations
python forecasting_app/db.py

# Test forecasting models
python forecasting_app/models.py
```

## 🎨 Visualization Features

### Candlestick Chart

- **Historical OHLC**: Last 90 days of price action
- **Green/Red Candles**: Up/down days visualization
- **Interactive**: Zoom, pan, hover tooltips

### Forecast Overlays

- **ARIMA**: Red dashed line with circle markers
- **LSTM**: Teal dotted line with square markers
- **Ensemble**: Blue solid line with diamond markers
- **Vertical Separator**: Gray line marking forecast start
- **Future Dates**: Extended x-axis for predictions

### Summary Cards

- **Current Price**: Last known Close price
- **Model Predictions**: Final forecast values
- **Change Indicators**: Percentage change with up/down arrows
- **Color Coding**: Green (positive), Red (negative)

## 📁 Database Schema

### Historical Table

- **Columns**: instrument, date, open, high, low, close, volume
- **Features**: MA_5, MA_20, Volatility, Avg_Sentiment, Daily_Return, RSI, MACD
- **Indexes**: (instrument, date) composite primary key
- **Purpose**: Store curated CSV data for training

### Predictions Table

- **Columns**: instrument, prediction_date, target_date, model_type, predicted_close, horizon
- **Confidence**: lower_bound, upper_bound (optional)
- **Indexes**: (instrument, model_type, horizon)
- **Purpose**: Store and retrieve model forecasts

## 🔧 Troubleshooting

### Issue: Database not initialized

**Solution**: Visit `/init-db` route or run:

```bash
python -c "from forecasting_app.db import init_db; init_db('data')"
```

### Issue: CSV files not found

**Solution**: Ensure CSV files are in `data/` directory with exact filenames:

- `AAPL_20250915_185850.csv`
- `MSFT_20250915_185853.csv`
- `BBTC-USD_20250915_185857.csv`

### Issue: TensorFlow/LSTM errors

**Solution**: Reduce epochs or look_back:

```python
lstm_forecast(df, horizon=7, look_back=30, epochs=20)
```

### Issue: ARIMA convergence warnings

**Solution**: Try different order parameters:

```python
arima_forecast(df, horizon=7, order=(3, 1, 0))
```

## 📝 Assignment Deliverables Checklist

- ✅ **Front-end**: Flask web interface with instrument/horizon selection
- ✅ **Back-end DB**: SQLite with historical data and predictions
- ✅ **Traditional Model**: ARIMA implementation with statsmodels
- ✅ **Neural Model**: LSTM with TensorFlow/Keras and multi-features
- ✅ **Ensemble Model**: Combined ARIMA + LSTM predictions
- ✅ **Visualization**: Candlestick charts with forecast overlays (Plotly)
- ✅ **Metrics**: RMSE, MAE, MAPE evaluation
- ✅ **Software Engineering**: Git, modularity, documentation, tests
- ✅ **Open-Source**: No paid APIs, all free libraries
- ✅ **Curated Datasets**: CSV files as data source

## 🔬 Model Evaluation

To evaluate models on test data:

```python
from forecasting_app.models import evaluate_models
from forecasting_app.utils import load_data

# Load data
data = load_data('data')
df = data['AAPL']

# Evaluate with 80/20 split
results = evaluate_models(df, test_size=0.2, horizon=7)

print(f"ARIMA RMSE: {results['arima']['rmse']:.2f}")
print(f"LSTM MAPE: {results['lstm']['mape']:.2f}%")
print(f"Ensemble MAE: {results['ensemble']['mae']:.2f}")
```

## 📚 Dependencies

- **Flask 3.0+**: Web framework
- **Pandas 2.2+**: Data manipulation
- **NumPy 1.26+**: Numerical computing
- **SQLAlchemy 2.0+**: ORM for database
- **Statsmodels 0.14+**: ARIMA models
- **TensorFlow 2.15+**: LSTM neural networks
- **Plotly 5.18+**: Interactive visualizations
- **Scikit-learn 1.3+**: Preprocessing and metrics

## 🌐 Production Deployment

For production use:

1. **Use production WSGI server**:

   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 forecasting_app.app:app
   ```

2. **Set environment variables**:

   ```bash
   export FLASK_ENV=production
   export SECRET_KEY=your-secure-secret-key
   ```

3. **Configure proper database**: Consider PostgreSQL for production
4. **Enable HTTPS**: Use SSL certificates
5. **Add authentication**: Implement user login if needed

## 📄 License

MIT License - Educational Project for CS4063

## 👨‍💻 Author

CS4063 Student - Financial Forecasting Assignment

## 🙏 Acknowledgments

- Open-source libraries: Flask, TensorFlow, Statsmodels, Plotly
- Dataset sources: Historical OHLC data with engineered features
- Assignment specifications: CS4063 course requirements
