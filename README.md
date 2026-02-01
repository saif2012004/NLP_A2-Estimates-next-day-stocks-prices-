# 📈 Financial Forecasting Application

A full-stack web application for forecasting financial instruments (stocks and cryptocurrencies) using traditional statistical models (ARIMA), deep learning (LSTM), and ensemble methods. Built with Flask, TensorFlow, and Plotly.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-green.svg)](https://flask.palletsprojects.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

![Demo Screenshot](docs/screenshot.png)

## 🌟 Features

- **Multi-Model Forecasting**: ARIMA, LSTM, and Ensemble predictions
- **Interactive Visualizations**: Candlestick charts with Plotly
- **Multiple Instruments**: Stocks (AAPL, MSFT) and Cryptocurrency (BTC-USD)
- **Flexible Horizons**: 1, 3, 7, and 14-day forecasts
- **Database Integration**: SQLite for data persistence
- **Performance Metrics**: RMSE, MAE, MAPE evaluation
- **Data Collection Module**: Automated financial data scraper

## 🏗️ Architecture

```
├── forecasting_app/          # Main application
│   ├── app.py                # Flask web server
│   ├── models.py             # ARIMA, LSTM, Ensemble models
│   ├── db.py                 # Database management
│   ├── utils.py              # Data utilities
│   └── templates/            # HTML templates
├── data_scraper/             # Data collection module
│   ├── financial_data_scraper.py
│   ├── interactive_scraper.py
│   └── README.md
├── data/                     # CSV datasets
├── tests/                    # Unit tests
└── requirements.txt          # Dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/financial-forecasting.git
cd financial-forecasting
```

2. **Create virtual environment** (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Prepare data**

Place your CSV files in the `data/` directory or use the data scraper:

```bash
cd data_scraper
python financial_data_scraper.py
```

5. **Run the application**

```bash
# Windows
python run.py

# Or directly
python forecasting_app/app.py
```

6. **Initialize database**

- Open browser: http://127.0.0.1:5000
- Click "Initialize Database" button
- Start forecasting!

## 📊 Models

### ARIMA (AutoRegressive Integrated Moving Average)

- **Type**: Traditional statistical model
- **Input**: Univariate time series (Close prices)
- **Order**: (5, 1, 0) - 5 lags, 1st differencing
- **Use Case**: Baseline predictions, linear trends

### LSTM (Long Short-Term Memory)

- **Type**: Deep learning neural network
- **Input**: Multi-feature (Close, MA_5, MA_20, Volatility, Sentiment, Daily_Return)
- **Architecture**: LSTM(50) → Dropout(0.2) → Dense(25) → Output(1)
- **Use Case**: Complex patterns, feature-rich predictions

### Ensemble Model

- **Type**: Hybrid approach
- **Method**: Weighted average (50% ARIMA + 50% LSTM)
- **Use Case**: Robust predictions, production forecasts

## 📈 Usage

### Web Interface

1. **Select Instrument**: Choose AAPL, MSFT, or BTC-USD
2. **Select Horizon**: Choose forecast period (1, 3, 7, or 14 days)
3. **Generate Forecast**: Click to run all three models
4. **View Results**: Interactive candlestick chart with predictions

### Programmatic Usage

```python
from forecasting_app.models import arima_forecast, lstm_forecast, ensemble_forecast
from forecasting_app.utils import load_data

# Load data
data = load_data('data')
df = data['AAPL']

# Generate forecasts
arima_pred = arima_forecast(df, horizon=7)
lstm_pred = lstm_forecast(df, horizon=7)
ensemble_pred = ensemble_forecast(df, horizon=7)

print(f"7-day ARIMA forecast: {arima_pred}")
print(f"7-day LSTM forecast: {lstm_pred}")
print(f"7-day Ensemble forecast: {ensemble_pred}")
```

### Model Evaluation

```python
from forecasting_app.models import evaluate_models

# Evaluate with 80/20 train-test split
results = evaluate_models(df, test_size=0.2, horizon=7)

print(f"ARIMA - RMSE: {results['arima']['rmse']:.2f}, MAPE: {results['arima']['mape']:.2f}%")
print(f"LSTM - RMSE: {results['lstm']['rmse']:.2f}, MAPE: {results['lstm']['mape']:.2f}%")
print(f"Ensemble - RMSE: {results['ensemble']['rmse']:.2f}, MAPE: {results['ensemble']['mape']:.2f}%")
```

## 🧪 Testing

Run unit tests:

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=forecasting_app

# Specific module
pytest tests/test_models.py -v
```

## 📁 Data Collection

The project includes a data scraper module to collect financial data:

```bash
cd data_scraper

# Batch mode (pre-configured symbols)
python financial_data_scraper.py

# Interactive mode
python interactive_scraper.py
```

Features collected:
- Price data (OHLC)
- Volume
- Technical indicators (MA, Volatility, RSI, MACD)
- Sentiment analysis from news
- Daily returns

## 🎨 Visualization

- **Candlestick Charts**: Historical OHLC data
- **Forecast Overlays**: 
  - 🔴 ARIMA: Red dashed line
  - 🟢 LSTM: Teal dotted line
  - 🔵 Ensemble: Blue solid line
- **Interactive**: Zoom, pan, hover tooltips
- **Summary Cards**: Predictions with percentage changes

## 📊 Performance Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **RMSE** | Root Mean Squared Error | Lower is better, penalizes large errors |
| **MAE** | Mean Absolute Error | Lower is better, average error |
| **MAPE** | Mean Absolute Percentage Error | Lower is better, scale-independent (%) |

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Backend** | Flask 3.0+ |
| **Database** | SQLite, SQLAlchemy 2.0+ |
| **ML/Statistical** | Statsmodels (ARIMA) |
| **Deep Learning** | TensorFlow 2.15+, Keras |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Plotly, Matplotlib |
| **Data Collection** | yfinance, BeautifulSoup, TextBlob |
| **Testing** | pytest |

## 📝 Project Structure Details

### forecasting_app/

- `app.py` - Flask application with routes and UI
- `models.py` - Forecasting models (ARIMA, LSTM, Ensemble)
- `db.py` - Database operations with SQLAlchemy
- `utils.py` - Data loading and preprocessing
- `templates/` - HTML templates with Jinja2

### data_scraper/

- `financial_data_scraper.py` - Main scraper class
- `interactive_scraper.py` - CLI interface
- `README.md` - Scraper documentation

### data/

- CSV files with historical price data
- Features: OHLC, Volume, MA, Volatility, Sentiment

## 🔧 Configuration

Environment variables (optional):

```bash
FLASK_ENV=development
FLASK_DEBUG=True
DATABASE_URL=sqlite:///forecasting.db
SECRET_KEY=your-secret-key
```

## 🚀 Deployment

For production deployment:

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

3. **Consider PostgreSQL** for production database
4. **Enable HTTPS** with SSL certificates
5. **Add authentication** if needed

## 📚 Documentation

- [Installation Guide](docs/installation.md)
- [User Guide](docs/user-guide.md)
- [API Reference](docs/api.md)
- [Model Documentation](docs/models.md)
- [Data Scraper Guide](data_scraper/README.md)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Your Name**
- Portfolio: [yourportfolio.com](https://yourportfolio.com)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- GitHub: [@yourusername](https://github.com/yourusername)

## 🙏 Acknowledgments

- Open-source libraries: Flask, TensorFlow, Statsmodels, Plotly
- Financial data sources: Yahoo Finance
- Inspired by modern FinTech applications

## 📞 Support

For support, email your.email@example.com or open an issue on GitHub.

---

**⭐ If you found this project helpful, please consider giving it a star!**
