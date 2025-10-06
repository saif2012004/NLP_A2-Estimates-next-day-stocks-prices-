# 🚀 Quick Start Guide - CS4063 Forecasting App

## ✅ What's Been Created

Your complete forecasting application is ready! Here's what you have:

### 📁 Core Modules

- ✅ **forecasting_app/utils.py** - Data loading from CSV files
- ✅ **forecasting_app/db.py** - SQLite database with SQLAlchemy ORM
- ✅ **forecasting_app/models.py** - ARIMA, LSTM, Ensemble forecasting
- ✅ **forecasting_app/app.py** - Flask web application

### 🎨 Web Interface

- ✅ **templates/index.html** - Main dashboard with form and chart
- ✅ **templates/stats.html** - Database statistics page
- ✅ **templates/404.html** - Error page
- ✅ **templates/500.html** - Server error page

### 📝 Documentation

- ✅ **README.md** - Complete documentation
- ✅ **QUICKSTART.md** - This file
- ✅ **requirements.txt** - All dependencies (installed ✓)

### 🔧 Helper Scripts

- ✅ **run_app.bat** - Windows quick start
- ✅ **run_app.sh** - Linux/Mac quick start
- ✅ **setup.bat/sh** - Project initialization

## 🎯 Next Steps (3 Simple Steps!)

### Step 1: Copy CSV Files ⚠️ IMPORTANT

Copy your 3 CSV files to the `data/` directory:

```
E:\7 semester\NLP\A2\data\
├── AAPL_20250915_185850.csv
├── MSFT_20250915_185853.csv
└── BTC-USD_20250915_185857.csv
```

**Quick Command (if CSVs are in current directory):**

```powershell
copy *.csv data\
```

### Step 2: Run the Application

**Windows:**

```powershell
run_app.bat
```

**Or manually:**

```powershell
python forecasting_app\app.py
```

### Step 3: Initialize Database

**Option A - Via Browser (Recommended):**

1. Open browser: http://127.0.0.1:5000
2. Click "🔄 Initialize Database" button

**Option B - Via Command:**

```powershell
python -c "from forecasting_app.db import init_db; init_db('data')"
```

## 🎨 Using the Application

### Main Dashboard (/)

1. **Select Instrument**: AAPL, MSFT, or BTC-USD
2. **Select Horizon**: 1, 3, 7, or 14 days
3. **Click "Generate Forecast"**
4. **View Results**:
   - Interactive candlestick chart
   - ARIMA predictions (red dashed line)
   - LSTM predictions (teal dotted line)
   - Ensemble predictions (blue solid line)
   - Summary cards with price changes

### Features

- 📊 **Interactive Charts**: Zoom, pan, hover for details
- 📈 **Multiple Models**: Compare ARIMA, LSTM, Ensemble
- 💾 **Database Storage**: All predictions saved
- 📉 **Historical Context**: Last 90 days of OHLC data
- 🎯 **Accuracy Metrics**: Percentage changes displayed

## 🧪 Testing

### Test Individual Modules

```powershell
# Test data loading
python forecasting_app\utils.py

# Test database
python forecasting_app\db.py

# Test models
python forecasting_app\models.py
```

### Run Unit Tests

```powershell
pytest forecasting_app\ -v
```

## 📊 Model Evaluation

To evaluate models with train/test split:

```python
from forecasting_app.models import evaluate_models
from forecasting_app.utils import load_data

data = load_data('data')
results = evaluate_models(data['AAPL'], test_size=0.2, horizon=7)

print(f"ARIMA RMSE: {results['arima']['rmse']:.2f}")
print(f"LSTM MAPE: {results['lstm']['mape']:.2f}%")
print(f"Ensemble MAE: {results['ensemble']['mae']:.2f}")
```

## 🎓 Assignment Checklist

### ✅ Completed

- [x] Front-end with instrument/horizon selection
- [x] Back-end SQLite database
- [x] ARIMA traditional model
- [x] LSTM neural network with features
- [x] Ensemble model
- [x] Candlestick visualization with overlays
- [x] RMSE, MAE, MAPE metrics
- [x] Git repository
- [x] Modular code structure
- [x] Documentation and docstrings
- [x] Unit tests (pytest)
- [x] Open-source libraries only
- [x] Curated CSV datasets

### 📝 Still Needed for Submission

- [ ] Copy CSV files to `data/` directory
- [ ] Run application and test all features
- [ ] Generate screenshots of:
  - Main dashboard with forecasts
  - Candlestick chart with overlays
  - Different instruments/horizons
  - Model evaluation results
- [ ] Write report with:
  - System architecture diagram
  - Model explanations (ARIMA, LSTM, Ensemble)
  - Performance metrics comparison
  - Screenshots of running app
  - Justification of design choices

## 🐛 Troubleshooting

### Problem: "No historical data found"

**Solution**: Initialize database via `/init-db` route

### Problem: "CSV file not found"

**Solution**: Ensure CSV files are in `data/` with exact filenames

### Problem: LSTM takes too long

**Solution**: Reduce epochs in models.py or use smaller look_back window

### Problem: Port 5000 already in use

**Solution**: Change port in app.py:

```python
app.run(debug=True, host='127.0.0.1', port=5001)
```

## 📸 Screenshots for Report

When running, capture these screenshots:

1. **Main Dashboard** - Form with instrument/horizon selection
2. **AAPL Forecast** - 7-day prediction with all three models
3. **MSFT Forecast** - Different horizon (e.g., 14 days)
4. **BTC-USD Forecast** - Crypto instrument
5. **Model Comparison** - Summary cards showing predictions
6. **Interactive Chart** - Hover tooltip showing details
7. **Database Stats** - `/stats` page
8. **Terminal Output** - Model evaluation results

## 🚀 Final Steps for Assignment

1. ✅ **Copy CSV files** to `data/` directory
2. ✅ **Run application**: `run_app.bat`
3. ✅ **Initialize database**: Click button or visit `/init-db`
4. ✅ **Test all instruments**: AAPL, MSFT, BTC-USD
5. ✅ **Test all horizons**: 1, 3, 7, 14 days
6. ✅ **Capture screenshots**: As listed above
7. ✅ **Run model evaluation**: See code above
8. ✅ **Write report**: Include screenshots and metrics
9. ✅ **Git commit**: `git add . && git commit -m "Complete forecasting app"`
10. ✅ **Submit**: Code + Report + Screenshots

## 📚 Key Files Summary

| File               | Purpose                           |
| ------------------ | --------------------------------- |
| `app.py`           | Flask web server and routes       |
| `models.py`        | ARIMA, LSTM, Ensemble forecasting |
| `db.py`            | Database operations (SQLAlchemy)  |
| `utils.py`         | CSV data loading                  |
| `index.html`       | Main web interface                |
| `requirements.txt` | Python dependencies               |
| `forecasting.db`   | SQLite database (auto-created)    |

## 🎉 Success Indicators

You'll know everything works when:

1. ✅ App runs at http://127.0.0.1:5000
2. ✅ Database shows >0 historical records
3. ✅ Forecasts generate for all instruments
4. ✅ Charts display with colored forecast lines
5. ✅ Summary cards show predictions
6. ✅ All three models complete without errors

## 💡 Tips

- **Start with AAPL**: Easiest to test first
- **Try 7-day horizon**: Good balance of accuracy
- **Compare models**: See which performs best
- **Use ensemble**: Usually most robust
- **Save screenshots**: For your report
- **Document everything**: In your assignment report

## 📞 Need Help?

Check the detailed README.md for:

- Complete API documentation
- Model architecture details
- Database schema
- Troubleshooting guide
- Production deployment tips

---

**Good luck with your CS4063 assignment! 🎓**
