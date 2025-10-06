#!/bin/bash
# CS4063 Forecasting Application - Project Setup Script
# This script creates the project directory structure and initializes git repository

echo "Setting up CS4063 Forecasting Application..."

# Create main project directory structure
mkdir -p forecasting_app
mkdir -p forecasting_app/templates
mkdir -p forecasting_app/static
mkdir -p forecasting_app/static/css
mkdir -p forecasting_app/static/js
mkdir -p tests
mkdir -p data
mkdir -p docs
mkdir -p notebooks

echo "✓ Created directory structure"

# Copy CSV files to data directory (assumes CSVs are in current directory)
# Uncomment and modify paths as needed:
# cp BTC-USD_20250915_185857.csv data/
# cp MSFT_20250915_185853.csv data/
# cp AAPL_20250915_185850.csv data/

echo "Note: Please copy CSV files to data/ directory:"
echo "  - BTC-USD_20250915_185857.csv"
echo "  - MSFT_20250915_185853.csv"
echo "  - AAPL_20250915_185850.csv"

# Initialize git repository
git init
echo "✓ Initialized git repository"

# Create .gitignore
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Database
*.db
*.sqlite
*.sqlite3

# Logs
*.log

# OS
.DS_Store
Thumbs.db

# Jupyter Notebooks
.ipynb_checkpoints/

# Environment variables
.env

# Model artifacts
models/*.h5
models/*.pkl
models/*.joblib

# Coverage reports
htmlcov/
.coverage
.coverage.*
coverage.xml
*.cover

# pytest
.pytest_cache/
EOF

echo "✓ Created .gitignore"

# Create empty __init__.py files
touch forecasting_app/__init__.py
touch tests/__init__.py

echo "✓ Created __init__.py files"

# Create README
cat > README.md << 'EOF'
# CS4063 Financial Forecasting Application

A Flask-based web application for forecasting financial instruments (AAPL, MSFT, BTC-USD) using traditional (ARIMA) and deep learning (LSTM) models.

## Project Structure

```
forecasting_app/
├── app.py              # Flask application entry point
├── models.py           # ARIMA, LSTM, and ensemble model implementations
├── db.py              # SQLite database management with SQLAlchemy
├── utils.py           # Data loading and preprocessing utilities
├── templates/         # HTML templates for Flask
├── static/           # CSS, JavaScript, and static assets
│   ├── css/
│   └── js/
tests/                # Unit and integration tests
data/                 # Curated CSV datasets
docs/                 # Documentation and report
notebooks/            # Jupyter notebooks for exploration
requirements.txt      # Python dependencies
```

## Setup

1. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure CSV datasets are in `data/` directory:
   - AAPL_20250915_185850.csv
   - MSFT_20250915_185853.csv
   - BTC-USD_20250915_185857.csv

4. Run the application:
   ```bash
   cd forecasting_app
   python app.py
   ```

## Features

- **Instruments**: AAPL, MSFT, BTC-USD
- **Forecast Horizons**: 1, 3, 7, 14 days
- **Models**: ARIMA (traditional), LSTM (neural), Ensemble
- **Visualization**: Interactive candlestick charts with Plotly
- **Database**: SQLite for historical data and predictions
- **Metrics**: RMSE, MAE, MAPE

## Assignment Requirements

This project fulfills CS4063 assignment requirements:
- ✓ Front-end (Flask) with instrument and horizon selection
- ✓ Back-end DB (SQLite) for historical data and predictions
- ✓ Traditional model (ARIMA)
- ✓ Neural network model (LSTM with features)
- ✓ Ensemble model
- ✓ Candlestick visualization with overlays
- ✓ Software engineering practices (Git, modularity, tests)
- ✓ Open-source only (no paid APIs)
- ✓ Curated datasets (CSV files)

## Testing

```bash
pytest tests/ -v --cov=forecasting_app
```

## License

MIT License - Educational Project for CS4063
EOF

echo "✓ Created README.md"

# Make initial git commit
git add .
git commit -m "Initial project setup for CS4063 Forecasting Application

- Created project structure with forecasting_app/, tests/, data/ directories
- Added requirements.txt with all open-source dependencies
- Added utils.py with data loading functionality for curated CSV datasets
- Initialized git repository with .gitignore
- Added README.md with project documentation
"

echo "✓ Made initial git commit"

echo ""
echo "=========================================="
echo "Setup complete! Next steps:"
echo "=========================================="
echo "1. Copy CSV files to data/ directory"
echo "2. Create virtual environment: python -m venv venv"
echo "3. Activate venv: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
echo "4. Install dependencies: pip install -r requirements.txt"
echo "5. Start implementing app.py, models.py, db.py"
echo ""
echo "Happy coding!"

