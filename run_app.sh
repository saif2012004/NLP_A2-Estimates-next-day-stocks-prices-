#!/bin/bash
# Quick start script for CS4063 Forecasting Application (Linux/Mac)

echo "============================================================"
echo "CS4063 FINANCIAL FORECASTING APPLICATION"
echo "============================================================"
echo

# Check if data directory exists and has CSV files
if [ ! -d "data" ]; then
    echo "[WARNING] data/ directory not found. Creating..."
    mkdir -p data
    echo
    echo "Please copy your CSV files to the data/ directory:"
    echo "  - AAPL_20250915_185850.csv"
    echo "  - MSFT_20250915_185853.csv"
    echo "  - BTC-USD_20250915_185857.csv"
    echo
    read -p "Press Enter to continue..."
fi

# Check if database exists
if [ ! -f "forecasting.db" ]; then
    echo "[INFO] Database not found. Will initialize on first run."
    echo "Visit /init-db route in browser to load CSV data."
    echo
fi

echo "[INFO] Starting Flask application..."
echo
echo "Access the application at: http://127.0.0.1:5000"
echo
echo "Available routes:"
echo "  /          - Main forecasting dashboard"
echo "  /init-db   - Initialize database from CSV files"
echo "  /stats     - View database statistics"
echo
echo "Press CTRL+C to stop the server"
echo "============================================================"
echo

# Run the Flask app
python forecasting_app/app.py

