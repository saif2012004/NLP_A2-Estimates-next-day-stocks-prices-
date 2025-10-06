@echo off
REM Quick start script for CS4063 Forecasting Application (Windows)

echo ============================================================
echo CS4063 FINANCIAL FORECASTING APPLICATION
echo ============================================================
echo.

REM Check if data directory exists and has CSV files
if not exist "data\" (
    echo [WARNING] data\ directory not found. Creating...
    mkdir data
    echo.
    echo Please copy your CSV files to the data\ directory:
    echo   - AAPL_20250915_185850.csv
    echo   - MSFT_20250915_185853.csv
    echo   - BTC-USD_20250915_185857.csv
    echo.
    pause
)

REM Check if database exists
if not exist "forecasting.db" (
    echo [INFO] Database not found. Will initialize on first run.
    echo Visit /init-db route in browser to load CSV data.
    echo.
)

echo [INFO] Starting Flask application...
echo.
echo Access the application at: http://127.0.0.1:5000
echo.
echo Available routes:
echo   /          - Main forecasting dashboard
echo   /init-db   - Initialize database from CSV files
echo   /stats     - View database statistics
echo.
echo Press CTRL+C to stop the server
echo ============================================================
echo.

REM Run the Flask app
python forecasting_app\app.py

pause

