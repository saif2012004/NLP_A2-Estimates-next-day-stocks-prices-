@echo off
REM CS4063 Forecasting Application - Project Setup Script (Windows)
REM This script creates the project directory structure and initializes git repository

echo Setting up CS4063 Forecasting Application...

REM Create main project directory structure
mkdir forecasting_app 2>nul
mkdir forecasting_app\templates 2>nul
mkdir forecasting_app\static 2>nul
mkdir forecasting_app\static\css 2>nul
mkdir forecasting_app\static\js 2>nul
mkdir tests 2>nul
mkdir data 2>nul
mkdir docs 2>nul
mkdir notebooks 2>nul

echo [OK] Created directory structure

echo.
echo Note: Please copy CSV files to data\ directory:
echo   - BTC-USD_20250915_185857.csv
echo   - MSFT_20250915_185853.csv
echo   - AAPL_20250915_185850.csv

REM Initialize git repository
git init
echo [OK] Initialized git repository

REM Create .gitignore
(
echo # Python
echo __pycache__/
echo *.py[cod]
echo *$py.class
echo *.so
echo .Python
echo env/
echo venv/
echo ENV/
echo build/
echo develop-eggs/
echo dist/
echo downloads/
echo eggs/
echo .eggs/
echo lib/
echo lib64/
echo parts/
echo sdist/
echo var/
echo wheels/
echo *.egg-info/
echo .installed.cfg
echo *.egg
echo.
echo # Virtual Environment
echo venv/
echo ENV/
echo env/
echo.
echo # IDEs
echo .vscode/
echo .idea/
echo *.swp
echo *.swo
echo *~
echo.
echo # Database
echo *.db
echo *.sqlite
echo *.sqlite3
echo.
echo # Logs
echo *.log
echo.
echo # OS
echo .DS_Store
echo Thumbs.db
echo.
echo # Jupyter Notebooks
echo .ipynb_checkpoints/
echo.
echo # Environment variables
echo .env
echo.
echo # Model artifacts
echo models/*.h5
echo models/*.pkl
echo models/*.joblib
echo.
echo # Coverage reports
echo htmlcov/
echo .coverage
echo .coverage.*
echo coverage.xml
echo *.cover
echo.
echo # pytest
echo .pytest_cache/
) > .gitignore

echo [OK] Created .gitignore

REM Create empty __init__.py files
type nul > forecasting_app\__init__.py
type nul > tests\__init__.py

echo [OK] Created __init__.py files

REM Create README (basic version for Windows batch)
(
echo # CS4063 Financial Forecasting Application
echo.
echo A Flask-based web application for forecasting financial instruments.
echo.
echo ## Setup
echo.
echo 1. Create virtual environment: python -m venv venv
echo 2. Activate: venv\Scripts\activate
echo 3. Install dependencies: pip install -r requirements.txt
echo 4. Copy CSV files to data\ directory
echo 5. Run: cd forecasting_app ^&^& python app.py
) > README.md

echo [OK] Created README.md

REM Make initial git commit
git add .
git commit -m "Initial project setup for CS4063 Forecasting Application"

echo [OK] Made initial git commit

echo.
echo ==========================================
echo Setup complete! Next steps:
echo ==========================================
echo 1. Copy CSV files to data\ directory
echo 2. Create virtual environment: python -m venv venv
echo 3. Activate venv: venv\Scripts\activate
echo 4. Install dependencies: pip install -r requirements.txt
echo 5. Start implementing app.py, models.py, db.py
echo.
echo Happy coding!
pause

