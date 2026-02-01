"""
Financial Forecasting Application
Main entry point for running the Flask application
"""

import os
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from forecasting_app.app import app

if __name__ == '__main__':
    print("=" * 60)
    print("Financial Forecasting Application")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("\nPress CTRL+C to quit")
    print("=" * 60)
    
    app.run(debug=True, host='127.0.0.1', port=5000)
