"""
Simple runner script for the Flask application.
This handles import paths correctly.
"""
import sys
import os

# Add the project directory to Python path
project_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_dir)

# Now import and run the app
from forecasting_app.app import app

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)

