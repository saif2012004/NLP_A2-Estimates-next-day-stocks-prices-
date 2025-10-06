"""
Flask Web Application for CS4063 Financial Forecasting System.

This application provides a user-friendly web interface for:
1. Instrument selection (AAPL, MSFT, BTC-USD)
2. Forecast horizon selection (1, 3, 7, 14 days)
3. Interactive candlestick visualization with forecast overlays
4. Model comparison (ARIMA, LSTM, Ensemble)

Visualization Justification:
- Candlestick charts: Industry-standard for OHLC financial data visualization
- Line overlays: Clear distinction between model predictions
- Interactive Plotly: Zoom, pan, hover tooltips for detailed analysis
- Color coding: Different colors for each model to aid comparison
- Future dates: Extended x-axis to show forecast period

Architecture:
- Flask backend for request handling and model orchestration
- SQLite database for data persistence
- Plotly for interactive visualizations
- Responsive HTML templates for cross-device compatibility
"""

from flask import Flask, render_template, request, flash, redirect, url_for
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Import our modules
try:
    # Try relative imports first (when imported as module)
    from .db import init_db, get_historical, insert_predictions, get_db_stats
    from .models import arima_forecast, lstm_forecast, ensemble_forecast
    from .utils import get_data_info
except ImportError:
    # Fall back to absolute imports (when run directly)
    from forecasting_app.db import init_db, get_historical, insert_predictions, get_db_stats
    from forecasting_app.models import arima_forecast, lstm_forecast, ensemble_forecast
    from forecasting_app.utils import get_data_info

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'cs4063-forecasting-app-secret-key-2024'

# Configuration
INSTRUMENTS = ['AAPL', 'MSFT', 'BTC-USD']
HORIZONS = [1, 3, 7, 14]
CHART_DAYS_TO_SHOW = 90  # Show last 90 days of historical data in chart

# Color scheme for models (distinct, colorblind-friendly)
MODEL_COLORS = {
    'arima': '#FF6B6B',      # Red
    'lstm': '#4ECDC4',       # Teal
    'ensemble': '#45B7D1'    # Blue
}


def create_forecast_chart(df: pd.DataFrame, forecasts: dict, instrument: str, horizon: int) -> str:
    """
    Create interactive Plotly candlestick chart with forecast overlays.
    
    Visualization Components:
    1. Candlestick: Historical OHLC data (last 90 days for clarity)
    2. Line Overlays: ARIMA, LSTM, Ensemble predictions
    3. Hover Info: Detailed price information on mouseover
    4. Legend: Model identification with color coding
    5. Future Dates: Extended x-axis showing forecast period
    
    Justification:
    - Candlestick: Standard in finance, shows price action (open, high, low, close)
    - Line overlays: Clear visual distinction between predictions
    - Interactive: Zoom, pan, and inspect specific data points
    - Color coding: Quick model identification (red=ARIMA, teal=LSTM, blue=Ensemble)
    
    Args:
        df (pd.DataFrame): Historical OHLC data with datetime index
        forecasts (dict): Dictionary with 'arima', 'lstm', 'ensemble' predictions
        instrument (str): Instrument symbol for title
        horizon (int): Forecast horizon in days
    
    Returns:
        str: HTML string of Plotly figure
    """
    # Limit historical data to last N days for better visualization
    df_recent = df.tail(CHART_DAYS_TO_SHOW).copy()
    
    # Create figure with secondary y-axis (optional for volume)
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=[f'{instrument} - {horizon}-Day Forecast'],
        vertical_spacing=0.03
    )
    
    # Add candlestick trace for historical data
    fig.add_trace(
        go.Candlestick(
            x=df_recent.index,
            open=df_recent['Open'],
            high=df_recent['High'],
            low=df_recent['Low'],
            close=df_recent['Close'],
            name='Historical',
            increasing_line_color='#26A69A',  # Green for up days
            decreasing_line_color='#EF5350',  # Red for down days
        ),
        row=1, col=1
    )
    
    # Generate future dates for forecasts
    last_date = df.index[-1]
    forecast_dates = pd.date_range(
        start=last_date + timedelta(days=1),
        periods=horizon,
        freq='D'
    )
    
    # Add ARIMA forecast line
    if 'arima' in forecasts and forecasts['arima'] is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecasts['arima'],
                mode='lines+markers',
                name='ARIMA Forecast',
                line=dict(color=MODEL_COLORS['arima'], width=3, dash='dash'),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='<b>ARIMA</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add LSTM forecast line
    if 'lstm' in forecasts and forecasts['lstm'] is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecasts['lstm'],
                mode='lines+markers',
                name='LSTM Forecast',
                line=dict(color=MODEL_COLORS['lstm'], width=3, dash='dot'),
                marker=dict(size=8, symbol='square'),
                hovertemplate='<b>LSTM</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add Ensemble forecast line
    if 'ensemble' in forecasts and forecasts['ensemble'] is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecasts['ensemble'],
                mode='lines+markers',
                name='Ensemble Forecast',
                line=dict(color=MODEL_COLORS['ensemble'], width=4),
                marker=dict(size=10, symbol='diamond'),
                hovertemplate='<b>Ensemble</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Add vertical line to separate historical and forecast
    fig.add_vline(
        x=last_date.timestamp() * 1000,  # Plotly uses milliseconds
        line_dash="dash",
        line_color="gray",
        opacity=0.5,
        annotation_text="Forecast Start",
        annotation_position="top"
    )
    
    # Update layout for better visualization
    fig.update_layout(
        title={
            'text': f'{instrument} Price Forecast - {horizon} Days Ahead',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 24, 'family': 'Arial, sans-serif'}
        },
        xaxis_title='Date',
        yaxis_title='Price ($)',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="gray",
            borderwidth=1
        ),
        xaxis=dict(
            rangeslider=dict(visible=False),  # Hide range slider for cleaner look
            type='date',
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='lightgray',
            tickprefix='$'
        )
    )
    
    # Convert to HTML (div only, not full page)
    chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
    
    return chart_html


@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main route for the forecasting application.
    
    GET: Display instrument/horizon selection form
    POST: Process selection, generate forecasts, display visualization
    
    Workflow:
    1. User selects instrument and horizon via form
    2. Fetch historical data from database
    3. Generate forecasts using ARIMA, LSTM, Ensemble models
    4. Store predictions in database
    5. Create interactive visualization
    6. Render template with chart
    
    Returns:
        Rendered HTML template with form and optional chart
    """
    chart_html = None
    selected_instrument = None
    selected_horizon = None
    forecast_summary = None
    
    if request.method == 'POST':
        try:
            # Get form data
            selected_instrument = request.form.get('instrument')
            selected_horizon = int(request.form.get('horizon'))
            
            # Validate inputs
            if selected_instrument not in INSTRUMENTS:
                flash(f'Invalid instrument: {selected_instrument}', 'danger')
                return redirect(url_for('index'))
            
            if selected_horizon not in HORIZONS:
                flash(f'Invalid horizon: {selected_horizon}', 'danger')
                return redirect(url_for('index'))
            
            print(f"\n{'='*60}")
            print(f"FORECAST REQUEST: {selected_instrument} - {selected_horizon} days")
            print(f"{'='*60}")
            
            # Fetch historical data from database
            print("[1/5] Fetching historical data from database...")
            df = get_historical(selected_instrument)
            
            if df.empty:
                flash(f'No historical data found for {selected_instrument}. Please initialize database.', 'danger')
                return redirect(url_for('index'))
            
            print(f"  Retrieved {len(df)} historical records")
            
            # Generate forecasts using all three models
            print("[2/5] Generating forecasts...")
            
            forecasts = {}
            
            # ARIMA forecast
            print("  Running ARIMA model...")
            try:
                arima_pred = arima_forecast(df, selected_horizon)
                forecasts['arima'] = arima_pred
                print(f"    ARIMA: {arima_pred[-1]:.2f} (final prediction)")
            except Exception as e:
                print(f"    ARIMA failed: {e}")
                forecasts['arima'] = None
            
            # LSTM forecast
            print("  Running LSTM model...")
            try:
                lstm_pred = lstm_forecast(df, selected_horizon, verbose=0)
                forecasts['lstm'] = lstm_pred
                print(f"    LSTM: {lstm_pred[-1]:.2f} (final prediction)")
            except Exception as e:
                print(f"    LSTM failed: {e}")
                forecasts['lstm'] = None
            
            # Ensemble forecast
            print("  Running Ensemble model...")
            try:
                ensemble_pred = ensemble_forecast(df, selected_horizon)
                forecasts['ensemble'] = ensemble_pred
                print(f"    Ensemble: {ensemble_pred[-1]:.2f} (final prediction)")
            except Exception as e:
                print(f"    Ensemble failed: {e}")
                forecasts['ensemble'] = None
            
            # Store predictions in database
            print("[3/5] Storing predictions in database...")
            last_date = df.index[-1]
            
            for model_type, predictions in forecasts.items():
                if predictions is not None:
                    pred_list = []
                    for i, pred_value in enumerate(predictions):
                        pred_list.append({
                            'target_date': last_date + timedelta(days=i+1),
                            'model_type': model_type.upper(),
                            'predicted_close': float(pred_value),
                            'horizon': selected_horizon
                        })
                    
                    insert_predictions(selected_instrument, pred_list)
            
            # Create visualization
            print("[4/5] Creating visualization...")
            chart_html = create_forecast_chart(df, forecasts, selected_instrument, selected_horizon)
            
            # Prepare forecast summary for display
            forecast_summary = {
                'instrument': selected_instrument,
                'horizon': selected_horizon,
                'last_price': df['Close'].iloc[-1],
                'forecasts': {}
            }
            
            for model_type, predictions in forecasts.items():
                if predictions is not None:
                    forecast_summary['forecasts'][model_type] = {
                        'final': predictions[-1],
                        'change': predictions[-1] - df['Close'].iloc[-1],
                        'change_pct': ((predictions[-1] - df['Close'].iloc[-1]) / df['Close'].iloc[-1]) * 100
                    }
            
            print("[5/5] Rendering visualization...")
            flash(f'Forecast generated successfully for {selected_instrument} ({selected_horizon} days)', 'success')
            print(f"{'='*60}\n")
            
        except Exception as e:
            flash(f'Error generating forecast: {str(e)}', 'danger')
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
    
    # Get database stats for display
    try:
        db_stats = get_db_stats()
    except:
        db_stats = {'historical_records': 0, 'prediction_records': 0, 'instruments': 0}
    
    return render_template(
        'index.html',
        instruments=INSTRUMENTS,
        horizons=HORIZONS,
        selected_instrument=selected_instrument,
        selected_horizon=selected_horizon,
        chart=chart_html,
        forecast_summary=forecast_summary,
        db_stats=db_stats
    )


@app.route('/init-db')
def initialize_database():
    """
    Route to initialize database with CSV data.
    
    This route can be accessed to load historical data from CSV files
    into the database. Should be called once during setup.
    
    Returns:
        Redirect to index with status message
    """
    try:
        print("\n[DATABASE INITIALIZATION]")
        init_db(data_dir='data', force_reload=False)
        flash('Database initialized successfully!', 'success')
    except Exception as e:
        flash(f'Database initialization failed: {str(e)}', 'danger')
        print(f"[ERROR] {e}")
    
    return redirect(url_for('index'))


@app.route('/stats')
def stats():
    """
    Route to display database statistics and system info.
    
    Returns:
        Rendered template with statistics
    """
    try:
        db_stats = get_db_stats()
        return render_template('stats.html', stats=db_stats)
    except Exception as e:
        flash(f'Error loading statistics: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return render_template('500.html'), 500


if __name__ == '__main__':
    """
    Run Flask development server.
    
    For production deployment:
    - Use a production WSGI server (gunicorn, uWSGI)
    - Set debug=False
    - Configure proper SECRET_KEY
    - Use environment variables for configuration
    """
    print("="*60)
    print("CS4063 FINANCIAL FORECASTING APPLICATION")
    print("="*60)
    print("\nStarting Flask development server...")
    print("Access the application at: http://127.0.0.1:5000")
    print("\nAvailable routes:")
    print("  /          - Main forecasting interface")
    print("  /init-db   - Initialize database from CSV files")
    print("  /stats     - View database statistics")
    print("\nPress CTRL+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, host='127.0.0.1', port=5000)

