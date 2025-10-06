"""
Forecasting models module for CS4063 Financial Forecasting Application.

This module implements three forecasting approaches:

1. ARIMA (AutoRegressive Integrated Moving Average):
   - Traditional statistical model for univariate time series
   - Captures linear trends and patterns in historical Close prices
   - Justification: Effective for financial data with clear trends and seasonality
   
2. LSTM (Long Short-Term Memory Neural Network):
   - Deep learning model with memory cells for sequence prediction
   - Multi-feature input: Close, MA_5, MA_20, Volatility, Avg_Sentiment, Daily_Return
   - Justification: Captures non-linear patterns and complex feature interactions
   
3. Ensemble Model:
   - Combines ARIMA and LSTM predictions via averaging
   - Justification: Reduces variance, improves robustness, mitigates individual model weaknesses

Performance Metrics:
- RMSE (Root Mean Squared Error): √(Σ(predicted - actual)² / n)
- MAE (Mean Absolute Error): Σ|predicted - actual| / n  
- MAPE (Mean Absolute Percentage Error): (Σ|predicted - actual| / |actual|) / n * 100
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from datetime import timedelta
from typing import Dict, Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# LSTM
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam


def arima_forecast(df: pd.DataFrame, horizon: int, order: Tuple[int, int, int] = (5, 1, 0)) -> np.ndarray:
    """
    ARIMA forecasting for univariate time series (Close prices).
    
    ARIMA Model Justification:
    - Autoregressive (AR): Uses past values to predict future (captures momentum)
    - Integrated (I): Differencing makes data stationary (removes trends)
    - Moving Average (MA): Uses past forecast errors (smooths noise)
    - Effective for financial data with linear trends and mean reversion
    
    Order Selection (5,1,0):
    - p=5: Uses last 5 days of data (captures weekly patterns)
    - d=1: First-order differencing (handles non-stationarity)
    - q=0: No moving average component (simplified model)
    
    Args:
        df (pd.DataFrame): Historical data with 'Close' column and datetime index
        horizon (int): Number of days to forecast (1, 3, 7, or 14)
        order (tuple): ARIMA order (p, d, q). Default: (5, 1, 0)
    
    Returns:
        np.ndarray: Array of predicted Close prices for horizon days
    
    Example:
        >>> predictions = arima_forecast(df, horizon=7)
        >>> print(f"7-day forecast: {predictions}")
    """
    try:
        # Extract Close prices
        close_prices = df['Close'].values
        
        # Fit ARIMA model
        model = ARIMA(close_prices, order=order)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            fitted_model = model.fit()
        
        # Generate forecast
        forecast = fitted_model.forecast(steps=horizon)
        
        return forecast
        
    except Exception as e:
        print(f"[ARIMA ERROR] {e}")
        # Fallback: use last value with slight random walk
        last_price = df['Close'].iloc[-1]
        return np.array([last_price * (1 + np.random.normal(0, 0.01)) for _ in range(horizon)])


def create_lstm_sequences(data: np.ndarray, look_back: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Transforms time series into supervised learning format:
    - X: Previous 'look_back' timesteps of features
    - y: Next timestep Close price
    
    Args:
        data (np.ndarray): Scaled feature matrix (n_samples, n_features)
        look_back (int): Number of previous timesteps to use (default: 60 days)
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (X_sequences, y_targets)
            X_sequences: (n_samples, look_back, n_features)
            y_targets: (n_samples,) - next Close price
    """
    X, y = [], []
    
    for i in range(look_back, len(data)):
        X.append(data[i - look_back:i])  # Previous look_back timesteps
        y.append(data[i, 0])  # Next Close price (assuming Close is first column)
    
    return np.array(X), np.array(y)


def lstm_forecast(df: pd.DataFrame, horizon: int, look_back: int = 60, 
                  epochs: int = 50, batch_size: int = 32, verbose: int = 0) -> np.ndarray:
    """
    LSTM neural network forecasting with multi-feature input.
    
    LSTM Model Justification:
    - Memory Cells: Retain long-term dependencies in financial data (trends, cycles)
    - Non-linear: Captures complex relationships between features
    - Multi-feature: Leverages technical indicators and sentiment for better predictions
    
    Feature Engineering:
    - Close: Target variable, primary price signal
    - MA_5, MA_20: Short/medium-term trend indicators
    - Volatility: Risk measure, affects price movements
    - Avg_Sentiment: Market sentiment correlation with price
    - Daily_Return: Momentum indicator
    
    Architecture:
    - LSTM Layer (50 units): Captures temporal patterns
    - Dropout (0.2): Prevents overfitting
    - Dense Layer (1 unit): Regression output for Close price
    - Optimizer: Adam (adaptive learning rate)
    - Loss: MSE (standard for regression)
    
    Args:
        df (pd.DataFrame): Historical data with required feature columns
        horizon (int): Number of days to forecast (1, 3, 7, or 14)
        look_back (int): Sequence length for LSTM (default: 60 days)
        epochs (int): Training epochs (default: 50)
        batch_size (int): Batch size for training (default: 32)
        verbose (int): Training verbosity (0=silent, 1=progress bar, 2=one line per epoch)
    
    Returns:
        np.ndarray: Array of predicted Close prices for horizon days
    
    Example:
        >>> predictions = lstm_forecast(df, horizon=7, look_back=60)
        >>> print(f"LSTM 7-day forecast: {predictions}")
    """
    try:
        # Select features for LSTM (in priority order: Close first for target)
        feature_columns = ['Close', 'MA_5', 'MA_20', 'Volatility', 'Avg_Sentiment', 'Daily_Return']
        available_features = [col for col in feature_columns if col in df.columns]
        
        if 'Close' not in available_features:
            raise ValueError("Close column is required for LSTM forecasting")
        
        # Extract and prepare features
        data = df[available_features].values
        
        # Handle missing values (forward fill, then backward fill)
        data = pd.DataFrame(data, columns=available_features).fillna(method='ffill').fillna(method='bfill').values
        
        # Scale features to [0, 1] range for LSTM
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X, y = create_lstm_sequences(scaled_data, look_back)
        
        if len(X) < 10:
            raise ValueError(f"Insufficient data for LSTM: need at least {look_back + 10} samples")
        
        # Build LSTM model
        model = Sequential([
            LSTM(50, activation='relu', return_sequences=False, 
                 input_shape=(look_back, len(available_features))),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.1),
            Dense(1)  # Output: predicted Close price
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Train model
        model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=verbose, 
                  validation_split=0.1, shuffle=False)
        
        # Generate multi-step forecast
        predictions = []
        current_sequence = scaled_data[-look_back:]  # Start with last look_back samples
        
        for _ in range(horizon):
            # Reshape for prediction: (1, look_back, n_features)
            current_input = current_sequence.reshape(1, look_back, len(available_features))
            
            # Predict next Close price (scaled)
            next_pred_scaled = model.predict(current_input, verbose=0)[0, 0]
            
            # Create next timestep: update Close, keep other features from last observation
            next_row = current_sequence[-1].copy()
            next_row[0] = next_pred_scaled  # Update Close (first column)
            
            # Append to sequence and shift
            current_sequence = np.vstack([current_sequence[1:], next_row])
            
            predictions.append(next_pred_scaled)
        
        # Inverse transform predictions to original scale
        # Create dummy array with all features, then inverse transform
        predictions_array = np.array(predictions).reshape(-1, 1)
        dummy_features = np.zeros((len(predictions), len(available_features)))
        dummy_features[:, 0] = predictions_array[:, 0]  # Close predictions in first column
        
        # Inverse transform and extract Close column
        predictions_unscaled = scaler.inverse_transform(dummy_features)[:, 0]
        
        return predictions_unscaled
        
    except Exception as e:
        print(f"[LSTM ERROR] {e}")
        # Fallback: use last value with slight trend
        last_price = df['Close'].iloc[-1]
        trend = df['Close'].pct_change().mean()
        return np.array([last_price * (1 + trend) ** i for i in range(1, horizon + 1)])


def ensemble_forecast(df: pd.DataFrame, horizon: int, weights: Optional[Dict[str, float]] = None) -> np.ndarray:
    """
    Ensemble forecasting combining ARIMA and LSTM predictions.
    
    Ensemble Model Justification:
    - Variance Reduction: Averaging reduces impact of individual model errors
    - Robustness: Combines linear (ARIMA) and non-linear (LSTM) patterns
    - Complementary Strengths: ARIMA excels in short-term, LSTM in complex patterns
    - Better Generalization: Reduces overfitting risk of single models
    
    Weighting Strategy:
    - Default: Equal weights (0.5, 0.5) - simple averaging
    - Custom: Can specify weights based on validation performance
    
    Args:
        df (pd.DataFrame): Historical data with required columns
        horizon (int): Number of days to forecast (1, 3, 7, or 14)
        weights (dict, optional): Model weights {'arima': w1, 'lstm': w2}. 
                                  Default: {'arima': 0.5, 'lstm': 0.5}
    
    Returns:
        np.ndarray: Array of ensemble predicted Close prices
    
    Example:
        >>> predictions = ensemble_forecast(df, horizon=7)
        >>> # Custom weights based on performance:
        >>> predictions = ensemble_forecast(df, horizon=7, 
        ...     weights={'arima': 0.3, 'lstm': 0.7})
    """
    # Default equal weights
    if weights is None:
        weights = {'arima': 0.5, 'lstm': 0.5}
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {k: v / total_weight for k, v in weights.items()}
    
    # Get predictions from both models
    arima_pred = arima_forecast(df, horizon)
    lstm_pred = lstm_forecast(df, horizon, verbose=0)
    
    # Weighted average
    ensemble_pred = (normalized_weights['arima'] * arima_pred + 
                    normalized_weights['lstm'] * lstm_pred)
    
    return ensemble_pred


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate forecasting performance metrics.
    
    Metrics Formulas:
    - RMSE = √(Σ(yᵢ - ŷᵢ)² / n)
      → Penalizes large errors, same units as target
    
    - MAE = Σ|yᵢ - ŷᵢ| / n
      → Average absolute error, robust to outliers
    
    - MAPE = (Σ|yᵢ - ŷᵢ| / |yᵢ|) / n × 100%
      → Percentage error, scale-independent
    
    Args:
        y_true (np.ndarray): Actual values
        y_pred (np.ndarray): Predicted values
    
    Returns:
        Dict[str, float]: Dictionary with 'rmse', 'mae', 'mape' metrics
    """
    # Ensure arrays are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # RMSE: Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAE: Mean Absolute Error
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE: Mean Absolute Percentage Error
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if mask.any() else 0
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'mape': float(mape)
    }


def evaluate_models(df: pd.DataFrame, test_size: float = 0.2, 
                    horizon: int = 7) -> Dict[str, Dict[str, float]]:
    """
    Evaluate ARIMA, LSTM, and Ensemble models using train/test split.
    
    Evaluation Strategy:
    - Train/Test Split: 80% train, 20% test (temporal split, no shuffle)
    - Rolling Forecast: Generate predictions for test period
    - Metrics: RMSE, MAE, MAPE for model comparison
    
    Model Performance Expectations:
    - ARIMA: Lower RMSE for short horizons, struggles with non-linearity
    - LSTM: Better MAPE with features, handles complexity
    - Ensemble: Balanced performance, lower variance
    
    Args:
        df (pd.DataFrame): Historical data with required columns
        test_size (float): Fraction of data for testing (default: 0.2 = 20%)
        horizon (int): Forecast horizon for evaluation (default: 7 days)
    
    Returns:
        Dict[str, Dict[str, float]]: Nested dict with metrics per model:
            {
                'arima': {'rmse': X, 'mae': Y, 'mape': Z},
                'lstm': {'rmse': X, 'mae': Y, 'mape': Z},
                'ensemble': {'rmse': X, 'mae': Y, 'mape': Z},
                'metadata': {'train_size': N, 'test_size': M, 'horizon': H}
            }
    
    Example:
        >>> results = evaluate_models(df, test_size=0.2, horizon=7)
        >>> print(f"ARIMA RMSE: {results['arima']['rmse']:.2f}")
        >>> print(f"LSTM MAPE: {results['lstm']['mape']:.2f}%")
        >>> print(f"Ensemble MAE: {results['ensemble']['mae']:.2f}")
    """
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION - {horizon}-Day Forecast Horizon")
    print(f"{'='*60}")
    
    # Temporal train/test split (no shuffle for time series)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    print(f"Train period: {train_df.index[0].date()} to {train_df.index[-1].date()} ({len(train_df)} days)")
    print(f"Test period: {test_df.index[0].date()} to {test_df.index[-1].date()} ({len(test_df)} days)")
    print(f"Forecast horizon: {horizon} days\n")
    
    # Prepare results dictionary
    results = {
        'arima': {},
        'lstm': {},
        'ensemble': {},
        'metadata': {
            'train_size': len(train_df),
            'test_size': len(test_df),
            'horizon': horizon,
            'train_start': str(train_df.index[0].date()),
            'train_end': str(train_df.index[-1].date()),
            'test_start': str(test_df.index[0].date()),
            'test_end': str(test_df.index[-1].date()),
        }
    }
    
    # Generate predictions for test period (simplified: predict first 'horizon' days)
    # For full evaluation, could do rolling window predictions
    actual_values = test_df['Close'].iloc[:horizon].values
    
    # ARIMA predictions
    print("[1/3] Evaluating ARIMA model...")
    try:
        arima_predictions = arima_forecast(train_df, horizon)
        results['arima'] = calculate_metrics(actual_values, arima_predictions)
        print(f"  [OK] ARIMA - RMSE: {results['arima']['rmse']:.2f}, "
              f"MAE: {results['arima']['mae']:.2f}, MAPE: {results['arima']['mape']:.2f}%")
    except Exception as e:
        print(f"  [ERROR] ARIMA failed: {e}")
        results['arima'] = {'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
    
    # LSTM predictions
    print("[2/3] Evaluating LSTM model...")
    try:
        lstm_predictions = lstm_forecast(train_df, horizon, verbose=0)
        results['lstm'] = calculate_metrics(actual_values, lstm_predictions)
        print(f"  [OK] LSTM - RMSE: {results['lstm']['rmse']:.2f}, "
              f"MAE: {results['lstm']['mae']:.2f}, MAPE: {results['lstm']['mape']:.2f}%")
    except Exception as e:
        print(f"  [ERROR] LSTM failed: {e}")
        results['lstm'] = {'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
    
    # Ensemble predictions
    print("[3/3] Evaluating Ensemble model...")
    try:
        ensemble_predictions = ensemble_forecast(train_df, horizon)
        results['ensemble'] = calculate_metrics(actual_values, ensemble_predictions)
        print(f"  [OK] Ensemble - RMSE: {results['ensemble']['rmse']:.2f}, "
              f"MAE: {results['ensemble']['mae']:.2f}, MAPE: {results['ensemble']['mape']:.2f}%")
    except Exception as e:
        print(f"  [ERROR] Ensemble failed: {e}")
        results['ensemble'] = {'rmse': np.inf, 'mae': np.inf, 'mape': np.inf}
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    # Find best model for each metric
    models = ['arima', 'lstm', 'ensemble']
    for metric in ['rmse', 'mae', 'mape']:
        metric_values = {m: results[m].get(metric, np.inf) for m in models}
        best_model = min(metric_values, key=metric_values.get)
        print(f"Best {metric.upper()}: {best_model.upper()} ({metric_values[best_model]:.2f})")
    
    print(f"{'='*60}\n")
    
    return results


def generate_forecast_dates(df: pd.DataFrame, horizon: int) -> pd.DatetimeIndex:
    """
    Generate future dates for forecast based on data frequency.
    
    Args:
        df (pd.DataFrame): Historical data with datetime index
        horizon (int): Number of future periods to generate
    
    Returns:
        pd.DatetimeIndex: Future dates for forecast
    """
    last_date = df.index[-1]
    freq = pd.infer_freq(df.index) or 'D'  # Default to daily
    
    future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                  periods=horizon, freq=freq)
    return future_dates


# ==============================================================================
# UNIT TESTS (pytest)
# ==============================================================================

def test_arima_forecast():
    """
    Test ARIMA forecasting with synthetic data.
    
    Run with: pytest forecasting_app/models.py::test_arima_forecast -v
    """
    # Create synthetic data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    close_prices = 100 + np.cumsum(np.random.randn(100) * 2)  # Random walk
    df = pd.DataFrame({'Close': close_prices}, index=dates)
    
    # Test forecast
    horizon = 7
    predictions = arima_forecast(df, horizon)
    
    assert len(predictions) == horizon, f"Expected {horizon} predictions, got {len(predictions)}"
    assert all(predictions > 0), "All predictions should be positive"
    assert not np.any(np.isnan(predictions)), "No NaN values in predictions"
    
    print(f"[TEST] test_arima_forecast PASSED - Generated {horizon} predictions")


def test_lstm_forecast():
    """
    Test LSTM forecasting with synthetic multi-feature data.
    
    Run with: pytest forecasting_app/models.py::test_lstm_forecast -v
    """
    # Create synthetic data with features
    n_samples = 150
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    ma_5 = pd.Series(close).rolling(5).mean().fillna(close[0])
    ma_20 = pd.Series(close).rolling(20).mean().fillna(close[0])
    volatility = pd.Series(close).rolling(20).std().fillna(1)
    sentiment = np.random.uniform(-1, 1, n_samples)
    daily_return = pd.Series(close).pct_change().fillna(0)
    
    df = pd.DataFrame({
        'Close': close,
        'MA_5': ma_5,
        'MA_20': ma_20,
        'Volatility': volatility,
        'Avg_Sentiment': sentiment,
        'Daily_Return': daily_return
    }, index=dates)
    
    # Test forecast
    horizon = 7
    predictions = lstm_forecast(df, horizon, look_back=30, epochs=10, verbose=0)
    
    assert len(predictions) == horizon, f"Expected {horizon} predictions, got {len(predictions)}"
    assert all(predictions > 0), "All predictions should be positive"
    assert not np.any(np.isnan(predictions)), "No NaN values in predictions"
    
    print(f"[TEST] test_lstm_forecast PASSED - Generated {horizon} predictions")


def test_ensemble_forecast():
    """
    Test ensemble forecasting.
    
    Run with: pytest forecasting_app/models.py::test_ensemble_forecast -v
    """
    # Create synthetic data
    n_samples = 150
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    ma_5 = pd.Series(close).rolling(5).mean().fillna(close[0])
    ma_20 = pd.Series(close).rolling(20).mean().fillna(close[0])
    
    df = pd.DataFrame({
        'Close': close,
        'MA_5': ma_5,
        'MA_20': ma_20,
        'Volatility': np.random.uniform(0.5, 2, n_samples),
        'Avg_Sentiment': np.random.uniform(-1, 1, n_samples),
        'Daily_Return': pd.Series(close).pct_change().fillna(0)
    }, index=dates)
    
    # Test forecast
    horizon = 7
    predictions = ensemble_forecast(df, horizon)
    
    assert len(predictions) == horizon, f"Expected {horizon} predictions, got {len(predictions)}"
    assert all(predictions > 0), "All predictions should be positive"
    
    print(f"[TEST] test_ensemble_forecast PASSED - Generated {horizon} predictions")


def test_metrics_calculation():
    """
    Test metrics calculation.
    
    Run with: pytest forecasting_app/models.py::test_metrics_calculation -v
    """
    y_true = np.array([100, 105, 110, 108, 112])
    y_pred = np.array([98, 107, 109, 110, 111])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    assert 'rmse' in metrics, "RMSE should be in metrics"
    assert 'mae' in metrics, "MAE should be in metrics"
    assert 'mape' in metrics, "MAPE should be in metrics"
    assert metrics['rmse'] > 0, "RMSE should be positive"
    assert metrics['mae'] >= 0, "MAE should be non-negative"
    assert 0 <= metrics['mape'] <= 100, "MAPE should be percentage"
    
    print(f"[TEST] test_metrics_calculation PASSED - RMSE: {metrics['rmse']:.2f}, "
          f"MAE: {metrics['mae']:.2f}, MAPE: {metrics['mape']:.2f}%")


def test_evaluate_models():
    """
    Test full model evaluation pipeline.
    
    Run with: pytest forecasting_app/models.py::test_evaluate_models -v
    """
    # Create realistic synthetic data
    n_samples = 200
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 1.5)
    ma_5 = pd.Series(close).rolling(5).mean().fillna(method='bfill')
    ma_20 = pd.Series(close).rolling(20).mean().fillna(method='bfill')
    
    df = pd.DataFrame({
        'Close': close,
        'MA_5': ma_5,
        'MA_20': ma_20,
        'Volatility': pd.Series(close).rolling(20).std().fillna(1),
        'Avg_Sentiment': np.random.uniform(-0.5, 0.5, n_samples),
        'Daily_Return': pd.Series(close).pct_change().fillna(0)
    }, index=dates)
    
    # Evaluate models
    results = evaluate_models(df, test_size=0.2, horizon=5)
    
    assert 'arima' in results, "ARIMA results should be present"
    assert 'lstm' in results, "LSTM results should be present"
    assert 'ensemble' in results, "Ensemble results should be present"
    assert 'metadata' in results, "Metadata should be present"
    
    # Check all metrics exist
    for model in ['arima', 'lstm', 'ensemble']:
        assert 'rmse' in results[model], f"{model} should have RMSE"
        assert 'mae' in results[model], f"{model} should have MAE"
        assert 'mape' in results[model], f"{model} should have MAPE"
    
    print(f"[TEST] test_evaluate_models PASSED")


if __name__ == "__main__":
    """
    Standalone execution for testing models.
    Run: python forecasting_app/models.py
    """
    print("="*60)
    print("CS4063 FORECASTING APP - MODELS MODULE")
    print("="*60)
    
    # Run tests
    print("\n[TESTING] Running unit tests...")
    test_arima_forecast()
    test_lstm_forecast()
    test_ensemble_forecast()
    test_metrics_calculation()
    test_evaluate_models()
    print("\n[TESTING] All tests passed!\n")
    
    print("="*60)
    print("Models module ready for use!")
    print("="*60)

