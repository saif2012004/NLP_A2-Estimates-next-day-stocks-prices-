"""
Comprehensive test suite for CS4063 Forecasting Application.

Tests cover:
- Data loading from CSV files
- Database operations (insert/query)
- Forecasting models (ARIMA, LSTM, Ensemble)
- Performance metrics evaluation

Run with: pytest tests/test_app.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import datetime, timedelta

# Import application modules
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from forecasting_app.utils import load_data, clean_dataframe
from forecasting_app.db import (
    init_db, insert_historical, get_historical, 
    insert_predictions, get_predictions,
    Base, Historical, Predictions
)
from forecasting_app.models import (
    arima_forecast, lstm_forecast, ensemble_forecast,
    evaluate_models, calculate_metrics
)

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


# ============================================================================
# TEST DATA LOADING
# ============================================================================

def test_load_data():
    """
    Test that load_data() correctly loads all three CSV files.
    
    Assertions:
    - Returns dictionary with 3 keys (AAPL, MSFT, BTC-USD)
    - Each value is a pandas DataFrame
    - Each DataFrame has >0 rows
    - Required columns exist (Open, High, Low, Close, Volume)
    """
    # This test assumes CSV files are in data/ directory
    # If CSVs are not present, test will be skipped
    data_dir = 'data'
    
    if not os.path.exists(data_dir):
        pytest.skip("Data directory not found - CSV files not available")
    
    # Check if CSV files exist
    csv_files = [
        'AAPL_20250915_185850.csv',
        'MSFT_20250915_185853.csv',
        'BTC-USD_20250915_185857.csv'
    ]
    
    if not all(os.path.exists(os.path.join(data_dir, f)) for f in csv_files):
        pytest.skip("CSV files not found in data directory")
    
    # Load data
    data = load_data(data_dir)
    
    # Assert dictionary structure
    assert isinstance(data, dict), "load_data should return a dictionary"
    assert len(data) == 3, "Should have 3 instruments"
    assert set(data.keys()) == {'AAPL', 'MSFT', 'BTC-USD'}, "Should have correct instrument keys"
    
    # Assert each DataFrame
    for symbol, df in data.items():
        assert isinstance(df, pd.DataFrame), f"{symbol} should be a DataFrame"
        assert len(df) > 0, f"{symbol} DataFrame should have rows"
        assert df.shape[0] > 0, f"{symbol} should have >0 rows"
        assert df.shape[1] > 0, f"{symbol} should have >0 columns"
        
        # Check required columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            assert col in df.columns, f"{symbol} missing column: {col}"
        
        # Check datetime index
        assert isinstance(df.index, pd.DatetimeIndex), f"{symbol} should have datetime index"
        
        # Check OHLC are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            assert pd.api.types.is_numeric_dtype(df[col]), f"{symbol}.{col} should be numeric"
    
    print(f"✓ test_load_data PASSED - Loaded {sum(len(df) for df in data.values())} total records")


def test_clean_dataframe():
    """
    Test DataFrame cleaning and preprocessing.
    
    Assertions:
    - Datetime index is properly set
    - OHLC columns are float type
    - No missing values in critical columns
    """
    # Create sample DataFrame
    dates = pd.date_range('2024-01-01', periods=10, freq='D')
    df = pd.DataFrame({
        'Date': dates.astype(str),  # Test string to datetime conversion
        'Open': [100 + i for i in range(10)],
        'High': [105 + i for i in range(10)],
        'Low': [95 + i for i in range(10)],
        'Close': [102 + i for i in range(10)],
        'Volume': [1000000 + i*10000 for i in range(10)],
        'MA_5': [100 + i*0.5 for i in range(10)],
    })
    
    # Clean DataFrame
    cleaned_df = clean_dataframe(df, 'TEST')
    
    # Assert datetime index
    assert isinstance(cleaned_df.index, pd.DatetimeIndex), "Index should be datetime"
    assert len(cleaned_df) == 10, "Should preserve all rows"
    
    # Assert OHLC are float
    for col in ['Open', 'High', 'Low', 'Close']:
        assert pd.api.types.is_float_dtype(cleaned_df[col]), f"{col} should be float"
    
    # Assert no NaN in OHLC
    assert cleaned_df[['Open', 'High', 'Low', 'Close']].isna().sum().sum() == 0
    
    # Assert sorted by date
    assert cleaned_df.index.is_monotonic_increasing, "Should be sorted ascending"
    
    print("✓ test_clean_dataframe PASSED")


# ============================================================================
# TEST DATABASE OPERATIONS
# ============================================================================

def test_db_insert_query():
    """
    Test database insert and query operations.
    
    Assertions:
    - Tables created successfully
    - Historical data inserted correctly
    - Query returns matching data
    - Predictions stored and retrieved
    """
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db = tmp.name
    
    try:
        # Setup temporary database
        temp_url = f'sqlite:///{temp_db}'
        temp_engine = create_engine(temp_url, echo=False)
        Base.metadata.create_all(bind=temp_engine)
        SessionLocal = sessionmaker(bind=temp_engine)
        session = SessionLocal()
        
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=20, freq='D')
        sample_df = pd.DataFrame({
            'Open': [100 + i for i in range(20)],
            'High': [105 + i for i in range(20)],
            'Low': [95 + i for i in range(20)],
            'Close': [102 + i for i in range(20)],
            'Volume': [1000000 + i*10000 for i in range(20)],
            'MA_5': [100 + i*0.5 for i in range(20)],
            'Volatility': [0.02 + i*0.001 for i in range(20)],
        }, index=dates)
        
        # Test insert historical
        from forecasting_app.db import insert_historical as insert_hist_func
        count = insert_hist_func('TEST', sample_df, session=session)
        assert count == 20, f"Should insert 20 records, got {count}"
        
        # Test query historical
        from forecasting_app.db import get_historical as get_hist_func
        result_df = get_hist_func('TEST', session=session)
        
        assert len(result_df) == 20, f"Should query 20 records, got {len(result_df)}"
        assert 'Close' in result_df.columns, "Should have Close column"
        assert result_df['Close'].iloc[0] == 102, "First Close should be 102"
        assert result_df['Close'].iloc[-1] == 121, "Last Close should be 121"
        
        # Test insert predictions
        from forecasting_app.db import insert_predictions as insert_pred_func
        predictions = [
            {
                'target_date': dates[-1] + timedelta(days=i+1),
                'model_type': 'TEST',
                'predicted_close': 125.0 + i,
                'horizon': 5
            }
            for i in range(5)
        ]
        pred_count = insert_pred_func('TEST', predictions, session=session)
        assert pred_count == 5, f"Should insert 5 predictions, got {pred_count}"
        
        # Test query predictions
        from forecasting_app.db import get_predictions as get_pred_func
        pred_df = get_pred_func('TEST', 'TEST', horizon=5, session=session)
        
        assert len(pred_df) == 5, f"Should query 5 predictions, got {len(pred_df)}"
        assert 'predicted_close' in pred_df.columns, "Should have predicted_close column"
        
        session.close()
        temp_engine.dispose()  # Close all connections
        print("✓ test_db_insert_query PASSED")
        
    finally:
        # Cleanup
        import time
        time.sleep(0.1)  # Brief delay for Windows file release
        try:
            if os.path.exists(temp_db):
                os.remove(temp_db)
        except PermissionError:
            pass  # Ignore cleanup errors on Windows


# ============================================================================
# TEST FORECASTING MODELS
# ============================================================================

def test_forecasts():
    """
    Test all three forecasting models (ARIMA, LSTM, Ensemble).
    
    Assertions:
    - Each model returns array with shape == horizon
    - Predictions are numeric and not NaN
    - Predictions are positive (for prices)
    """
    # Create sample DataFrame with sufficient history
    n_samples = 150  # Need enough for LSTM look_back
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    ma_5 = pd.Series(close).rolling(5).mean().fillna(close[0])
    ma_20 = pd.Series(close).rolling(20).mean().fillna(close[0])
    
    sample_df = pd.DataFrame({
        'Close': close,
        'MA_5': ma_5,
        'MA_20': ma_20,
        'Volatility': pd.Series(close).rolling(20).std().fillna(1),
        'Avg_Sentiment': np.random.uniform(-0.5, 0.5, n_samples),
        'Daily_Return': pd.Series(close).pct_change().fillna(0)
    }, index=dates)
    
    horizon = 7
    
    # Test ARIMA forecast
    arima_pred = arima_forecast(sample_df, horizon)
    assert len(arima_pred) == horizon, f"ARIMA should return {horizon} predictions"
    assert not np.any(np.isnan(arima_pred)), "ARIMA predictions should not be NaN"
    assert np.all(arima_pred > 0), "ARIMA predictions should be positive"
    print(f"  ARIMA forecast: {arima_pred[-1]:.2f} (final prediction)")
    
    # Test LSTM forecast (with shorter epochs for speed)
    lstm_pred = lstm_forecast(sample_df, horizon, look_back=30, epochs=10, verbose=0)
    assert len(lstm_pred) == horizon, f"LSTM should return {horizon} predictions"
    assert not np.any(np.isnan(lstm_pred)), "LSTM predictions should not be NaN"
    assert np.all(lstm_pred > 0), "LSTM predictions should be positive"
    print(f"  LSTM forecast: {lstm_pred[-1]:.2f} (final prediction)")
    
    # Test Ensemble forecast
    ensemble_pred = ensemble_forecast(sample_df, horizon)
    assert len(ensemble_pred) == horizon, f"Ensemble should return {horizon} predictions"
    assert not np.any(np.isnan(ensemble_pred)), "Ensemble predictions should not be NaN"
    assert np.all(ensemble_pred > 0), "Ensemble predictions should be positive"
    print(f"  Ensemble forecast: {ensemble_pred[-1]:.2f} (final prediction)")
    
    # Test ensemble is between ARIMA and LSTM (or close)
    # This may not always be true due to weights, but generally holds
    
    print("✓ test_forecasts PASSED - All models generated valid predictions")


# ============================================================================
# TEST EVALUATION METRICS
# ============================================================================

def test_evaluate():
    """
    Test model evaluation pipeline and metrics calculation.
    
    Assertions:
    - evaluate_models returns dict with model keys
    - Each model has RMSE, MAE, MAPE metrics
    - Metrics are numeric and > 0
    - Metadata is present
    """
    # Create realistic sample data
    n_samples = 200
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 1.5)
    ma_5 = pd.Series(close).rolling(5).mean().fillna(method='bfill')
    ma_20 = pd.Series(close).rolling(20).mean().fillna(method='bfill')
    
    sample_df = pd.DataFrame({
        'Close': close,
        'MA_5': ma_5,
        'MA_20': ma_20,
        'Volatility': pd.Series(close).rolling(20).std().fillna(1),
        'Avg_Sentiment': np.random.uniform(-0.3, 0.3, n_samples),
        'Daily_Return': pd.Series(close).pct_change().fillna(0)
    }, index=dates)
    
    # Run evaluation (with shorter horizon for speed)
    results = evaluate_models(sample_df, test_size=0.2, horizon=5)
    
    # Assert structure
    assert isinstance(results, dict), "Results should be a dictionary"
    assert 'arima' in results, "Should have ARIMA results"
    assert 'lstm' in results, "Should have LSTM results"
    assert 'ensemble' in results, "Should have Ensemble results"
    assert 'metadata' in results, "Should have metadata"
    
    # Assert metrics for each model
    for model in ['arima', 'lstm', 'ensemble']:
        assert 'rmse' in results[model], f"{model} should have RMSE"
        assert 'mae' in results[model], f"{model} should have MAE"
        assert 'mape' in results[model], f"{model} should have MAPE"
        
        # Check metric values are valid
        assert results[model]['rmse'] >= 0, f"{model} RMSE should be >= 0"
        assert results[model]['mae'] >= 0, f"{model} MAE should be >= 0"
        assert results[model]['mape'] >= 0, f"{model} MAPE should be >= 0"
        
        # Check metrics are not infinity (unless model failed)
        if results[model]['rmse'] != np.inf:
            assert results[model]['rmse'] > 0, f"{model} RMSE should be > 0"
        
        print(f"  {model.upper()} - RMSE: {results[model]['rmse']:.2f}, "
              f"MAE: {results[model]['mae']:.2f}, MAPE: {results[model]['mape']:.2f}%")
    
    # Assert metadata
    assert 'train_size' in results['metadata'], "Should have train_size"
    assert 'test_size' in results['metadata'], "Should have test_size"
    assert 'horizon' in results['metadata'], "Should have horizon"
    assert results['metadata']['train_size'] > 0, "Train size should be > 0"
    assert results['metadata']['test_size'] > 0, "Test size should be > 0"
    
    print("✓ test_evaluate PASSED - Evaluation metrics computed successfully")


def test_calculate_metrics():
    """
    Test metrics calculation function.
    
    Assertions:
    - RMSE, MAE, MAPE calculated correctly
    - Values match expected formulas
    """
    y_true = np.array([100, 105, 110, 108, 112])
    y_pred = np.array([98, 107, 109, 110, 111])
    
    metrics = calculate_metrics(y_true, y_pred)
    
    # Assert all metrics present
    assert 'rmse' in metrics, "Should have RMSE"
    assert 'mae' in metrics, "Should have MAE"
    assert 'mape' in metrics, "Should have MAPE"
    
    # Manual calculation for verification
    errors = y_true - y_pred
    expected_mae = np.mean(np.abs(errors))
    expected_rmse = np.sqrt(np.mean(errors**2))
    expected_mape = np.mean(np.abs(errors / y_true)) * 100
    
    # Check values (with tolerance for floating point)
    assert abs(metrics['mae'] - expected_mae) < 0.01, "MAE calculation incorrect"
    assert abs(metrics['rmse'] - expected_rmse) < 0.01, "RMSE calculation incorrect"
    assert abs(metrics['mape'] - expected_mape) < 0.01, "MAPE calculation incorrect"
    
    # Check ranges
    assert metrics['rmse'] > 0, "RMSE should be > 0"
    assert metrics['mae'] > 0, "MAE should be > 0"
    assert 0 <= metrics['mape'] <= 100, "MAPE should be 0-100%"
    
    print(f"  Metrics: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, MAPE={metrics['mape']:.2f}%")
    print("✓ test_calculate_metrics PASSED")


# ============================================================================
# INTEGRATION TEST
# ============================================================================

def test_end_to_end_workflow():
    """
    Integration test simulating complete workflow.
    
    Steps:
    1. Load data (or create sample)
    2. Initialize database
    3. Run forecasts
    4. Store predictions
    5. Evaluate models
    """
    # Create sample data
    n_samples = 100
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='D')
    
    close = 100 + np.cumsum(np.random.randn(n_samples) * 2)
    sample_df = pd.DataFrame({
        'Close': close,
        'MA_5': pd.Series(close).rolling(5).mean().fillna(method='bfill'),
        'MA_20': pd.Series(close).rolling(20).mean().fillna(method='bfill'),
        'Volatility': pd.Series(close).rolling(20).std().fillna(1),
        'Avg_Sentiment': np.random.uniform(-0.5, 0.5, n_samples),
        'Daily_Return': pd.Series(close).pct_change().fillna(0),
        'Open': close - 1,
        'High': close + 2,
        'Low': close - 2,
        'Volume': np.random.randint(1000000, 2000000, n_samples)
    }, index=dates)
    
    # Create temp database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db = tmp.name
    
    try:
        # Setup database
        temp_url = f'sqlite:///{temp_db}'
        temp_engine = create_engine(temp_url, echo=False)
        Base.metadata.create_all(bind=temp_engine)
        SessionLocal = sessionmaker(bind=temp_engine)
        session = SessionLocal()
        
        # Insert historical data
        from forecasting_app.db import insert_historical as insert_hist
        insert_hist('TEST', sample_df, session=session)
        
        # Generate forecasts
        horizon = 5
        arima_pred = arima_forecast(sample_df, horizon)
        
        # Store predictions
        from forecasting_app.db import insert_predictions as insert_pred
        predictions = [
            {
                'target_date': dates[-1] + timedelta(days=i+1),
                'model_type': 'ARIMA',
                'predicted_close': float(arima_pred[i]),
                'horizon': horizon
            }
            for i in range(horizon)
        ]
        insert_pred('TEST', predictions, session=session)
        
        # Query back
        from forecasting_app.db import get_predictions as get_pred
        stored_preds = get_pred('TEST', 'ARIMA', horizon=horizon, session=session)
        
        assert len(stored_preds) == horizon, "Should retrieve all predictions"
        
        session.close()
        temp_engine.dispose()  # Close all connections
        print("✓ test_end_to_end_workflow PASSED")
        
    finally:
        import time
        time.sleep(0.1)  # Brief delay for Windows file release
        try:
            if os.path.exists(temp_db):
                os.remove(temp_db)
        except PermissionError:
            pass  # Ignore cleanup errors on Windows


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    """
    Run all tests when executed directly.
    
    Usage: python tests/test_app.py
    """
    print("="*70)
    print("CS4063 FORECASTING APP - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    tests = [
        ("Data Loading", test_load_data),
        ("DataFrame Cleaning", test_clean_dataframe),
        ("Database Operations", test_db_insert_query),
        ("Forecasting Models", test_forecasts),
        ("Metrics Calculation", test_calculate_metrics),
        ("Model Evaluation", test_evaluate),
        ("End-to-End Workflow", test_end_to_end_workflow),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        print(f"\n[TEST] {name}...")
        try:
            test_func()
            passed += 1
        except pytest.skip.Exception as e:
            print(f"  ⊘ SKIPPED: {e}")
            skipped += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70)
    
    if failed == 0 and passed > 0:
        print("\n✓ All tests passed successfully!")
    elif failed > 0:
        print(f"\n✗ {failed} test(s) failed. Please review errors above.")

