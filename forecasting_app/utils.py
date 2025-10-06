"""
Utility functions for the CS4063 Forecasting Application.

This module provides data loading and preprocessing utilities for the assignment.
Uses curated CSV datasets as required by the assignment specifications, containing
historical OHLC data with engineered features (MA_5, Volatility, Avg_Sentiment).
"""

import pandas as pd
import os
from typing import Dict
from pathlib import Path


def load_data(data_dir: str = 'data') -> Dict[str, pd.DataFrame]:
    """
    Load curated CSV datasets for AAPL, MSFT, and BTC-USD instruments.
    
    This function loads the three CSV files specified in the CS4063 assignment
    as curated datasets. Each CSV contains:
    - OHLC data: Open, High, Low, Close
    - Volume information
    - Engineered features: MA_5 (5-day moving average), Volatility, Avg_Sentiment
    
    The datasets are curated offline datasets that serve as the historical data
    source for model training and forecasting, as per assignment requirements
    (no paid APIs, open-source only).
    
    Args:
        data_dir (str): Path to directory containing CSV files. Defaults to 'data'.
    
    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping instrument symbols to DataFrames.
            Keys: 'AAPL', 'MSFT', 'BTC-USD'
            Values: Cleaned DataFrames with datetime index and float OHLC columns
    
    Raises:
        FileNotFoundError: If CSV files are not found in the specified directory.
        ValueError: If CSV format is invalid or missing required columns.
    
    Example:
        >>> data = load_data('data')
        >>> print(data['AAPL'].head())
        >>> print(f"AAPL shape: {data['AAPL'].shape}")
    """
    # Define CSV file mappings (symbol -> filename)
    csv_files = {
        'AAPL': 'AAPL_20250915_185850.csv',
        'MSFT': 'MSFT_20250915_185853.csv',
        'BTC-USD': 'BTC-USD_20250915_185857.csv'
    }
    
    # Required columns for validation
    required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    data = {}
    
    for symbol, filename in csv_files.items():
        filepath = os.path.join(data_dir, filename)
        
        # Check file existence
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"CSV file not found: {filepath}\n"
                f"Please ensure curated datasets are placed in the '{data_dir}' directory."
            )
        
        # Load CSV
        print(f"Loading {symbol} data from {filename}...")
        df = pd.read_csv(filepath)
        
        # Validate required columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {filename}: {missing_cols}\n"
                f"Found columns: {list(df.columns)}"
            )
        
        # Clean and preprocess
        df = clean_dataframe(df, symbol)
        
        data[symbol] = df
        print(f"  [OK] Loaded {len(df)} records for {symbol}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return data


def clean_dataframe(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean and preprocess a single instrument DataFrame.
    
    Performs the following operations:
    1. Parse 'Date' column as datetime and set as index
    2. Convert OHLC columns to float type
    3. Sort by date ascending
    4. Handle missing values (forward fill for features)
    5. Retain all feature columns (MA_5, Volatility, Avg_Sentiment, etc.)
    
    Args:
        df (pd.DataFrame): Raw DataFrame loaded from CSV
        symbol (str): Instrument symbol (for logging/error messages)
    
    Returns:
        pd.DataFrame: Cleaned DataFrame with datetime index and proper dtypes
    """
    df = df.copy()
    
    # Parse Date column as datetime
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Check for invalid dates
        if df['Date'].isna().any():
            invalid_count = df['Date'].isna().sum()
            print(f"  Warning: {invalid_count} invalid dates found in {symbol}, dropping rows")
            df = df.dropna(subset=['Date'])
        
        # Set Date as index
        df.set_index('Date', inplace=True)
    
    # Sort by date ascending
    df.sort_index(inplace=True)
    
    # Convert OHLC columns to float (explicitly float64)
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    for col in ohlc_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
    
    # Convert Volume to numeric (int64 or float64)
    if 'Volume' in df.columns:
        df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')
    
    # Convert feature columns to float (MA_5, Volatility, Avg_Sentiment, etc.)
    # Identify numeric feature columns (exclude OHLCV)
    feature_cols = [col for col in df.columns 
                   if col not in ohlc_columns + ['Volume']]
    
    for col in feature_cols:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values - forward fill for features, drop for OHLC
    # For OHLC: drop rows with any missing values (critical data)
    critical_cols = ['Open', 'High', 'Low', 'Close']
    before_drop = len(df)
    df.dropna(subset=critical_cols, inplace=True)
    after_drop = len(df)
    
    if before_drop > after_drop:
        print(f"  Warning: Dropped {before_drop - after_drop} rows with missing OHLC data")
    
    # For feature columns: forward fill (carry last valid value forward)
    df[feature_cols] = df[feature_cols].fillna(method='ffill')
    
    # If any features still have NaN at the beginning, backfill
    df[feature_cols] = df[feature_cols].fillna(method='bfill')
    
    # Final check: drop any remaining rows with all NaN values
    df.dropna(how='all', inplace=True)
    
    return df


def get_data_info(data: Dict[str, pd.DataFrame]) -> None:
    """
    Print summary information about loaded datasets.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dictionary of DataFrames from load_data()
    """
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    for symbol, df in data.items():
        print(f"\n{symbol}:")
        print(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  Date Range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"  Columns: {', '.join(df.columns.tolist())}")
        print(f"  Memory Usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        # Show sample statistics for Close price
        if 'Close' in df.columns:
            print(f"  Close Price: min=${df['Close'].min():.2f}, "
                  f"max=${df['Close'].max():.2f}, "
                  f"mean=${df['Close'].mean():.2f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    """
    Standalone execution for testing data loading.
    Run: python utils.py
    """
    try:
        # Load data from default location
        data = load_data(data_dir='../data')
        
        # Display summary
        get_data_info(data)
        
        # Show first few rows of each dataset
        print("\nSample Data (first 3 rows):")
        for symbol, df in data.items():
            print(f"\n{symbol}:")
            print(df.head(3))
        
        print("\n[SUCCESS] Data loading successful!")
        
    except Exception as e:
        print(f"[ERROR] Error loading data: {e}")
        raise

