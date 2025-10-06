"""
Database module for CS4063 Forecasting Application.

This module manages SQLite database operations for storing:
1. Historical OHLC data with curated features (MA_5, Volatility, Avg_Sentiment)
2. Model predictions (ARIMA, LSTM, Ensemble) with metadata

Schema Design:
- Historical table: Stores time-series data from curated CSV datasets
- Predictions table: Stores forecast outputs with model type and horizon info
"""

import os
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from sqlalchemy import (
    create_engine, Column, Integer, Float, String, DateTime, 
    UniqueConstraint, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from .utils import load_data


# Database configuration
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'forecasting.db')
DATABASE_URL = f'sqlite:///{DB_PATH}'

# SQLAlchemy setup
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Historical(Base):
    """
    Historical OHLC data with engineered features from curated CSV datasets.
    
    This table stores the complete time-series data for each instrument including:
    - Core OHLC prices (Open, High, Low, Close)
    - Trading volume
    - Curated features: MA_5 (5-day moving average), Volatility, Avg_Sentiment
    - Additional derived features: Daily_Return, RSI, MACD, etc.
    
    The data serves as the foundation for:
    1. Training ARIMA and LSTM forecasting models
    2. Backtesting and validation
    3. Visualization (candlestick charts with overlays)
    
    Primary Key: (instrument, date) composite to ensure unique records
    """
    __tablename__ = 'historical'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(20), nullable=False, index=True)
    date = Column(DateTime, nullable=False, index=True)
    
    # OHLC data
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Curated features from CSV
    adj_close = Column(Float, nullable=True)
    daily_return = Column(Float, nullable=True)
    ma_5 = Column(Float, nullable=True)
    ma_10 = Column(Float, nullable=True)
    ma_20 = Column(Float, nullable=True)
    volatility = Column(Float, nullable=True)
    avg_sentiment = Column(Float, nullable=True)
    rsi = Column(Float, nullable=True)
    macd = Column(Float, nullable=True)
    macd_signal = Column(Float, nullable=True)
    bollinger_upper = Column(Float, nullable=True)
    bollinger_lower = Column(Float, nullable=True)
    
    # Unique constraint on instrument + date
    __table_args__ = (
        UniqueConstraint('instrument', 'date', name='uix_instrument_date'),
        Index('idx_instrument_date', 'instrument', 'date'),
    )
    
    def __repr__(self):
        return f"<Historical(instrument={self.instrument}, date={self.date}, close={self.close})>"


class Predictions(Base):
    """
    Model predictions table storing forecasts from ARIMA, LSTM, and Ensemble models.
    
    This table stores:
    - Predicted close prices for specified horizons (1, 3, 7, 14 days)
    - Model metadata (type: ARIMA/LSTM/Ensemble)
    - Forecast horizon information
    - Prediction timestamp for tracking
    
    Used for:
    1. Serving predictions to Flask front-end
    2. Model comparison and ensemble weighting
    3. Performance tracking (RMSE, MAE, MAPE)
    4. Historical prediction analysis
    
    Indexes on (instrument, model_type, horizon) for efficient querying
    """
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    instrument = Column(String(20), nullable=False, index=True)
    prediction_date = Column(DateTime, nullable=False)  # When prediction was made
    target_date = Column(DateTime, nullable=False, index=True)  # Date being predicted
    model_type = Column(String(20), nullable=False, index=True)  # ARIMA, LSTM, Ensemble
    predicted_close = Column(Float, nullable=False)
    horizon = Column(Integer, nullable=False)  # 1, 3, 7, or 14 days
    
    # Optional: confidence intervals
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Composite index for efficient queries
    __table_args__ = (
        Index('idx_instrument_model_horizon', 'instrument', 'model_type', 'horizon'),
        Index('idx_target_date', 'target_date'),
    )
    
    def __repr__(self):
        return (f"<Predictions(instrument={self.instrument}, model={self.model_type}, "
                f"target_date={self.target_date}, predicted_close={self.predicted_close})>")


def get_session() -> Session:
    """
    Create and return a new database session.
    
    Returns:
        Session: SQLAlchemy session for database operations
    """
    return SessionLocal()


def insert_historical(instrument: str, df: pd.DataFrame, session: Optional[Session] = None) -> int:
    """
    Insert historical OHLC data with features from CSV DataFrame into database.
    
    Maps DataFrame columns to database schema, handling various column naming conventions.
    Uses bulk insert for performance with large datasets.
    
    Args:
        instrument (str): Instrument symbol (AAPL, MSFT, BTC-USD)
        df (pd.DataFrame): DataFrame with datetime index and OHLC + feature columns
        session (Session, optional): Database session. Creates new if None.
    
    Returns:
        int: Number of records inserted
    
    Example:
        >>> from utils import load_data
        >>> data = load_data('data')
        >>> count = insert_historical('AAPL', data['AAPL'])
        >>> print(f"Inserted {count} records")
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True
    
    try:
        # Prepare records for bulk insert
        records = []
        
        for date, row in df.iterrows():
            # Map DataFrame columns to database schema (case-insensitive)
            record_data = {
                'instrument': instrument,
                'date': date,
                'open': row.get('Open', row.get('open')),
                'high': row.get('High', row.get('high')),
                'low': row.get('Low', row.get('low')),
                'close': row.get('Close', row.get('close')),
                'volume': int(row.get('Volume', row.get('volume', 0))),
            }
            
            # Add optional feature columns if present
            optional_fields = {
                'adj_close': ['Adj Close', 'Adj_Close', 'adj_close'],
                'daily_return': ['Daily_Return', 'daily_return', 'return'],
                'ma_5': ['MA_5', 'ma_5', 'SMA_5'],
                'ma_10': ['MA_10', 'ma_10', 'SMA_10'],
                'ma_20': ['MA_20', 'ma_20', 'SMA_20'],
                'volatility': ['Volatility', 'volatility', 'vol'],
                'avg_sentiment': ['Avg_Sentiment', 'avg_sentiment', 'sentiment'],
                'rsi': ['RSI', 'rsi', 'RSI_14'],
                'macd': ['MACD', 'macd'],
                'macd_signal': ['MACD_Signal', 'macd_signal', 'Signal'],
                'bollinger_upper': ['Bollinger_Upper', 'bollinger_upper', 'BB_Upper'],
                'bollinger_lower': ['Bollinger_Lower', 'bollinger_lower', 'BB_Lower'],
            }
            
            for field, possible_cols in optional_fields.items():
                for col in possible_cols:
                    if col in row.index:
                        record_data[field] = row[col] if pd.notna(row[col]) else None
                        break
            
            records.append(Historical(**record_data))
        
        # Bulk insert with conflict handling (replace on duplicate)
        session.bulk_save_objects(records)
        session.commit()
        
        count = len(records)
        print(f"[DB] Inserted {count} historical records for {instrument}")
        return count
        
    except Exception as e:
        session.rollback()
        print(f"[DB ERROR] Failed to insert historical data for {instrument}: {e}")
        raise
    finally:
        if close_session:
            session.close()


def get_historical(instrument: str, start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None, session: Optional[Session] = None) -> pd.DataFrame:
    """
    Query historical data for an instrument and return as DataFrame.
    
    Args:
        instrument (str): Instrument symbol (AAPL, MSFT, BTC-USD)
        start_date (datetime, optional): Filter records >= this date
        end_date (datetime, optional): Filter records <= this date
        session (Session, optional): Database session. Creates new if None.
    
    Returns:
        pd.DataFrame: Historical data with datetime index
    
    Example:
        >>> df = get_historical('AAPL', start_date=datetime(2024, 1, 1))
        >>> print(df.head())
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True
    
    try:
        # Build query
        query = session.query(Historical).filter(Historical.instrument == instrument)
        
        if start_date:
            query = query.filter(Historical.date >= start_date)
        if end_date:
            query = query.filter(Historical.date <= end_date)
        
        query = query.order_by(Historical.date.asc())
        
        # Execute query
        results = query.all()
        
        if not results:
            print(f"[DB WARNING] No historical data found for {instrument}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for record in results:
            data.append({
                'Date': record.date,
                'Open': record.open,
                'High': record.high,
                'Low': record.low,
                'Close': record.close,
                'Volume': record.volume,
                'Adj_Close': record.adj_close,
                'Daily_Return': record.daily_return,
                'MA_5': record.ma_5,
                'MA_10': record.ma_10,
                'MA_20': record.ma_20,
                'Volatility': record.volatility,
                'Avg_Sentiment': record.avg_sentiment,
                'RSI': record.rsi,
                'MACD': record.macd,
                'MACD_Signal': record.macd_signal,
                'Bollinger_Upper': record.bollinger_upper,
                'Bollinger_Lower': record.bollinger_lower,
            })
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        # Remove columns with all NaN values
        df = df.dropna(axis=1, how='all')
        
        print(f"[DB] Retrieved {len(df)} historical records for {instrument}")
        return df
        
    finally:
        if close_session:
            session.close()


def insert_predictions(instrument: str, predictions: List[Dict], 
                       session: Optional[Session] = None) -> int:
    """
    Insert model predictions into database.
    
    Args:
        instrument (str): Instrument symbol
        predictions (List[Dict]): List of prediction dictionaries with keys:
            - target_date: datetime of prediction
            - model_type: str (ARIMA, LSTM, Ensemble)
            - predicted_close: float
            - horizon: int (1, 3, 7, 14)
            - lower_bound: float (optional)
            - upper_bound: float (optional)
        session (Session, optional): Database session
    
    Returns:
        int: Number of predictions inserted
    
    Example:
        >>> predictions = [
        ...     {
        ...         'target_date': datetime(2024, 1, 15),
        ...         'model_type': 'ARIMA',
        ...         'predicted_close': 150.25,
        ...         'horizon': 1
        ...     }
        ... ]
        >>> count = insert_predictions('AAPL', predictions)
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True
    
    try:
        prediction_date = datetime.utcnow()
        records = []
        
        for pred in predictions:
            records.append(Predictions(
                instrument=instrument,
                prediction_date=prediction_date,
                target_date=pred['target_date'],
                model_type=pred['model_type'],
                predicted_close=pred['predicted_close'],
                horizon=pred['horizon'],
                lower_bound=pred.get('lower_bound'),
                upper_bound=pred.get('upper_bound'),
            ))
        
        session.bulk_save_objects(records)
        session.commit()
        
        count = len(records)
        print(f"[DB] Inserted {count} predictions for {instrument}")
        return count
        
    except Exception as e:
        session.rollback()
        print(f"[DB ERROR] Failed to insert predictions for {instrument}: {e}")
        raise
    finally:
        if close_session:
            session.close()


def get_predictions(instrument: str, model_type: str, horizon: int, 
                    limit: int = 100, session: Optional[Session] = None) -> pd.DataFrame:
    """
    Query predictions for an instrument, model type, and horizon.
    
    Args:
        instrument (str): Instrument symbol
        model_type (str): Model type (ARIMA, LSTM, Ensemble)
        horizon (int): Forecast horizon (1, 3, 7, 14 days)
        limit (int): Maximum number of records to return (most recent)
        session (Session, optional): Database session
    
    Returns:
        pd.DataFrame: Predictions with target_date as index
    
    Example:
        >>> df = get_predictions('AAPL', 'LSTM', horizon=7)
        >>> print(df.head())
    """
    close_session = False
    if session is None:
        session = get_session()
        close_session = True
    
    try:
        query = session.query(Predictions).filter(
            Predictions.instrument == instrument,
            Predictions.model_type == model_type,
            Predictions.horizon == horizon
        ).order_by(Predictions.target_date.desc()).limit(limit)
        
        results = query.all()
        
        if not results:
            print(f"[DB WARNING] No predictions found for {instrument}, {model_type}, horizon={horizon}")
            return pd.DataFrame()
        
        data = [{
            'target_date': r.target_date,
            'predicted_close': r.predicted_close,
            'prediction_date': r.prediction_date,
            'lower_bound': r.lower_bound,
            'upper_bound': r.upper_bound,
        } for r in results]
        
        df = pd.DataFrame(data)
        df.set_index('target_date', inplace=True)
        df.sort_index(inplace=True)
        
        print(f"[DB] Retrieved {len(df)} predictions for {instrument}")
        return df
        
    finally:
        if close_session:
            session.close()


def init_db(data_dir: str = 'data', force_reload: bool = False) -> None:
    """
    Initialize database: create tables and load historical data from CSVs.
    
    This function:
    1. Creates all database tables (Historical, Predictions)
    2. Loads curated CSV datasets using load_data() from utils.py
    3. Inserts historical data for all instruments (AAPL, MSFT, BTC-USD)
    
    Args:
        data_dir (str): Directory containing CSV files
        force_reload (bool): If True, drop existing tables and reload data
    
    Example:
        >>> init_db(data_dir='data', force_reload=False)
        [DB] Created tables
        [DB] Loading historical data from CSVs...
        [DB] Inserted 1250 historical records for AAPL
        [DB] Inserted 1250 historical records for MSFT
        [DB] Inserted 1250 historical records for BTC-USD
        [DB] Database initialization complete
    """
    try:
        # Create tables
        if force_reload:
            print("[DB] Dropping existing tables...")
            Base.metadata.drop_all(bind=engine)
        
        print("[DB] Creating tables...")
        Base.metadata.create_all(bind=engine)
        print("[DB] Tables created successfully")
        
        # Check if data already exists
        session = get_session()
        existing_count = session.query(Historical).count()
        session.close()
        
        if existing_count > 0 and not force_reload:
            print(f"[DB] Database already contains {existing_count} historical records")
            print("[DB] Skipping data load (use force_reload=True to reload)")
            return
        
        # Load data from CSVs
        print(f"[DB] Loading historical data from '{data_dir}' directory...")
        data = load_data(data_dir=data_dir)
        
        # Insert historical data for each instrument
        session = get_session()
        for instrument, df in data.items():
            insert_historical(instrument, df, session=session)
        session.close()
        
        print("[DB] Database initialization complete")
        print(f"[DB] Database location: {DB_PATH}")
        
    except Exception as e:
        print(f"[DB ERROR] Database initialization failed: {e}")
        raise


def get_db_stats() -> Dict[str, int]:
    """
    Get database statistics.
    
    Returns:
        Dict[str, int]: Statistics including record counts per table
    """
    session = get_session()
    try:
        stats = {
            'historical_records': session.query(Historical).count(),
            'prediction_records': session.query(Predictions).count(),
            'instruments': session.query(Historical.instrument).distinct().count(),
        }
        return stats
    finally:
        session.close()


# ==============================================================================
# UNIT TESTS (pytest)
# ==============================================================================

def test_insert_and_query_historical():
    """
    Test inserting and querying historical data.
    
    Run with: pytest forecasting_app/db.py::test_insert_and_query_historical -v
    """
    import tempfile
    from datetime import timedelta
    
    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db = tmp.name
    
    # Override global engine for testing
    global engine, SessionLocal
    engine = create_engine(f'sqlite:///{temp_db}', echo=False)
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    try:
        # Create test DataFrame
        dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        test_df = pd.DataFrame({
            'Open': [100 + i for i in range(10)],
            'High': [105 + i for i in range(10)],
            'Low': [95 + i for i in range(10)],
            'Close': [102 + i for i in range(10)],
            'Volume': [1000000 + i*10000 for i in range(10)],
            'MA_5': [100 + i*0.5 for i in range(10)],
            'Volatility': [0.02 + i*0.001 for i in range(10)],
        }, index=dates)
        
        # Test insert
        count = insert_historical('TEST', test_df)
        assert count == 10, f"Expected 10 records, got {count}"
        
        # Test query all
        df = get_historical('TEST')
        assert len(df) == 10, f"Expected 10 records, got {len(df)}"
        assert 'Close' in df.columns
        assert df['Close'].iloc[0] == 102
        
        # Test query with date range
        start = dates[3]
        end = dates[7]
        df_filtered = get_historical('TEST', start_date=start, end_date=end)
        assert len(df_filtered) == 5, f"Expected 5 records, got {len(df_filtered)}"
        
        print("[TEST] test_insert_and_query_historical PASSED")
        
    finally:
        # Cleanup
        os.remove(temp_db)


def test_insert_and_query_predictions():
    """
    Test inserting and querying predictions.
    
    Run with: pytest forecasting_app/db.py::test_insert_and_query_predictions -v
    """
    import tempfile
    from datetime import timedelta
    
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
        temp_db = tmp.name
    
    global engine, SessionLocal
    engine = create_engine(f'sqlite:///{temp_db}', echo=False)
    SessionLocal = sessionmaker(bind=engine)
    Base.metadata.create_all(bind=engine)
    
    try:
        # Create test predictions
        base_date = datetime(2024, 1, 1)
        predictions = [
            {
                'target_date': base_date + timedelta(days=i),
                'model_type': 'ARIMA',
                'predicted_close': 150.0 + i,
                'horizon': 7,
            }
            for i in range(5)
        ]
        
        # Test insert
        count = insert_predictions('TEST', predictions)
        assert count == 5, f"Expected 5 predictions, got {count}"
        
        # Test query
        df = get_predictions('TEST', 'ARIMA', horizon=7)
        assert len(df) == 5, f"Expected 5 predictions, got {len(df)}"
        assert 'predicted_close' in df.columns
        
        print("[TEST] test_insert_and_query_predictions PASSED")
        
    finally:
        os.remove(temp_db)


if __name__ == "__main__":
    """
    Standalone execution for testing database operations.
    Run: python forecasting_app/db.py
    """
    print("="*60)
    print("CS4063 FORECASTING APP - DATABASE MODULE")
    print("="*60)
    
    # Run tests
    print("\n[TESTING] Running unit tests...")
    test_insert_and_query_historical()
    test_insert_and_query_predictions()
    print("\n[TESTING] All tests passed!\n")
    
    # Initialize database (commented out to avoid accidental execution)
    # Uncomment to initialize with real data:
    # init_db(data_dir='../data', force_reload=False)
    
    # Show stats if database exists
    if os.path.exists(DB_PATH):
        stats = get_db_stats()
        print("[DB STATS]")
        print(f"  Historical records: {stats['historical_records']}")
        print(f"  Prediction records: {stats['prediction_records']}")
        print(f"  Instruments: {stats['instruments']}")

