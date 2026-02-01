# FinTech Data Curation for Next-Day Price Prediction

CS4063 Natural Language Processing - Assignment 1

## Overview

This project implements a comprehensive financial data scraper that collects both structured (numerical) and unstructured (textual) data for predicting next-day stock and cryptocurrency prices.

## Features

- **Structured Data**: Historical prices, volume, technical indicators (moving averages, volatility, Bollinger Bands)
- **Unstructured Data**: News headlines, sentiment analysis, market coverage metrics
- **Multi-Asset Support**: Stocks and cryptocurrencies
- **Robust Design**: Exception handling and fallback mechanisms
- **Export Formats**: CSV and JSON output

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Batch Processing (Pre-configured)

```bash
python financial_data_scraper.py
```

Processes AAPL, MSFT stocks and BTC-USD cryptocurrency.

### Interactive Mode

```bash
python interactive_scraper.py
```

Allows manual input of exchange and symbol.

## Sample Output

The scraper generates CSV files with 27 features including:

- Price data (OHLCV)
- Technical indicators (returns, moving averages, volatility)
- Sentiment metrics (news sentiment, coverage volume)
- Next-day targets for evaluation

## Sample Results

- **AAPL (NASDAQ)**: 29 records with real-time news sentiment
- **MSFT (NYSE)**: 29 records with market indicators
- **BTC-USD (Crypto)**: 29 records with cryptocurrency-specific metrics

## Files Generated

- `{SYMBOL}_{TIMESTAMP}.csv` - Main dataset
- Sample files provided: `AAPL_20250915_185850.csv`, `MSFT_20250915_185853.csv`, `BTC-USD_20250915_185857.csv`

## Architecture

- `StructuredDataScraper`: Yahoo Finance API integration
- `UnstructuredDataScraper`: RSS feeds and sentiment analysis
- `DataProcessor`: Data alignment and export functionality

## Error Handling

- Network timeout protection
- Fallback synthetic data generation
- Graceful handling of API changes
- Missing data imputation
