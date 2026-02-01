Feature Selection Justification for Next-Day Stock Price Prediction

Name: Saif ur Rehman
Roll No: 22i-8767
Section: SE-A
Course: CS4063 – Natural Language Processing
Date: September 18, 2025

Executive Summary

This report presents the justification for selecting a concise yet comprehensive feature set for predicting next-day stock prices. The approach combines structured market data with unstructured news sentiment, aiming to capture both quantitative signals and qualitative investor psychology. By balancing predictive accuracy with computational efficiency, the selected features provide a practical foundation for short-term price forecasting.

Selected Feature Architecture

1. Structured Features (Market Data)

Core OHLCV Data: Fundamental open, high, low, close, and volume statistics.

Technical Indicators: 5-day and 20-day moving averages, daily returns, and 20-day rolling volatility (σ).

Market Microstructure Metrics: Relative volume, intraday price positioning, and Bollinger Band indicators.

2. Unstructured Features (News Sentiment)

Sentiment Analytics: Average sentiment score, sentiment volatility, and ratio of positive to negative news.

Information Flow: News volume and recency-weighted headline content.

Justification for Feature Minimality

The feature set is deliberately minimal, ensuring parsimony without sacrificing predictive power. Each feature contributes a unique dimension:

Price Dynamics: OHLCV reflects the underlying supply-demand balance.

Temporal Patterns: Short- and medium-term moving averages detect momentum shifts.

Volatility Regimes: Rolling standard deviation and Bollinger Bands quantify uncertainty.

Sentiment Influence: News sentiment adds a forward-looking psychological component.

By avoiding overlapping signals and multicollinearity, the configuration efficiently covers the three pillars of short-term stock prediction: trend, volatility, and sentiment.

Sufficiency for Next-Day Prediction
Technical Foundation

Next-day stock prices are strongly influenced by overnight information and opening gap dynamics. The chosen structured features address:

Momentum persistence (captured by moving averages).

Volatility clustering (via rolling standard deviation).

Volume confirmation (through relative volume and intraday metrics).

These collectively explain short-term price continuation or reversal.

Behavioral Finance Integration

At 24-hour horizons, markets tend to incorporate sentiment gradually. News-driven features capture:

Delayed information absorption (investors reacting over time).

Uncertainty assessment (via sentiment volatility).

Directional bias (through positive news ratio).

This bridges technical signals with investor psychology, making predictions more robust.

Empirical Validation

Research consistently shows that blending technical indicators with sentiment analysis outperforms purely quantitative models. The 24-hour prediction window offers the ideal balance—long enough for sentiment to influence trading behavior, yet short enough to preserve technical signal integrity.

Implementation Considerations

Data Sources:

Yahoo Finance for reliable historical and real-time OHLCV data.

RSS news feeds for structured text suitable for sentiment analysis.

Synthetic data fallback to ensure resilience and enable backtesting.

Parameter Optimization:

20-day lookback window balances responsiveness with stability.

TextBlob sentiment scoring provides accurate yet efficient NLP analysis.

Temporal alignment ensures news data and market data reflect the same state.

Conclusion

The selected feature set integrates the strengths of technical analysis and sentiment analysis, covering the key forces driving next-day stock movements. By combining momentum, volatility, and investor psychology, it provides comprehensive market insights while avoiding overfitting. The features are grounded in theory, supported by empirical evidence, and computationally efficient—making them well-suited for real-time trading systems.

References

Bollen, J., Mao, H., & Zeng, X. (2011). Twitter mood predicts the stock market. Journal of Computational Science, 2(1), 1–8.

Tetlock, P. C. (2007). Giving content to investor sentiment: The role of media in the stock market. The Journal of Finance, 62(3), 1139–1168.

Fama, E. F., & French, K. R. (1988). Permanent and temporary components of stock prices. Journal of Political Economy, 96(2), 246–273.

Murphy, J. J. (1999). Technical Analysis of the Financial Markets. New York Institute of Finance.
