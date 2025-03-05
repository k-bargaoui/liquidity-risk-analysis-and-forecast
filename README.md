# liquidity-risk-analysis-and-forecast
Liquidity risk analyzing toolkit for ETFs and single stocks. A bid-ask forecaster using XGboost and LinearRegssion is also included

This repository provides a set of functions to compute various liquidity and market impact metrics. These metrics can be useful for analyzing the liquidity of financial instruments, assessing market conditions, and quantifying the costs of executing trades.

Metrics Computed
The following liquidity and market impact metrics are computed by this project:

Volatility Metrics

Rolling volatility based on a specified window.
Exponentially weighted moving average (EWMA) volatility.
Liquidity Metrics

Average Spread: The average difference between the high and low prices.
Amihud Illiquidity: Measures illiquidity based on price changes and volume.
Kyle's Lambda: Price impact based on the relationship between returns and volume.
Pastor-Stambaugh Liquidity Measure: Measures liquidity based on return and volume changes.
Turnover Ratio: Ratio of average volume to market capitalization.
Market Impact Metrics

Market Impact (Linear, Non-Linear, Almgren-Chriss): Measures market impact based on different models.
Execution Cost: Measures the cost of executing a trade based on spread and impact cost.
Price Impact Measures

Roll’s Impact: Measures price impact based on autocovariance of price changes.
Corwin-Schultz Spread: Spread estimator using high/low ratios.
Hasbrouck Lambda: A refined price impact measure.
Other Metrics

Order Book Imbalance: Measures the imbalance between bid and ask volumes.
Market Resilience: Measures how fast spreads close after widening.
Hurst Exponent: Measures the persistence of a time series.
