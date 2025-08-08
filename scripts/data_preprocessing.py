"""
Data Preprocessing and EDA for Time Series Forecasting
- Downloads data for TSLA, BND, SPY using yfinance
- Cleans and preprocesses the data
- Performs EDA: plots, statistics, volatility, outlier detection, stationarity tests, risk metrics
"""
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

TICKERS = ['TSLA', 'BND', 'SPY']
START_DATE = '2015-01-01'
END_DATE = None  # Use latest available

def download_data(tickers, start, end=None):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=True)
    return data

def preprocess_data(data):
    # Extract Close prices
    close = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in TICKERS})
    # Check for missing values
    close = close.ffill().bfill()
    return close

def calculate_daily_returns(close):
    return close.pct_change().dropna()

def plot_closing_prices(close):
    close.plot(figsize=(12,6))
    plt.title('Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_daily_returns(returns):
    returns.plot(figsize=(12,6))
    plt.title('Daily Percentage Change')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_rolling_stats(close, window=21):
    rolling_mean = close.rolling(window).mean()
    rolling_std = close.rolling(window).std()
    plt.figure(figsize=(12,6))
    for ticker in close.columns:
        plt.plot(close[ticker], label=f'{ticker} Close')
        plt.plot(rolling_mean[ticker], label=f'{ticker} {window}d MA')
        plt.fill_between(rolling_std.index, rolling_mean[ticker]-rolling_std[ticker], rolling_mean[ticker]+rolling_std[ticker], alpha=0.2)
    plt.title(f'Rolling Mean and Std (window={window})')
    plt.legend()
    plt.tight_layout()
    plt.show()

def detect_outliers(returns, threshold=3):
    z_scores = (returns - returns.mean())/returns.std()
    outliers = (np.abs(z_scores) > threshold)
    return outliers

def adf_test(series):
    result = adfuller(series.dropna())
    return {'ADF Statistic': result[0], 'p-value': result[1], 'Critical Values': result[4]}

def value_at_risk(returns, confidence=0.05):
    return returns.quantile(confidence)

def sharpe_ratio(returns, risk_free_rate=0.0):
    return (returns.mean() - risk_free_rate) / returns.std()

def main():
    data = download_data(TICKERS, START_DATE, END_DATE)
    close = preprocess_data(data)
    returns = calculate_daily_returns(close)
    print('Basic Statistics:')
    print(close.describe())
    print('\nMissing values per ticker:')
    print(close.isnull().sum())
    plot_closing_prices(close)
    plot_daily_returns(returns)
    plot_rolling_stats(close)
    outliers = detect_outliers(returns)
    print('\nOutlier days:')
    print(returns[outliers.any(axis=1)])
    for ticker in close.columns:
        print(f'\nADF Test for {ticker} Close:')
        print(adf_test(close[ticker]))
        print(f'ADF Test for {ticker} Returns:')
        print(adf_test(returns[ticker]))
        print(f'Value at Risk (5%) for {ticker}: {value_at_risk(returns[ticker])}')
        print(f'Sharpe Ratio for {ticker}: {sharpe_ratio(returns[ticker])}')

if __name__ == '__main__':
    main()
