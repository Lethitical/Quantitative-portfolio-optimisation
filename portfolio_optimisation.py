import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize


# 50 S&P 500 stocks across different sectors
tickers = [
    # Technology
    'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'TSLA', 'AVGO', 'ORCL', 'AMD', 'INTC',
    # Finance
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'SCHW', 'C', 'USB',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN',
    # Consumer
    'AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'COST', 'WMT', 'PG', 'KO',
    # Energy & Industrial
    'XOM', 'CVX', 'CAT', 'BA', 'HON', 'UPS', 'LMT', 'RTX', 'GE', 'MMM'
]

# Downloading 5 years of daily closing prices
print("Downloading stock data...")
data = yf.download(tickers, start='2019-01-01', end='2024-01-01')['Close']

print(f"Downloaded {data.shape[1]} stocks and {data.shape[0]} days of data")
print(data.head())