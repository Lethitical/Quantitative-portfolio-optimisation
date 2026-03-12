import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# -----------------------------
# STEP 1 — Define Stock Universe
# -----------------------------

tickers = [
    # Technology
    'AAPL','MSFT','GOOGL','NVDA','META','TSLA','AVGO','ORCL','AMD','INTC',

    # Finance
    'JPM','BAC','WFC','GS','MS','BLK','AXP','SCHW','C','USB',

    # Healthcare
    'JNJ','UNH','PFE','ABBV','MRK','TMO','ABT','DHR','BMY','AMGN',

    # Consumer
    'AMZN','HD','MCD','NKE','SBUX','TGT','COST','WMT','PG','KO',

    # Energy / Industrial
    'XOM','CVX','CAT','BA','HON','UPS','LMT','RTX','GE','MMM'
]

# -----------------------------
# STEP 2 — Download Price Data
# -----------------------------

print("Downloading stock data...")

data = yf.download(tickers, start="2019-01-01", end="2024-01-01")['Close']

print(f"Downloaded {data.shape[1]} stocks and {data.shape[0]} trading days")

# -----------------------------
# STEP 3 — Calculate Returns
# -----------------------------

returns = data.pct_change().dropna()

mean_returns = returns.mean() * 252
cov_matrix = returns.cov() * 252

num_stocks = len(tickers)

# -----------------------------
# STEP 4 — Simulate Random Portfolios
# -----------------------------

num_portfolios = 5000

results = np.zeros((3, num_portfolios))
weights_record = []

print("Simulating portfolios...")

for i in range(num_portfolios):

    weights = np.random.dirichlet(np.ones(num_stocks))
    weights_record.append(weights)

    portfolio_return = np.dot(weights, mean_returns)

    portfolio_volatility = np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

    sharpe_ratio = portfolio_return / portfolio_volatility

    results[0, i] = portfolio_return
    results[1, i] = portfolio_volatility
    results[2, i] = sharpe_ratio

print("Simulation complete.")

# -----------------------------
# STEP 5 — Find Optimal Portfolio
# -----------------------------

max_sharpe_idx = results[2].argmax()

optimal_weights = weights_record[max_sharpe_idx]

print("\nOptimal Portfolio Allocation\n")

for ticker, weight in zip(tickers, optimal_weights):
    if weight > 0.01:
        print(f"{ticker}: {weight:.2%}")

print("\nOptimal Portfolio Performance")

print(f"Expected Return: {results[0,max_sharpe_idx]:.2%}")
print(f"Volatility: {results[1,max_sharpe_idx]:.2%}")
print(f"Sharpe Ratio: {results[2,max_sharpe_idx]:.2f}")

# -----------------------------
# STEP 6 — Efficient Frontier Curve
# -----------------------------

def portfolio_volatility(weights):

    return np.sqrt(
        np.dot(weights.T, np.dot(cov_matrix, weights))
    )

def portfolio_return(weights):

    return np.dot(weights, mean_returns)

def minimise_volatility(weights):

    return portfolio_volatility(weights)

constraints = (
    {'type':'eq','fun': lambda x: np.sum(x)-1}
)

bounds = tuple((0,1) for _ in range(num_stocks))

efficient_returns = np.linspace(
    mean_returns.min(),
    mean_returns.max(),
    50
)

efficient_volatility = []

for target_return in efficient_returns:

    constraints = (
        {'type':'eq','fun': lambda x: np.sum(x)-1},
        {'type':'eq','fun': lambda x: portfolio_return(x)-target_return}
    )

    result = minimize(
        minimise_volatility,
        num_stocks*[1./num_stocks],
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    efficient_volatility.append(result.fun)

# -----------------------------
# STEP 7 — Plot Results
# -----------------------------

plt.figure(figsize=(12,8))

scatter = plt.scatter(
    results[1,:],
    results[0,:],
    c=results[2,:],
    cmap='viridis',
    s=10,
    alpha=0.6
)

plt.colorbar(scatter,label='Sharpe Ratio')

plt.scatter(
    results[1,max_sharpe_idx],
    results[0,max_sharpe_idx],
    color='red',
    marker='*',
    s=300,
    label='Max Sharpe Portfolio'
)

plt.plot(
    efficient_volatility,
    efficient_returns,
    color='black',
    linewidth=2,
    label='Efficient Frontier'
)

plt.xlabel("Annual Volatility (Risk)")
plt.ylabel("Annual Return")

plt.title("Efficient Frontier — 50 S&P 500 Stocks (2019–2024)")

plt.legend()

plt.tight_layout()

plt.show()

# -----------------------------
# STEP 8 — Benchmark vs S&P 500
# -----------------------------

print("\nDownloading S&P 500 benchmark...")

sp500 = yf.download('^GSPC', start='2019-01-01', end='2024-01-01')['Close']

sp_returns = sp500.pct_change().dropna()

sp_return = sp_returns.mean()*252
sp_vol = sp_returns.std()*np.sqrt(252)

print("\nS&P 500 Performance")

print(f"Annual Return: {sp_return:.2%}")
print(f"Volatility: {sp_vol:.2%}")
print(f"Sharpe Ratio: {(sp_return/sp_vol):.2f}")