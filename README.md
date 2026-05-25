# Quantitative Portfolio Optimisation

Mean-variance portfolio optimisation across 50 S&P 500 constituents (2019–2024).
Simulates 5,000 random portfolios, plots the efficient frontier, and identifies
the maximum Sharpe ratio portfolio.

## What It Does

1. Downloads 5 years of price data for 50 S&P 500 stocks via `yfinance`
2. Computes historical returns and the covariance matrix
3. Runs a Monte Carlo simulation of 5,000 randomly weighted portfolios
4. Calculates annualised return, volatility, and Sharpe ratio for each
5. Identifies the maximum Sharpe ratio and minimum variance portfolios
6. Plots the efficient frontier with portfolios colour-mapped by Sharpe ratio

## Methodology

Each simulated portfolio is evaluated on:

$$R_p = \mathbf{w}^\top \boldsymbol{\mu}$$

$$\sigma_p = \sqrt{\mathbf{w}^\top \boldsymbol{\Sigma} \mathbf{w}}$$

$$\text{Sharpe} = \frac{R_p - r_f}{\sigma_p}$$

| Symbol | Meaning |
|--------|---------|
| $\mathbf{w}$ | Portfolio weight vector |
| $\boldsymbol{\mu}$ | Vector of mean annualised returns |
| $\boldsymbol{\Sigma}$ | Covariance matrix of returns |
| $r_f$ | Risk-free rate (US 10Y Treasury) |

## Results

| Portfolio | Annualised Return | Volatility | Sharpe Ratio |
|-----------|-------------------|------------|--------------|
| Max Sharpe | 26.89% | 22.29% | 1.21 |
| S&P 500 benchmark | ~15% | ~17% | ~0.88 |
## Limitations & Extensions

Monte Carlo simulation converges on good portfolios but doesn't guarantee
the true optimum. Natural extensions include:

- **Scipy optimisation**: direct maximisation of Sharpe ratio using
  `scipy.optimize.minimize` for an exact solution
- **Constraints**: sector caps, position limits, no-short-selling
- **Shrinkage estimators**: Ledoit-Wolf covariance shrinkage to reduce
  estimation error on the covariance matrix with limited data
- **Rolling window**: re-optimise quarterly to account for non-stationarity

## Stack

`numpy` · `pandas` · `scipy` · `matplotlib` · `yfinance`

## Run

```bash
