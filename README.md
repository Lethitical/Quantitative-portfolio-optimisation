# Quantitative Portfolio Optimisation

## Overview

This project implements a mean-variance portfolio optimisation model using Python.

Using historical data from 50 S&P 500 companies (2019–2024), the script:

- Calculates stock returns and covariance
- Simulates 5000 random portfolios
- Computes return, volatility, and Sharpe ratio
- Identifies the optimal portfolio
- Visualises the Efficient Frontier

## Methodology

Portfolio performance is calculated using:

Expected Return

R_p = w^T μ

Portfolio Volatility

σ_p = sqrt(w^T Σ w)

Where:

- w = portfolio weights
- μ = mean returns
- Σ = covariance matrix

## Results

The optimal portfolio achieved a higher Sharpe ratio than the S&P 500 benchmark.

## Libraries Used

- NumPy
- Pandas
- Matplotlib
- yfinance
- SciPy

## How to Run
