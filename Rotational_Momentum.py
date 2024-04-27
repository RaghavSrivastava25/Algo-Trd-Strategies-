import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Investment universe of 5 ETFs (SPY, EFA, BND, VNQ, GSG)
etfs = ["SPY", "EFA", "BND", "VNQ", "GSG"]

# Fetch historical price data
start_date = '2005-01-01'
end_date = '2022-12-31'
data = yf.download(etfs, start=start_date, end=end_date)['Adj Close']

# Calculate 12-month returns (momentum)
momentum = data.pct_change(252)  # 252 trading days in a year

# Define a DataFrame to hold monthly portfolio selections
portfolio = pd.DataFrame(index=momentum.index, columns=etfs)

# Strategy setup: Select the top 3 ETFs with the highest 12-month momentum
for date in momentum.index[252::21]:  # Start after first 12 months, rebalance monthly (21 trading days)
    top_3 = momentum.loc[date].nlargest(3).index  # Top 3 ETFs
    portfolio.loc[date, top_3] = 1  # Equal-weight

# Forward-fill to maintain positions
portfolio = portfolio.fillna(0)

# Calculate portfolio returns
portfolio_returns = portfolio.shift(1).multiply(data.pct_change()).sum(axis=1)

# Calculate cumulative returns
cumulative_returns = (1 + portfolio_returns).cumprod()

# Plot cumulative returns
plt.figure(figsize=(10, 6))
cumulative_returns.plot(label='Rotational Momentum Strategy')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.title('Backtest of Rotational Momentum Strategy')
plt.legend()
plt.show()

# Calculate Sharpe ratio
risk_free_rate = 0.01  # Assuming a 1% risk-free rate
sharpe_ratio = (portfolio_returns.mean() - risk_free_rate / 252) / portfolio_returns.std() * np.sqrt(252)
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

# Calculate maximum drawdown
cumulative_max = cumulative_returns.cummax()
drawdowns = cumulative_returns / cumulative_max - 1
max_drawdown = drawdowns.min()
print(f"Maximum Drawdown: {max_drawdown:.2f}")

# Calculate annualized volatility
annualized_volatility = portfolio_returns.std() * np.sqrt(252)
print(f"Annualized Volatility: {annualized_volatility:.2f}")
