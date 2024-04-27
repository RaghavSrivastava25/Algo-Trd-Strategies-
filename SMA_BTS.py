import numpy as np
import pandas as pd
# import quantstats as qs
# import matplotlib.pyplot as plt
import itertools
from multiprocessing import Pool
import json


def read_config(file_path):
    with open(file_path, 'r') as f:
        config = json.load(f)
    return config

def calculate_contango_or_backwardation(data):
    data['C_B'] = np.where(data['Close_L2'] > data['Close_L1'], 'Contango', 
                           np.where(data['Close_L2'] < data['Close_L1'], 'Backwardation', 'Neutral'))
    return data


def calculate_volatility(data):
    volatility = data.groupby('Ticker')['Vol_L1'].std().reset_index()
    return volatility


def inverse_volatility_position_sizing(data, volatility):
    data = pd.merge(data, volatility, on='Ticker', how='left')
    data['Position_Size'] = data['Vol_L1'].mean() / data['Vol_L1']
    return data


def mean_reversion_strategy(data, lookback_period=20, num_std_dev=2):
    data['Spread_Mean'] = data['Spread'].rolling(window=lookback_period).mean()
    data['Spread_Std'] = data['Spread'].rolling(window=lookback_period).std()
    data['Signal'] = 0
    data.loc[data['Spread'] < (data['Spread_Mean'] - num_std_dev * data['Spread_Std']), 'Signal'] = 1  # Buy signal
    data.loc[data['Spread'] > (data['Spread_Mean'] + num_std_dev * data['Spread_Std']), 'Signal'] = -1  # Sell signal
    data['Position'] = data['Signal'].shift(1)
    
    return data

def mean_reversion_strategy_BC_filter(data, lookback_period=20, num_std_dev=2):
    data['Spread_Mean'] = data['Spread'].rolling(window=lookback_period).mean()
    data['Spread_Std'] = data['Spread'].rolling(window=lookback_period).std()
    data['C_B'] = np.where(data['Close_L2'] > data['Close_L1'], 'Contango', 
                           np.where(data['Close_L2'] < data['Close_L1'], 'Backwardation', 'Neutral'))
    data['Signal'] = 0
    data.loc[(data['Spread'] < (data['Spread_Mean'] - num_std_dev * data['Spread_Std'])) & (data['C_B'] == 'Contango'), 'Signal'] = 1  # Buy signal
    data.loc[(data['Spread'] > (data['Spread_Mean'] + num_std_dev * data['Spread_Std'])) & (data['C_B'] == 'Backwardation'), 'Signal'] = -1  # Sell signal
    data['Position'] = data['Signal'].shift(1)
    
    return data



def moving_average_crossover_strategy_BC_filter(data, short_window=1, long_window=26):
    data['SMA_short'] = data.groupby('Ticker')['Continuous_Close'].rolling(window=short_window, min_periods=1).mean().reset_index(level=0, drop=True)
    data['SMA_long'] = data.groupby('Ticker')['Continuous_Close'].rolling(window=long_window, min_periods=1).mean().reset_index(level=0, drop=True)
    data['Signal'] = 0
    data.loc[(data['SMA_short'] < data['SMA_long']) & (data['C_B'] == 'Contango'), 'Signal'] = -1  # Sell signal
    data.loc[(data['SMA_short'] > data['SMA_long']) & (data['C_B'] == 'Backwardation'), 'Signal'] = 1  # Buy signal
    data['Position'] = data.groupby('Ticker')['Signal'].shift(1)

    return data


def moving_average_crossover_strategy(data, short_window=4, long_window=13):
    data['SMA_short'] = data.groupby('Ticker')['Continuous_Close'].rolling(window=short_window, min_periods=1).mean().reset_index(level=0, drop=True)
    data['SMA_long'] = data.groupby('Ticker')['Continuous_Close'].rolling(window=long_window, min_periods=1).mean().reset_index(level=0, drop=True)
    data['Signal'] = 0
    data.loc[data['SMA_short'] > data['SMA_long'], 'Signal'] = 1  # Short MA crosses above Long MA
    data.loc[data['SMA_short'] < data['SMA_long'], 'Signal'] = -1  # Short MA crosses below Long MA

    data['Position'] = data.groupby('Ticker')['Signal'].shift(1)

    return data


def backtest_strategy_sma_lsa(data, initial_capital=1000000, notional_per_trade=500000, stop_loss=None):
    capital = initial_capital
    daily_results = []
    position_open = {}

    for index, row in data.iterrows():
        ticker = row['Ticker']
        if ticker not in position_open:
            position_open[ticker] = False

        if row['Position'] == 1 and not position_open[ticker]:  # Buy signal
            entry_date = row['Date']
            entry_price = row['Continuous_Close']
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and position_open[ticker]:  # Sell signal with open position
            exit_date = row['Date']
            exit_price = row['Continuous_Close']
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
            
        # Check for stop loss on a daily basis
        if stop_loss is not None and position_open[ticker]:
            current_price = row['Continuous_Close']
            if row['Position'] == 1:  # Long position
                if entry_price - current_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price - stop_loss
                    pnl = lot_size * (exit_price - entry_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
            elif row['Position'] == -1:  # Short position
                if current_price - entry_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price + stop_loss
                    pnl = lot_size * (entry_price - exit_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
                    
    return daily_results


def backtest_strategy_sma_lsa_bc(data, initial_capital=1000000, notional_per_trade=500000, stop_loss=None):
    capital = initial_capital
    daily_results = []
    position_open = {}

    for index, row in data.iterrows():
        ticker = row['Ticker']
        if ticker not in position_open:
            position_open[ticker] = False

        if row['Position'] == 1 and not position_open[ticker] and row['C_B'] == 'Backwardation':  # Buy signal in backwardation
            entry_date = row['Date']
            entry_price = row['Continuous_Close']
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and not position_open[ticker] and row['C_B'] == 'Contango':  # Short signal in contango
            entry_date = row['Date']
            entry_price = row['Continuous_Close']
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and position_open[ticker] and row['C_B'] == 'Backwardation':  # Closing short position in backwardation
            exit_date = row['Date']
            exit_price = row['Continuous_Close']
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
        elif row['Position'] == 1 and position_open[ticker] and row['C_B'] == 'Contango':  # Closing long position in contango
            exit_date = row['Date']
            exit_price = row['Continuous_Close']
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
            
        # Check for stop loss on a daily basis
        if stop_loss is not None and position_open[ticker]:
            current_price = row['Continuous_Close']
            if row['Position'] == 1:  # Long position
                if entry_price - current_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price - stop_loss
                    pnl = lot_size * (exit_price - entry_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
            elif row['Position'] == -1:  # Short position
                if current_price - entry_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price + stop_loss
                    pnl = lot_size * (entry_price - exit_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
                    
    return daily_results



def backtest_strategy_mean_reversion(data, initial_capital=1000000, notional_per_trade=500000, stop_loss=None):
    capital = initial_capital
    daily_results = []
    position_open = {}

    for index, row in data.iterrows():
        ticker = row['Ticker']
        if ticker not in position_open:
            position_open[ticker] = False

        # Enter a trade if the spread is below the mean
        if row['Position'] == 1 and not position_open[ticker]:  # Buy signal
            entry_date = row['Date']
            entry_price = row['Spread']  # Use spread as entry price
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and position_open[ticker]:  # Sell signal with open position
            exit_date = row['Date']
            exit_price = row['Spread']  # Use spread as exit price
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
            
        # Check for stop loss on a daily basis
        if stop_loss is not None and position_open[ticker]:
            current_price = row['Spread']  # Use spread as current price
            if row['Position'] == 1:  # Long position
                if entry_price - current_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price - stop_loss
                    pnl = lot_size * (exit_price - entry_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
            elif row['Position'] == -1:  # Short position
                if current_price - entry_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price + stop_loss
                    pnl = lot_size * (entry_price - exit_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
                    
    return daily_results


def backtest_strategy_mean_reversion_bc(data, initial_capital=1000000, notional_per_trade=500000, stop_loss=None):
    capital = initial_capital
    daily_results = []
    position_open = {}

    for index, row in data.iterrows():
        ticker = row['Ticker']
        if ticker not in position_open:
            position_open[ticker] = False

        if row['Position'] == 1 and not position_open[ticker] and row['C_B'] == 'Backwardation':  # Buy signal in backwardation
            entry_date = row['Date']
            entry_price = row['Spread']  # Use spread as entry price
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and not position_open[ticker] and row['C_B'] == 'Contango':  # Short signal in contango
            entry_date = row['Date']
            entry_price = row['Spread']  # Use spread as entry price
            lot_size = calculate_lot_size(notional_per_trade, entry_price, capital)
            lot_value = entry_price * lot_size
            if lot_value <= capital:
                capital -= lot_value
                daily_results.append([ticker, entry_date, None, entry_price, None, lot_size, None, capital])
                position_open[ticker] = True  # Set position to open
        elif row['Position'] == -1 and position_open[ticker] and row['C_B'] == 'Backwardation':  # Closing short position in backwardation
            exit_date = row['Date']
            exit_price = row['Spread']  # Use spread as exit price
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
        elif row['Position'] == 1 and position_open[ticker] and row['C_B'] == 'Contango':  # Closing long position in contango
            exit_date = row['Date']
            exit_price = row['Spread']  # Use spread as exit price
            pnl = lot_size * (exit_price - entry_price)  # Calculate PnL
            capital += pnl + lot_value
            if stop_loss is not None and (entry_price - exit_price) > stop_loss:
                exit_price = entry_price - stop_loss
                pnl = lot_size * (exit_price - entry_price)
                capital += pnl + lot_value

            daily_results[-1][2] = exit_date
            daily_results[-1][4] = exit_price
            daily_results[-1][6] = pnl
            daily_results[-1][7] = capital
            position_open[ticker] = False
            
        # Check for stop loss on a daily basis
        if stop_loss is not None and position_open[ticker]:
            current_price = row['Spread']  # Use spread as current price
            if row['Position'] == 1:  # Long position
                if entry_price - current_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price - stop_loss
                    pnl = lot_size * (exit_price - entry_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
            elif row['Position'] == -1:  # Short position
                if current_price - entry_price > stop_loss:
                    exit_date = row['Date']
                    exit_price = entry_price + stop_loss
                    pnl = lot_size * (entry_price - exit_price)
                    capital += pnl + lot_value

                    daily_results[-1][2] = exit_date
                    daily_results[-1][4] = exit_price
                    daily_results[-1][6] = pnl
                    daily_results[-1][7] = capital
                    position_open[ticker] = False
                    
    return daily_results


def summary_stats(daily_results):
    df = pd.DataFrame(daily_results, columns=['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
    initial_capital = df['Capital'].iloc[0]
    capital = initial_capital
    df['PnL'] = df['PnL'].astype(float)

    df['Entry_Date'] = pd.to_datetime(df['Entry_Date'])
    df['Exit_Date'] = pd.to_datetime(df['Exit_Date'])
    df['turnover'] = (df['Entry_Price'] + df['Exit_Price']) * df['Lot_Size']
    df['sellturnover'] = df['Entry_Price'] * df['Lot_Size']

    df.sort_values(by='Entry_Date', ascending=True, inplace=True)

    df['Capital'] = capital + df['PnL'].cumsum()
    sumdf = pd.DataFrame(df.groupby(df['Entry_Date'].dt.date)['PnL'].sum())
    sumdf['Capital'] = sumdf['PnL'].cumsum() + capital
    sumdf['Return'] = sumdf['Capital'].pct_change()
    sumdf['Return'].iloc[0] = (sumdf['Capital'].iloc[0] - capital) / capital
    sumdf.index = pd.to_datetime(sumdf.index)

    # qs.reports.html(sumdf['Return'], title='Strategy Summary', output='strategy_summary.html')
    sumdf.to_csv('daily_summary.csv')
    df.to_csv('trade_sheet.csv')


def calculate_lot_size(notional_per_trade, entry_price, capital):
    if entry_price == 0:
        return 1  # Set lot size to 1 if entry price is 0
    else:
        lot_size = int(notional_per_trade / entry_price)
        return lot_size



def backtest_single_strategy(params):
    short, long = params
    data = pd.read_csv(r"/Users/raghavsrivastava/Desktop/commodities_special_HLCV.csv")
    data = calculate_contango_or_backwardation(data)
    data = moving_average_crossover_strategy_BC_filter(data, short_window=short, long_window=long)
    results = backtest_strategy_sma_lsa_bc(data)
    res_df = pd.DataFrame(results, columns=['Ticker','Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
    res_df['Return'] = res_df['PnL'] / res_df['Capital'].shift(1)  # Calculate return
    if len(res_df['Capital']) == 0:
        final_capital = 0
    else:
        final_capital = res_df['Capital'].iloc[-1]
    return res_df['Return'].sum(), final_capital, res_df


def main_backtest_param_opt(parameter_combinations):
    with Pool() as pool:
        results = pool.map(backtest_single_strategy, parameter_combinations)

    # Find the index of the parameter set with the highest total return
    best_index = max(range(len(results)), key=lambda i: results[i][1])
    best_params = parameter_combinations[best_index]
    best_final_capital = results[best_index][0]

    print("Best parameter set:", best_params)
    print("Final capital for best parameter set:", best_final_capital)
    

def main_backtest():

    data = pd.read_csv(r"C:\Users\t0013\Desktop\Commodities_Project\resources\csv_xlsx\Special_Capiq_WA_CV.csv")
    print(data.columns)
    data = calculate_contango_or_backwardation(data)
    data = moving_average_crossover_strategy(data)
    print(data.head(5))
    results = backtest_strategy_sma_lsa(data)
    print(results[:5])
    res_df = pd.DataFrame(results, columns=['Ticker','Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
    print(res_df)
    summary_stats(res_df)


def run_backtests(param_sets):
    with Pool() as pool:
        results = pool.map(backtest_single_strategy, param_sets)
    return results

def generate_report(strategy_name, param_sets, results, res_dfs):
    plt.figure(figsize=(10, 6))
    for i, (params, res_df) in enumerate(zip(param_sets, res_dfs)):
        plt.plot(res_df['Entry_Date'], res_df['Capital'], label=f'Params: {params}')
    
    start_date = min(df['Entry_Date'].min() for df in res_dfs)
    end_date = max(df['Entry_Date'].max() for df in res_dfs)
    
    plt.title(f'Capital for {strategy_name} with Different Parameter Sets')
    plt.xlabel('Date')
    plt.ylabel('Capital')
    plt.legend()
    plt.grid(True)
    plt.show()
    


    
def main():
    config = read_config("/Users/raghavsrivastava/Desktop/SBTS_config.json")
    strategy = config["strategy"]
    strategy_params = config.get("strategy_params", {})
    backtest_params = config.get("backtest_params", {})
    commodity = config.get("commodity")

    data = pd.read_csv(r"/Users/raghavsrivastava/Desktop/commodities_special_HLCV.csv")
    data = calculate_contango_or_backwardation(data)
    
    if commodity:
        data = data[data['Ticker'] == commodity]

    if strategy == "sma_lsa_bc":
        # Run backtest for SMA with long-short approach and backwardation-contango filter
        data = moving_average_crossover_strategy_BC_filter(data, short_window=strategy_params.get("short_window", 1), long_window=strategy_params.get("long_window", 26))
        results = backtest_strategy_sma_lsa_bc(data, initial_capital=backtest_params.get("initial_capital", 1000000), notional_per_trade=backtest_params.get("notional_per_trade", 500000), stop_loss=backtest_params.get("stop_loss", None))
        res_df = pd.DataFrame(results, columns=['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
        summary_stats(res_df)
        
    elif strategy == "sma_lsa":
        # Run backtest for SMA with long-short approach
        data = moving_average_crossover_strategy(data, short_window=strategy_params.get("short_window", 4), long_window=strategy_params.get("long_window", 13))
        results = backtest_strategy_sma_lsa(data, initial_capital=backtest_params.get("initial_capital", 1000000), notional_per_trade=backtest_params.get("notional_per_trade", 500000), stop_loss=backtest_params.get("stop_loss", None))
        res_df = pd.DataFrame(results, columns=['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
        summary_stats(res_df)
        
    elif strategy == "mean_reversion_bc":
        # Run backtest for Mean Reversion with backwardation-contango filter
        data = mean_reversion_strategy_BC_filter(data, lookback_period=strategy_params.get("lookback_period", 20), num_std_dev=strategy_params.get("num_std_dev", 2))
        results = backtest_strategy_mean_reversion_bc(data, initial_capital=backtest_params.get("initial_capital", 1000000), notional_per_trade=backtest_params.get("notional_per_trade", 500000), stop_loss=backtest_params.get("stop_loss", None))
        res_df = pd.DataFrame(results, columns=['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
        summary_stats(res_df)
        
    elif strategy == "mean_reversion":
        # Run backtest for Mean Reversion
        data = mean_reversion_strategy(data, lookback_period=strategy_params.get("lookback_period", 20), num_std_dev=strategy_params.get("num_std_dev", 2))
        results = backtest_strategy_mean_reversion(data, initial_capital=backtest_params.get("initial_capital", 1000000), notional_per_trade=backtest_params.get("notional_per_trade", 500000), stop_loss=backtest_params.get("stop_loss", None))
        res_df = pd.DataFrame(results, columns=['Ticker', 'Entry_Date', 'Exit_Date', 'Entry_Price', 'Exit_Price', 'Lot_Size', 'PnL', 'Capital'])
        summary_stats(res_df)
        
    else:
        print("Invalid strategy specified in the config file.")

if __name__ == "__main__":
    main()



 