import matplotlib.pyplot as plt
import xorbits
import xorbits.pandas as xpd
import pandas as pd
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")


def zscore(df):
    #
    return ((df - df.mean()) / df.std()).iloc[-1]


def stock_benchmark():
    data_path = '../test_csv/stock.csv'
    xorbits.init()

    p_stocks_df = pd.read_csv(data_path).iloc[:10000]
    x_stocks_df = xpd.read_csv(data_path, chunk_bytes=200 * 1024 * 1024).iloc[:10000]
    xorbits.run(x_stocks_df)

    # Calculating Moving Averages for Stock Prices
    start_time = time.time()
    p_stock_mean_df = p_stocks_df.groupby(
        ['StockName']).rolling(window=10).mean(numeric_only=True)[['Open', 'High', 'Low', 'Close', 'Volume']]
    end_time = time.time()
    print("Pandas Calculating Moving Averages for Stock Prices Execution time: ", end_time - start_time)

    start_time = time.time()
    x_stock_mean_df = x_stocks_df.groupby(
        'StockName', group_keys=True
    ).apply(lambda x: x.rolling(window=10).mean(numeric_only=True))[['Open', 'High', 'Low', 'Close', 'Volume']]
    xorbits.run(x_stock_mean_df)
    end_time = time.time()
    print("Xorbits Calculating Moving Averages for Stock Prices Execution time: ", end_time - start_time)

    # Returns
    start_time = time.time()
    p_stocks_df['Ret'] = p_stocks_df[['StockName', 'Close']].groupby(['StockName']).pct_change()
    end_time = time.time()
    print("Pandas Returns Execution time: ", end_time - start_time)

    start_time = time.time()
    x_stocks_ret_df = x_stocks_df[['StockName', 'Close']].groupby(
        'StockName', group_keys=True
    ).apply(lambda x: x.pct_change())

    x_stocks_ret_df = x_stocks_ret_df.reset_index(level=0)
    x_stocks_df['Ret'] = x_stocks_ret_df['Close']
    xorbits.run(x_stocks_ret_df)
    end_time = time.time()
    print("Xorbits Returns Execution time: ", end_time - start_time)

    # Calculating the Moving Annualized Volatility of Stock Returns
    start_time = time.time()
    p_Volatility = p_stocks_df[['StockName', 'Ret']].groupby(['StockName']).rolling(window=10).std() * np.sqrt(252)
    end_time = time.time()
    print("Pandas Calculating the Moving Annualized Volatility of Stock Returns Execution time: ",
          end_time - start_time)

    start_time = time.time()
    x_Volatility = x_stocks_df[['StockName', 'Ret']].groupby(
        'StockName', group_keys=True
    ).apply(lambda x: x.rolling(window=10).std()) * np.sqrt(252)

    xorbits.run(x_Volatility)
    end_time = time.time()
    print("Xorbits Calculating the Moving Annualized Volatility of Stock Returns Execution time: ",
          end_time - start_time)

    # Window Normalization
    start_time = time.time()
    tmp = p_stocks_df.groupby(['StockName'])['Ret'].rolling(window=10).apply(zscore).reset_index(level=0)
    p_stocks_df['Rolling_Nor_Ret'] = tmp['Ret']
    end_time = time.time()
    print("Pandas Window Normalization Execution time: ", end_time - start_time)

    start_time = time.time()
    tmp = x_stocks_df[['StockName', 'Ret']].groupby(
        'StockName', group_keys=True
    )['Ret'].apply(lambda x: x.rolling(window=10).apply(zscore)).reset_index()
    xorbits.run(tmp)
    x_stocks_df['Rolling_Nor_Ret'] = tmp['Ret']
    end_time = time.time()
    print("Xorbits Window Normalization Execution time: ", end_time - start_time)

    # Test equal
    pd.testing.assert_frame_equal(p_stocks_df, x_stocks_df.to_pandas())
    print("Xorbits, Pandas, The same result")


def stock_minute():
    stock_df = pd.read_parquet('data.pq')

    # Calculate the average price and volume per stock for each date
    grouped_df = stock_df.groupby(
        [stock_df['symbol'], stock_df['date_time'].dt.date]
    ).agg({'price': 'mean', 'volume': 'sum'})

    # Calculate the average price and volume per stock for each hour
    hourly_data = stock_df.groupby(
        [stock_df['symbol'], stock_df['date_time'].dt.floor('H')]
    ).agg({'price': 'mean', 'volume': 'sum'})

    # Set the date_time column as the time series index
    stock_df.set_index('date_time', inplace=True)

    # Example 1: Calculate the hourly price change rate for each stock
    start_time = time.time()
    hourly_data['price_change'] = hourly_data.groupby('symbol')['price'].pct_change()
    end_time = time.time()
    print("Pandas Calculate the hourly price change rate for each stock Execution time: ", end_time - start_time)

    # Example 2: Create price labels based on price change
    hourly_data['price_label'] = hourly_data['price_change'].apply(
        lambda x: 'up' if x > 0 else 'down' if x < 0 else 'zero')

    # Example 3: Count the number of price labels for each stock
    price_label_counts = hourly_data.groupby(['symbol', 'price_label']).size().unstack()

    # Example 4: Calculate the volume rank for each stock every hour
    hourly_data['volume_rank'] = hourly_data.groupby(['symbol'])['volume'].rank(ascending=False)

    # Example 5: Calculate the average price curve for each stock per hour
    hourly_price_data = hourly_data['price'].unstack(level='symbol')

    # Example 6: Calculate the total volume for each stock per day
    daily_volume_sum = stock_df.groupby([pd.Grouper(freq='D'), 'symbol'])['volume'].sum()

    # Example 7: Calculate the weighted average price based on price and volume for each stock
    hourly_data['weighted_price'] = hourly_data['price'] * hourly_data['volume']
    weighted_avg_price = hourly_data.groupby('symbol')['weighted_price'].sum() / hourly_data.groupby('symbol')[
        'volume'].sum()

    # Example 8: Calculate the price volatility (standard deviation) for each stock per hour
    hourly_data['price_std'] = stock_df.groupby(['symbol', pd.Grouper(freq='H')])['price'].std()

    # Example 9: Calculate the price change statistics for each stock per hour
    price_change_stats = stock_df.groupby(['symbol', pd.Grouper(freq='H')])['price'].apply(
        lambda x: (x.max() - x.min()) / x.min())

    # Example 10: Calculate the volume-weighted average price per hour for each stock
    hourly_data['volume_weighted_price'] = stock_df.groupby(['symbol', pd.Grouper(freq='H')]).apply(
        lambda x: (x['price'] * x['volume']).sum() / x['volume'].sum())

    # Example 11: Calculate the price change based on the volume-weighted average price for each stock
    hourly_data['volume_weighted_price_change'] = hourly_data.groupby('symbol')['volume_weighted_price'].pct_change()

    # Example 12: Calculate the cumulative price change for each stock per hour
    hourly_data['price_cumulative_change'] = hourly_data.groupby('symbol')['price_change'].cumsum()

    # Example 13: Count the number of trades for each stock based on the first trade time per day
    daily_trade_count = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['volume'].count()

    # Example 14: Calculate the number of price limit-up and limit-down occurrences per day for each stock
    stock_df['price_limit_up'] = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['price'].diff() > 0.1
    stock_df['price_limit_down'] = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['price'].diff() < -0.1
    price_limit_counts = stock_df.groupby(['symbol', pd.Grouper(freq='D')])[
        ['price_limit_up', 'price_limit_down']].sum()
    stock_df.drop('price_limit_up', axis=1, inplace=True)
    stock_df.drop('price_limit_down', axis=1, inplace=True)

    # Example 15: Calculate the peak trading volume per day for each stock
    daily_volume_peak = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['volume'].max()

    # Example 16: Calculate the volatility based on the percentage change in daily trading volume for each stock
    stock_df['volume_pct'] = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['volume'].pct_change()
    daily_volatility = stock_df.groupby(['symbol', pd.Grouper(freq='D')])['volume_pct'].std()
    stock_df.drop('volume_pct', axis=1, inplace=True)

    # GroupbyRolling Example 1: Calculate the rolling mean of price for each stock every hour
    price_mean_rolling = stock_df.groupby(['symbol'])['price'].rolling(window=5).mean().reset_index(level=0, drop=True)

    # GroupbyRolling Example 2: Calculate the rolling sum of volume for each stock every hour
    volume_sum_rolling = stock_df.groupby(['symbol'])['volume'].rolling(window=10).sum().reset_index(level=0, drop=True)

    # GroupbyRolling Example 3: Calculate the difference between the maximum and minimum price within each symbol and time window
    stock_df['price_range'] = stock_df.groupby('symbol')['price'].rolling(window=5).apply(
        lambda x: x.max() - x.min()).reset_index(level=0, drop=True)

    # GroupbyRolling Example 4: Calculate the weighted average of price and volume for each symbol within each time window
    stock_df['weighted_average'] = stock_df.groupby('symbol').apply(
        lambda x: np.average(x['price'], weights=x['volume'])).reset_index(level=0, drop=True)


if __name__ == '__main__':
    stock_benchmark()
    # stock_minute()
