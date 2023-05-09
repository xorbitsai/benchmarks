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
    print("Pandas Calculating the Moving Annualized Volatility of Stock Returns Execution time: ", end_time - start_time)

    start_time = time.time()
    x_Volatility = x_stocks_df[['StockName', 'Ret']].groupby(
        'StockName', group_keys=True
    ).apply(lambda x: x.rolling(window=10).std()) * np.sqrt(252)

    xorbits.run(x_Volatility)
    end_time = time.time()
    print("Xorbits Calculating the Moving Annualized Volatility of Stock Returns Execution time: ", end_time - start_time)

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


if __name__ == '__main__':
    stock_benchmark()