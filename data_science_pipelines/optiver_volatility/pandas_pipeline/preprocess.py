import time
import argparse
import traceback
from pathlib import Path

import numpy as np
import pandas as pd


def calc_wap1(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]) / (df["bid_size1"] + df["ask_size1"])
    return wap


def calc_wap2(df: pd.DataFrame) -> pd.Series:
    wap = (df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]) / (df["bid_size2"] + df["ask_size2"])
    return wap


def realized_volatility(series):
    return np.sqrt(np.sum(series**2))


def log_return(series: np.ndarray):
    return np.log(series).diff()


def log_return_df2(series: np.ndarray):
    return np.log(series).diff(2)


def flatten_name(prefix, src_names):
    ret = []
    for c in src_names:
        if c[0] in ["time_id", "stock_id"]:
            ret.append(c[0])
        else:
            ret.append(".".join([prefix] + list(c)))
    return ret


def make_book_feature(book_path):
    gb_cols = ["stock_id", "time_id"]
    book = pd.read_parquet(book_path)
    book["wap1"] = calc_wap1(book)
    book["wap2"] = calc_wap2(book)

    book["log_return1"] = book.groupby(gb_cols, group_keys=False)["wap1"].apply(log_return)
    book["log_return2"] = book.groupby(gb_cols, group_keys=False)["wap2"].apply(log_return)
    book["log_return_ask1"] = book.groupby(gb_cols, group_keys=False)["ask_price1"].apply(
        log_return
    )
    book["log_return_ask2"] = book.groupby(gb_cols, group_keys=False)["ask_price2"].apply(
        log_return
    )
    book["log_return_bid1"] = book.groupby(gb_cols, group_keys=False)["bid_price1"].apply(
        log_return
    )
    book["log_return_bid2"] = book.groupby(gb_cols, group_keys=False)["bid_price2"].apply(
        log_return
    )

    book["wap_balance"] = abs(book["wap1"] - book["wap2"])
    book["price_spread"] = (book["ask_price1"] - book["bid_price1"]) / (
        (book["ask_price1"] + book["bid_price1"]) / 2
    )
    book["bid_spread"] = book["bid_price1"] - book["bid_price2"]
    book["ask_spread"] = book["ask_price1"] - book["ask_price2"]
    book["total_volume"] = (book["ask_size1"] + book["ask_size2"]) + (
        book["bid_size1"] + book["bid_size2"]
    )
    book["volume_imbalance"] = abs(
        (book["ask_size1"] + book["ask_size2"]) - (book["bid_size1"] + book["bid_size2"])
    )

    features = {
        "seconds_in_bucket": ["count"],
        "wap1": [np.sum, np.mean, np.std],
        "wap2": [np.sum, np.mean, np.std],
        "log_return1": [np.sum, realized_volatility, np.mean, np.std],
        "log_return2": [np.sum, realized_volatility, np.mean, np.std],
        "log_return_ask1": [np.sum, realized_volatility, np.mean, np.std],
        "log_return_ask2": [np.sum, realized_volatility, np.mean, np.std],
        "log_return_bid1": [np.sum, realized_volatility, np.mean, np.std],
        "log_return_bid2": [np.sum, realized_volatility, np.mean, np.std],
        "wap_balance": [np.sum, np.mean, np.std],
        "price_spread": [np.sum, np.mean, np.std],
        "bid_spread": [np.sum, np.mean, np.std],
        "ask_spread": [np.sum, np.mean, np.std],
        "total_volume": [np.sum, np.mean, np.std],
        "volume_imbalance": [np.sum, np.mean, np.std],
    }

    agg = book.groupby(gb_cols).agg(features).reset_index(drop=False)
    agg.columns = flatten_name("book", agg.columns)

    time = 450
    d = (
        book[book["seconds_in_bucket"] >= time]
        .groupby(gb_cols)
        .agg(features)
        .reset_index(drop=False)
    )
    d.columns = flatten_name(f"book_{time}", d.columns)
    agg = pd.merge(agg, d, on=gb_cols, how="left")

    time = 300
    d = (
        book[book["seconds_in_bucket"] >= time]
        .groupby(gb_cols)
        .agg(features)
        .reset_index(drop=False)
    )
    d.columns = flatten_name(f"book_{time}", d.columns)
    agg = pd.merge(agg, d, on=gb_cols, how="left")

    time = 150
    d = (
        book[book["seconds_in_bucket"] >= time]
        .groupby(gb_cols)
        .agg(features)
        .reset_index(drop=False)
    )
    d.columns = flatten_name(f"book_{time}", d.columns)
    agg = pd.merge(agg, d, on=gb_cols, how="left")
    print(agg)

    # for time in [450, 300, 150]:
    #     d = (
    #         book[book["seconds_in_bucket"] >= time]
    #         .groupby(gb_cols)
    #         .agg(features)
    #         .reset_index(drop=False)
    #     )
    #     d.columns = flatten_name(f"book_{time}", d.columns)
    #     agg = pd.merge(agg, d, on=gb_cols, how="left")
    return agg


def make_trade_feature(trade_path):
    gb_cols = ["stock_id", "time_id"]

    trade = pd.read_parquet(trade_path)

    trade["log_return"] = trade.groupby(gb_cols, group_keys=False)["price"].apply(log_return)

    features = {
        "log_return": [realized_volatility],
        "seconds_in_bucket": ["count"],
        "size": [np.sum],
        "order_count": [np.mean],
    }

    agg = trade.groupby(gb_cols).agg(features).reset_index()
    agg.columns = flatten_name("trade", agg.columns)

    for time in [450, 300, 150]:
        d = (
            trade[trade["seconds_in_bucket"] >= time]
            .groupby(gb_cols)
            .agg(features)
            .reset_index(drop=False)
        )
        d.columns = flatten_name(f"trade_{time}", d.columns)
        agg = pd.merge(agg, d, on=gb_cols, how="left")
    return agg


def make_book_feature_v2(book_path):
    gb_cols = ["stock_id", "time_id"]

    book = pd.read_parquet(book_path)

    prices = book[
        ["stock_id", "time_id", *["bid_price1", "ask_price1", "bid_price2", "ask_price2"]]
    ]

    def find_smallest_spread(df: pd.DataFrame):
        """This looks like we want to find the smallest difference between prices at a given
        time. So it's like the smallest spread."""
        try:
            price_list = np.unique(df.values.flatten())
            price_list.sort()
            return np.diff(price_list).min()
        except Exception:
            error_name = str(df[gb_cols].iloc[0])
            print(f'ERROR RAISED IN {error_name or "anonymous"}')
            print(traceback.format_exc())
            return np.nan

    ticks = prices.groupby(gb_cols).apply(find_smallest_spread)

    if type(ticks) == pd.DataFrame:
        ticks: pd.Series = ticks.squeeze(axis=1)

    ticks.name = "tick_size"
    ticks_df = ticks.reset_index()

    return ticks_df


def preprocess(paths: dict[str, Path]):
    start = time.time()
    train = pd.read_csv(paths["train"])
    book = make_book_feature(paths["book"])
    trade = make_trade_feature(paths["trade"])
    book_v2 = make_book_feature_v2(paths["book"])
    df = pd.merge(train, book, on=["stock_id", "time_id"], how="left")
    df = pd.merge(df, trade, on=["stock_id", "time_id"], how="left")
    df = pd.merge(df, book_v2, on=["stock_id", "time_id"], how="left")
    
    test_df = df.iloc[:170000].copy()
    test_df["time_id"] += 32767
    test_df["row_id"] = ""

    df = pd.concat([df, test_df.drop("row_id", axis=1)]).reset_index(drop=True)
    df.to_feather(paths["preprocessed"])
    print(f"time: {time.time() - start}")

def fe(preprocessed_path):
    df = pd.read_feather(preprocessed_path)

def main():
    parser = argparse.ArgumentParser(description="Optiver Realized Volatility Prediction")
    parser.add_argument(
        "--raw_data_path", type=str, required=True, help="path to the Optiver dataset."
    )
    parser.add_argument(
        "--workdir", type=str, default="./workdir", help="path to the Optiver dataset."
    )
    args = parser.parse_args()

    workdir = Path(args.workdir)
    raw_data = Path(args.raw_data_path)

    paths = dict(
        workdir=workdir,
        raw_data=raw_data,
        train=raw_data / "train.csv",
        book=raw_data / "book_train.parquet",
        trade=raw_data / "trade_train.parquet",
        preprocessed=workdir / "features_v2.f",
        train_dataset=workdir / "train_dataset.f",
        test_dataset=workdir / "test_dataset.f",
        folds=workdir / "folds.pkl",
    )
    workdir.mkdir(exist_ok=True, parents=True)

    preprocess(paths=paths)

if __name__ == "__main__":
    main()