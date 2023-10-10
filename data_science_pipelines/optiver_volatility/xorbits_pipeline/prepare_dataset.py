import time
import argparse
import traceback
from pathlib import Path

from typing import Dict, List, Optional
import pickle

import numpy as np
import xorbits
import xorbits.pandas as pd
import scipy

from xorbits.sklearn.neighbors import NearestNeighbors
from xorbits.sklearn.preprocessing import minmax_scale

N_NEIGHBORS_MAX = 80


class Neighbors:
    def __init__(
        self,
        name: str,
        pivot: pd.DataFrame,
        p: float,
        metric: str = "minkowski",
        metric_params: Optional[Dict] = None,
        exclude_self: bool = False,
    ):
        self.name = name
        self.exclude_self = exclude_self
        self.p = p
        self.metric = metric

        nn = NearestNeighbors(
            n_neighbors=N_NEIGHBORS_MAX, p=p, metric=metric, metric_params=metric_params
        )
        nn.fit(pivot)

        _, self.neighbors = nn.kneighbors(pivot, return_distance=True)

        self.columns = self.index = self.feature_values = self.feature_col = None

    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        raise NotImplementedError()

    def make_nn_feature(self, n=5, agg=np.mean) -> pd.DataFrame:
        assert self.feature_values is not None, "should call rearrange_feature_values beforehand"

        start = 1 if self.exclude_self else 0

        pivot_aggs = pd.DataFrame(
            agg(self.feature_values[start:n, :, :], axis=0), columns=self.columns, index=self.index
        )

        dst = pivot_aggs.unstack().reset_index()
        dst.columns = [
            "stock_id",
            "time_id",
            f"{self.feature_col}_nn{n}_{self.name}_{agg.__name__}",
        ]
        return dst


class TimeIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        feature_pivot = df.pivot(index="time_id", columns="stock_id", values=feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        feature_values = np.zeros((N_NEIGHBORS_MAX, *feature_pivot.shape))

        for i in range(N_NEIGHBORS_MAX):
            feature_values[i, :, :] += feature_pivot.values[self.neighbors[:, i], :]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"time-id_nn_(name={self.name}, metric={self.metric}, p={self.p})"


class StockIdNeighbors(Neighbors):
    def rearrange_feature_values(self, df: pd.DataFrame, feature_col: str) -> None:
        """stock-id based nearest neighbor features"""
        feature_pivot = df.pivot(index="time_id", columns="stock_id", values=feature_col)
        feature_pivot = feature_pivot.fillna(feature_pivot.mean())

        feature_values = np.zeros((N_NEIGHBORS_MAX, *feature_pivot.shape))

        for i in range(N_NEIGHBORS_MAX):
            feature_values[i, :, :] += feature_pivot.values[:, self.neighbors[:, i]]

        self.columns = list(feature_pivot.columns)
        self.index = list(feature_pivot.index)
        self.feature_values = feature_values
        self.feature_col = feature_col

    def __repr__(self) -> str:
        return f"stock-id NN (name={self.name}, metric={self.metric}, p={self.p})"

def train_nearest_neighbors(df):
    time_id_neighbors: List[Neighbors] = []
    stock_id_neighbors: List[Neighbors] = []

    df_pv = df[["stock_id", "time_id"]].copy()
    df_pv["price"] = 0.01 / df["tick_size"]
    df_pv["vol"] = df["book.log_return1.realized_volatility"]
    df_pv["trade.tau"] = df["trade.tau"]
    df_pv["trade.size.sum"] = df["book.total_volume.sum"]

    # Price features
    pivot = df_pv.pivot(index="time_id", columns="stock_id", values="price")
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))

    time_id_neighbors.append(
        TimeIdNeighbors("time_price_c", pivot, p=2, metric="canberra", exclude_self=True)
    )

    vi = scipy.linalg.inv(np.cov(pivot.values))

    time_id_neighbors.append(
        TimeIdNeighbors(
            "time_price_m",
            pivot,
            p=2,
            metric="mahalanobis",
            # metric_params={'V':np.cov(pivot.values.T)}
            metric_params={"VI": vi},
        )
    )
    stock_id_neighbors.append(
        StockIdNeighbors(
            "stock_price_l1", minmax_scale(pivot.transpose()), p=1, exclude_self=True
        )
    )

    pivot = df_pv.pivot(index="time_id", columns="stock_id", values="vol")
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))

    time_id_neighbors.append(TimeIdNeighbors("time_vol_l1", pivot, p=1))

    stock_id_neighbors.append(
        StockIdNeighbors(
            "stock_vol_l1", minmax_scale(pivot.transpose()), p=1, exclude_self=True
        )
    )

    # size nn features
    pivot = df_pv.pivot(index="time_id", columns="stock_id", values="trade.size.sum")
    pivot = pivot.fillna(pivot.mean())
    pivot = pd.DataFrame(minmax_scale(pivot))

    vi = scipy.linalg.inv(np.cov(pivot.values))
    time_id_neighbors.append(
        TimeIdNeighbors(
            "time_size_m", pivot, p=2, metric="mahalanobis", metric_params={"VI": vi}
        )
    )
    time_id_neighbors.append(TimeIdNeighbors("time_size_c", pivot, p=2, metric="canberra"))
    return df_pv, pivot, time_id_neighbors, stock_id_neighbors


def normalize_rank(df):
    # features with large changes over time are converted to relative ranks within time-id
    df["trade.order_count.mean"] = df.groupby("time_id")["trade.order_count.mean"].rank()
    df["book.total_volume.sum"] = df.groupby("time_id")["book.total_volume.sum"].rank()
    df["book.total_volume.mean"] = df.groupby("time_id")["book.total_volume.mean"].rank()
    df["book.total_volume.std"] = df.groupby("time_id")["book.total_volume.std"].rank()

    df["trade.tau"] = df.groupby("time_id")["trade.tau"].rank()

    for dt in [150, 300, 450]:
        df[f"book_{dt}.total_volume.sum"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.sum"
        ].rank()
        df[f"book_{dt}.total_volume.mean"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.mean"
        ].rank()
        df[f"book_{dt}.total_volume.std"] = df.groupby("time_id")[
            f"book_{dt}.total_volume.std"
        ].rank()
        df[f"trade_{dt}.order_count.mean"] = df.groupby("time_id")[
            f"trade_{dt}.order_count.mean"
        ].rank()


def make_nearest_neighbor_feature(
    df: pd.DataFrame, stock_id_neighbors, time_id_neighbors
) -> pd.DataFrame:
    df2 = df.copy()

    feature_cols_stock = {
        "book.log_return1.realized_volatility": [np.mean, np.min, np.max, np.std],
        "trade.seconds_in_bucket.count": [np.mean],
        "trade.tau": [np.mean],
        "trade_150.tau": [np.mean],
        "book.tau": [np.mean],
        "trade.size.sum": [np.mean],
        "book.seconds_in_bucket.count": [np.mean],
    }

    feature_cols = {
        "book.log_return1.realized_volatility": [np.mean, np.min, np.max, np.std],
        "real_price": [np.max, np.mean, np.min],
        "trade.seconds_in_bucket.count": [np.mean],
        "trade.tau": [np.mean],
        "trade.size.sum": [np.mean],
        "book.seconds_in_bucket.count": [np.mean],
        "trade_150.tau_nn20_stock_vol_l1_mean": [np.mean],
        "trade.size.sum_nn20_stock_vol_l1_mean": [np.mean],
    }

    time_id_neigbor_sizes = [3, 5, 10, 20, 40]
    time_id_neigbor_sizes_vol = [2, 3, 5, 10, 20, 40]
    stock_id_neighbor_sizes = [10, 20, 40]

    ndf: Optional[pd.DataFrame] = None

    def _add_ndf(ndf: Optional[pd.DataFrame], dst: pd.DataFrame) -> pd.DataFrame:
        if ndf is None:
            return dst
        else:
            ndf[dst.columns[-1]] = dst[dst.columns[-1]].astype(np.float32)
            return ndf

    # neighbor stock_id
    for feature_col in feature_cols_stock.keys():
        if feature_col not in df2.columns:
            continue

        if not stock_id_neighbors:
            continue

        for nn in stock_id_neighbors:
            nn.rearrange_feature_values(df2, feature_col)

        for agg in feature_cols_stock[feature_col]:
            for n in stock_id_neighbor_sizes:
                for nn in stock_id_neighbors:
                    dst = nn.make_nn_feature(n, agg)
                    ndf = _add_ndf(ndf, dst)

    if ndf is not None:
        df2 = pd.merge(df2, ndf, on=["time_id", "stock_id"], how="left")
    ndf = None

    # neighbor time_id
    for feature_col in feature_cols.keys():
        if feature_col not in df2.columns:
            continue

        for nn in time_id_neighbors:
            nn.rearrange_feature_values(df2, feature_col)

        if "volatility" in feature_col:
            time_id_ns = time_id_neigbor_sizes_vol
        else:
            time_id_ns = time_id_neigbor_sizes

        for agg in feature_cols[feature_col]:
            for n in time_id_ns:
                for nn in time_id_neighbors:
                    dst = nn.make_nn_feature(n, agg)
                    ndf = _add_ndf(ndf, dst)

    if ndf is not None:
        df2 = pd.merge(df2, ndf, on=["time_id", "stock_id"], how="left")

    # features further derived from nearest neighbor features
    for sz in time_id_neigbor_sizes:
        denominator = f"real_price_nn{sz}_time_price_c"

        df2[f"real_price_rankmin_{sz}"] = df2["real_price"] / df2[f"{denominator}_amin"]
        df2[f"real_price_rankmax_{sz}"] = df2["real_price"] / df2[f"{denominator}_amax"]
        df2[f"real_price_rankmean_{sz}"] = df2["real_price"] / df2[f"{denominator}_mean"]

    for sz in time_id_neigbor_sizes_vol:
        denominator = f"book.log_return1.realized_volatility_nn{sz}_time_price_c"

        df2[f"vol_rankmin_{sz}"] = (
            df2["book.log_return1.realized_volatility"] / df2[f"{denominator}_amin"]
        )
        df2[f"vol_rankmax_{sz}"] = (
            df2["book.log_return1.realized_volatility"] / df2[f"{denominator}_amax"]
        )

    price_cols = [c for c in df2.columns if "real_price" in c and "rank" not in c]
    for c in price_cols:
        del df2[c]

    for sz in time_id_neigbor_sizes_vol:
        tgt = f"book.log_return1.realized_volatility_nn{sz}_time_price_m_mean"
        df2[f"{tgt}_rank"] = df2.groupby("time_id")[tgt].rank()
    return df2


def skew_correction(df2):
    """Skew correction for NN"""
    cols_to_log = [
        "trade.size.sum",
        "trade_150.size.sum",
        "trade_300.size.sum",
        "trade_450.size.sum",
        "volume_imbalance",
    ]
    for c in df2.columns:
        for check in cols_to_log:
            try:
                if check in c:
                    df2[c] = np.log(df2[c] + 1)
                    break
            except Exception:
                print(traceback.format_exc())


def rolling_average(df2):
    """Rolling average of RV for similar trading volume"""
    try:
        df2.sort_values(by=["stock_id", "book.total_volume.sum"], inplace=True)
        df2.reset_index(drop=True, inplace=True)

        roll_target = "book.log_return1.realized_volatility"

        for window_size in [3, 10]:
            df2[f"realized_volatility_roll{window_size}_by_book.total_volume.mean"] = (
                df2.groupby("stock_id")[roll_target]
                .rolling(window_size, center=True, min_periods=1)
                .mean()
                .reset_index()
                .sort_values(by=["level_1"])[roll_target]
                .values
            )
    except Exception:
        print(traceback.format_exc())


def fe(preprocessed_path):
    print(type(preprocessed_path))
    print(preprocessed_path)
    preprocessed_path = "/fs/fast/u20200002/tmp/workdir/features_v2.f"
    df = pd.read_feather(preprocessed_path)

    # the tau itself is meaningless for GBDT, but useful as input to aggregate in Nearest Neighbor features
    # df["trade.tau"] = np.sqrt(1 / df["trade.seconds_in_bucket.count"])
    # df["trade_150.tau"] = np.sqrt(1 / df["trade_150.seconds_in_bucket.count"])
    # df["book.tau"] = np.sqrt(1 / df["book.seconds_in_bucket.count"])

    # use apply
    print(df.head(10))
    df["trade.tau"] = df["trade.seconds_in_bucket.count"].apply(lambda x: np.sqrt(1 / x))
    df["trade_150.tau"] = df["trade_150.seconds_in_bucket.count"].apply(lambda x: np.sqrt(1 / x))
    df["book.tau"] = df["book.seconds_in_bucket.count"].apply(lambda x: np.sqrt(1 / x))
    df["real_price"] = 0.01 / df["tick_size"]

    df_pv, pivot, time_id_neighbors, stock_id_neighbors = train_nearest_neighbors(df)

    normalize_rank(df)

    df2 = make_nearest_neighbor_feature(df, stock_id_neighbors, time_id_neighbors)

    skew_correction(df2)
    rolling_average(df2)

    return df2


def prepare_dataset(paths):
    df2 = fe(paths["preprocessed"])
    xorbits.run(df2)
    

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
    posix_paths = {}
    for key, value in paths.items():
        posix_paths[key] = value.as_posix()
    xorbits.init()

    prepare_dataset(paths=posix_paths)
    xorbits.shutdown()

if __name__ == "__main__":
    main()