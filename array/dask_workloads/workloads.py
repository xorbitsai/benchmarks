import time
import argparse
import traceback

import numpy as np
import dask
import dask.array as da
from dask.distributed import Client

from common_utils import average_without_extremes, log_time_fn


def matmul(size, chunk, client):
    if not chunk:
        chunks = "auto"
    else:
        chunks = (int(size / chunk), int(size / chunk))
    A = da.random.random((size, size), chunks=chunks)
    B = da.random.random((size, size), chunks=chunks)

    C = da.matmul(A, B)
    C = dask.compute(C)
    return C


def blacksch(size, chunk, client):
    S0L = 10.0
    S0H = 50.0
    XL = 10.0
    XH = 50.0
    TL = 1.0
    TH = 2.0
    RISK_FREE = 0.1
    VOLATILITY = 0.2

    def gen_data():
        if not chunk:
            chunks = "auto"
        else:
            chunks = (size / chunk,)
        return (
            da.random.uniform(S0L, S0H, size, chunks=chunks),
            da.random.uniform(XL, XH, size, chunks=chunks),
            da.random.uniform(TL, TH, size, chunks=chunks),
        )

    def blacksch_np(price, strike, t):
        from scipy.special import erf

        invsqrt = lambda x: 1.0 / np.sqrt(x)
        exp = np.exp
        log = np.log

        call = np.zeros(size, dtype=np.float64)
        put = -np.ones(size, dtype=np.float64)

        rate = RISK_FREE
        vol = VOLATILITY
        mr = -rate
        sig_sig_two = vol * vol * 2

        P = price
        S = strike
        T = t

        a = log(P / S)
        b = T * mr

        z = T * sig_sig_two
        c = 0.25 * z

        y = invsqrt(z)

        w1 = (a - b + c) * y
        w2 = (a - b - c) * y

        d1 = 0.5 + 0.5 * erf(w1)
        d2 = 0.5 + 0.5 * erf(w2)

        Se = exp(b) * S

        call = P * d1 - Se * d2
        put = call - P + Se

        return np.stack((call, put))

    price, strike, t = gen_data()
    da.map_blocks(blacksch_np, price, strike, t, new_axis=0).compute(scheduler=client)


def qr(size, chunk, client):
    if not chunk:
        chunk = 10
    a = da.random.random(size=(size, size), chunks=(int(size / chunk), size))

    Q, R = da.linalg.qr(a=a)

    Q = Q.compute()
    R = R.compute()

    return (Q, R)


def fft(size, chunk, client):
    # if not chunk:
    #     chunk = 10
    # a = da.random.random(size=(size, size))
    a = np.exp(2j * np.pi * np.arange(size) / 8)
    # a = a.rechunk(size, int(size / chunk))
    s = da.fft.fft(a=a)

    s = s.compute()
    return s


def reg(size, chunk, client):
    from dask_glm.datasets import make_regression
    from dask_ml.linear_model import LinearRegression

    X, y = make_regression(n_samples=size, n_features=100, n_informative=5, chunksize=chunk)
    lr = LinearRegression()
    lr.fit(X, y)
    return lr


workload_to_runner = {
    "matmul": {
        "func": matmul,
        "sizes": {
            "1": 5_000,
            "2": 7_071,
            "3": 8_660,
            "4": 10_000,
            "5": 11_180,
            "6": 12_247,
            "7": 13_228,
            "8": 14_142,
        },
    },
    "blacksch": {
        "func": blacksch,
        "sizes": {
            "1": 1_000_000,
            "2": 2_000_000,
            "3": 3_000_000,
            "4": 4_000_000,
            "5": 5_000_000,
            "6": 6_000_000,
            "7": 7_000_000,
            "8": 8_000_000,
        },
    },
    "qr": {
        "func": qr,
        "sizes": {
            "1": 5_000,
            "2": 7_071,
            "3": 8_660,
            "4": 10_000,
            "5": 11_180,
            "6": 12_247,
            "7": 13_228,
            "8": 14_142,
        },
    },
    "fft": {
        "func": fft,
        "sizes": {
            "1": 4_000,
            "2": 8_000,
            "3": 12_000,
            "4": 16_000,
            "5": 20_000,
            "6": 24_000,
            "7": 28_000,
            "8": 32_000,
        },
    },
    "reg": {
        "func": reg,
        "sizes": {
            "1": 1_000_000,
            "2": 2_000_000,
            "3": 3_000_000,
            "4": 4_000_000,
            "5": 5_000_000,
            "6": 6_000_000,
            "7": 7_000_000,
            "8": 8_000_000,
        },
    },
}


def run_workloads(workloads, sizes, runs, chunk, client):
    version = dask.__version__
    for workload in workloads:
        func = workload_to_runner[workload]["func"]
        sizes_dict = workload_to_runner[workload]["sizes"]
        for key, size in sizes_dict.items():
            if key in sizes:
                times = []
                try:
                    for _ in range(runs):
                        start = time.time()
                        result = func(size, chunk, client)
                        end = time.time()
                        t = end - start
                        times.append(t)
                    duration = average_without_extremes(times)
                    success = True
                except Exception as e:
                    print("".join(traceback.TracebackException.from_exception(e).format()))
                    duration = 0.0
                    success = False
                finally:
                    pass
                log_time_fn("dask", workload=workload, version=version, size=size, duration=duration, success=success)


def main():
    parser = argparse.ArgumentParser(description="Array benchmark")
    parser.add_argument("--endpoint", type=str, default=None, help="cluster endpoint")
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        required=False,
        default=["matmul", "blacksch", "qr", "fft", "reg"],
        help="Comma separated workloads to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        nargs="+",
        required=False,
        default=["1"],
        help="Comma separated size to run.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        required=False,
        default=7,
        help="Number of runs.",
    )
    parser.add_argument(
        "--chunk", type=int, required=False, default=None, help="Chunk for Dask"
    )
    args = parser.parse_args()

    if args.endpoint == "local" or args.endpoint is None:
        from dask.distributed import LocalCluster

        client = LocalCluster()
    elif args.endpoint:
        client = Client(args.endpoint)
    run_workloads(args.workloads, args.size, args.runs, args.chunk, client)


if __name__ == "__main__":
    main()
