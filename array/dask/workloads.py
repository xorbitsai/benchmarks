import argparse
import time

import numpy as np
import dask.array as da
from dask.distributed import Client


def matmul(size, chunk, client):
    if not chunk:
        chunks = "auto"
    else:
        chunks = (int(size / chunk), int(size / chunk))
    A = da.random.random((size, size), chunks=chunks)
    B = da.random.random((size, size), chunks=chunks)

    C = da.matmul(A, B)
    C.compute()

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
    if not chunk:
        chunk = 10
    a = da.random.random(size=(size, size))
    a = a.rechunk(size, int(size / chunk))
    s = da.fft.fft(a=a)

    s = s.compute()
    return s


def lregression(size, chunk, client):
    from dask_ml.linear_model import LinearRegression

    p = int(da.log(size) + 100)
    X = da.random.random((size, p))
    y = da.random.random(size)

    lr = LinearRegression()
    lr.fit(X, y)
    return lr


workload_to_runner = {
    "matmul": {
        "func": matmul,
        "sizes": {
            "s": 10_000,
            "m": 20_000,
            "l": 50_000,
            "xl": 100_000,
        },
    },
    "blacksch": {
        "func": blacksch,
        "sizes": {
            "s": 100_000,
            "m": 1_000_000,
            "l": 100_000_000,
            "xl": 1_000_000_000,
        },
    },
    "qr": {
        "func": qr,
        "sizes": {
            "s": 1_000,
            "m": 5_000,
            "l": 8_000,
            "xl": 10_000,
        },
    },
    "fft": {
        "func": fft,
        "sizes": {
            "s": 1_000,
            "m": 10_000,
            "l": 20_000,
            "xl": 40_000,
        },
    },
    "lregression": {
        "func": lregression,
        "sizes": {
            "s": 100_000,
            "m": 1_000_000,
            "l": 10_000_000,
            "xl": 100_000_000,
        },
    },
}


def run_workloads(workloads, sizes, chunk, client):
    for workload in workloads:
        func = workload_to_runner[workload]["func"]
        sizes_dict = workload_to_runner[workload]["sizes"]
        for key, size in sizes_dict.items():
            if key in sizes:
                start = time.time()
                result = func(size, chunk, client)
                end = time.time()
                duration = end - start
                print(f"{workload},{size},{duration}")


def main():
    parser = argparse.ArgumentParser(description="Dask Array benchmark")
    parser.add_argument("--endpoint", type=str, default=None, help="cluster endpoint")
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        required=False,
        default=["matmul", "blacksch", "qr", "fft", "lregression"],
        help="Comma separated workloads to run.",
    )
    parser.add_argument(
        "--size",
        type=str,
        nargs="+",
        required=False,
        default=["s", "m", "l", "xl"],
        help="Comma separated size to run.",
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
    run_workloads(args.workloads, args.size, args.chunk, client)


if __name__ == "__main__":
    main()
