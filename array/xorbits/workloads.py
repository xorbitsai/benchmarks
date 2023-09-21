import argparse
import time

import xorbits
import xorbits.numpy as np
from xorbits.pandas import set_option

set_option("show_progress", False)


def matmul(size):
    A = np.random.random((size, size))
    B = np.random.random((size, size))

    C = np.matmul(A, B)
    C.execute()

    return C


def blacksch(size):
    erf = np.special.erf
    invsqrt = lambda x: 1.0 / np.sqrt(x)
    exp = np.exp
    log = np.log

    S0L = 10.0
    S0H = 50.0
    XL = 10.0
    XH = 50.0
    TL = 1.0
    TH = 2.0
    RISK_FREE = 0.1
    VOLATILITY = 0.2

    def gen_data(nopt):
        return (
            np.random.uniform(S0L, S0H, nopt),
            np.random.uniform(XL, XH, nopt),
            np.random.uniform(TL, TH, nopt),
        )

    price, strike, t = gen_data(size)
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

    call = call.execute()
    put = put.execute()

    return (call, put)


def qr(size):
    a = np.random.random(size=(size, size))
    Q, R = np.linalg.qr(a=a)

    Q.execute()
    R.execute()

    return (Q, R)


def fft(size):
    a = np.random.random(size=(size, size))
    s = np.fft.fft(a=a)

    s = s.execute()
    return s


def lregression(size):
    from xorbits.sklearn.linear_model import LinearRegression

    p = int(np.log(size) + 100)
    X = np.random.random((size, p))
    y = np.random.random(size)

    lr = LinearRegression()
    lr.fit(X, y)
    return lr


workload_to_runner = {
    "matmul": {
        "func": matmul,
        "sizes": {
            "s": 10_000,
            "m": 20_000,
            "l": 40_000,
            "xl": 80_000,
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
            "m": 2_000,
            "l": 4_000,
            "xl": 8_000,
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


def run_workloads(workloads, sizes):
    for workload in workloads:
        func = workload_to_runner[workload]["func"]
        sizes_dict = workload_to_runner[workload]["sizes"]
        for key, size in sizes_dict.items():
            if key in sizes:
                start = time.time()
                func(size)
                end = time.time()
                duration = end - start
                print(f"{workload},{size},{duration}")


def main():
    parser = argparse.ArgumentParser(description="Xorbits Array benchmark")
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
    args = parser.parse_args()

    if args.endpoint:
        xorbits.init(address=args.endpoint)
    else:
        xorbits.init()
    run_workloads(args.workloads, args.size)
    xorbits.shutdown()


if __name__ == "__main__":
    main()
