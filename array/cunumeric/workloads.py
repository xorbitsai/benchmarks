import argparse

import cunumeric as np
from legate.timing import time


def matmul(size):
    A = np.random.random((size, size))
    B = np.random.random((size, size))

    C = np.linalg.matmul(A, B)

    return C


def fft(size):
    a = np.random.random(size=(size, size))
    s = np.fft.fft(a=a)

    return s


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
    "fft": {
        "func": fft,
        "sizes": {
            "s": 1_000,
            "m": 10_000,
            "l": 20_000,
            "xl": 40_000,
        },
    },
}


def run_workloads(workloads, sizes):
    for workload in workloads:
        func = workload_to_runner[workload]["func"]
        sizes_dict = workload_to_runner[workload]["sizes"]
        for key, size in sizes_dict.items():
            if key in sizes:
                start = time("s")
                func(size)
                end = time("s")
                duration = end - start
                print(f"{workload},{size},{duration}")


def main():
    parser = argparse.ArgumentParser(description="Legate cunumeric Array benchmark")
    parser.add_argument(
        "--workloads",
        type=str,
        nargs="+",
        required=False,
        default=["matmul", "fft"],
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

    run_workloads(args.workloads, args.size)


if __name__ == "__main__":
    main()
