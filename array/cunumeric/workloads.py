import argparse
import traceback

import cunumeric as np
from legate.timing import time

from common_utils import average_without_extremes, log_time_fn


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
}


def run_workloads(workloads, sizes, runs):
    # this is the version of cunumeric rather than numpy
    version = np.__version__
    for workload in workloads:
        func = workload_to_runner[workload]["func"]
        sizes_dict = workload_to_runner[workload]["sizes"]
        for key, size in sizes_dict.items():
            if key in sizes:
                times = []
                try:
                    for _ in range(runs):
                        start = time("s")
                        result = func(size)
                        end = time("s")
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
                log_time_fn("cunumeric", workload=workload, version=version, size=size, duration=duration, success=success)


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
    parser.add_argument(
        "--runs",
        type=int,
        required=False,
        default=7,
        help="Number of runs.",
    )
    args = parser.parse_args()

    run_workloads(args.workloads, args.size, args.runs)


if __name__ == "__main__":
    main()
