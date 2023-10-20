import os
import json

from argparse import ArgumentParser

import pandas as pd

SCALE_FACTOR = os.environ.get("SCALE_FACTOR", "1")

CWD = os.path.dirname(os.path.realpath(__file__))

def log_time_fn(
    solution: str, 
    q: str, 
    version: str,
    without_io_time: float = 0.0,
    with_io_time: float = 0.0,
    success=True
):
    TIMINGS_FILE = f"time-{solution}.csv"
    with open(TIMINGS_FILE, "a") as f:
        metric = {
            "framework": solution,
            "version": version,
            "query": q,
            "without_io_time": without_io_time,
            "with_io_time": with_io_time,
            "is_success": success
        }
        json_metric = json.dumps(metric)
        f.write(json_metric + "\n")
        print(json_metric)


def print_result_fn(solution: str, result: pd.DataFrame, query: str):
    cwd = os.getcwd()
    result_prefix = f"{cwd}/results-{solution}"
    if not os.path.exists(result_prefix):
        os.makedirs(result_prefix)
    result_path = f"{result_prefix}/{query}.out"
    result.to_csv(result_path, index=False)


def parse_common_arguments(parser: ArgumentParser):
    parser.add_argument(
        "--path", type=str, required=True, help="path to the TPC-H dataset."
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="whitespace separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--log_time",
        action="store_true",
        help="log time metrics or not.",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="print result.",
    )
    parser.add_argument(
        "--include_io",
        action="store_true",
        help="include io or not.",
    )

    return parser