import os

SCALE_FACTOR = os.environ.get("SCALE_FACTOR", "1")

CWD = os.path.dirname(os.path.realpath(__file__))
DATASET_BASE_DIR = os.path.join(CWD, f"SF_{SCALE_FACTOR}")
ANSWERS_BASE_DIR = os.path.join(CWD, "tpch-dbgen/answers")

TIMINGS_FILE = os.path.join(CWD, "time.csv")
DEFAULT_PLOTS_DIR = os.path.join(CWD, "plots")

WRITE_PLOT = bool(os.environ.get("WRITE_PLOT", False))


def append_row(solution: str, q: str, secs: float, version: str, success=True):
    with open(TIMINGS_FILE, "a") as f:
        if f.tell() == 0:
            f.write("solution,version,query_no,duration[s],success\n")
        f.write(f"{solution},{version},{q},{secs},{success}\n")
        print(f"{solution},{version},{q},{secs},{success}")
