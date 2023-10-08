import json
from typing import List

def append_row(
    solution: str, 
    workload: str, 
    version: str,
    size: str,
    duration: float = 0.0,
    success=True
):
    TIMINGS_FILE = f"time-{solution}.csv"
    with open(TIMINGS_FILE, "a") as f:
        metric = {
            "framework": solution,
            "version": version,
            "workload": workload,
            "size": size,
            "duration": duration,
            "is_success": success
        }
        json_metric = json.dumps(metric)
        f.write(json_metric + "\n")
        print(json_metric)

def average_without_extremes(times: List):
    if len(times) > 3:
        times.remove(max(times))
        times.remove(min(times))
    avg_time = sum(times) / len(times)
    return avg_time