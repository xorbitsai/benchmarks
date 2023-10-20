def print_results(results, framework=None, ignore_fields=[]):
    import json

    TIMINGS_FILE = "time.out"
    results["framework"] = framework

    with open(TIMINGS_FILE, "a") as f:
        json_metric = json.dumps(results)
        f.write(json_metric + "\n")
        print(json_metric)