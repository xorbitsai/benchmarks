import time
import argparse
import traceback

import ray
import modin.pandas as pd

from utils import print_results


def mse(y_test, y_pred):
    return ((y_test - y_pred) ** 2).mean()


def cod(y_test, y_pred):
    y_bar = y_test.mean()
    total = ((y_test - y_bar) ** 2).sum()
    residuals = ((y_test - y_pred) ** 2).sum()
    return 1 - (residuals / total)


def etl(filename, columns_names, columns_types, etl_keys):
    etl_times = {key: 0.0 for key in etl_keys}

    t0 = time.time()
    types = {columns_names[i]: columns_types[i] for i in range(len(columns_names))}
    df = pd.read_csv(
        filename,
        names=columns_names,
        nrows=None,
        header=0,
        dtype=types,
        parse_dates=None,
    )
    
    etl_times["t_readcsv"] = time.time() - t0

    t_etl_start = time.time()

    keep_cols = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "PERNUM",
        "SEX",
        "AGE",
        "INCTOT",
        "EDUC",
        "EDUCD",
        "EDUC_HEAD",
        "EDUC_POP",
        "EDUC_MOM",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
        "INCTOT_HEAD",
        "SEX_HEAD",
    ]
    df = df[keep_cols]

    df = df[df["INCTOT"] != 9999999]
    df = df[df["EDUC"] != -1]
    df = df[df["EDUCD"] != -1]

    df["INCTOT"] = df["INCTOT"] * df["CPI99"]

    df = df.apply(lambda x: x.fillna(-1))

    rows_with_na = df[df.isna().any(axis=1)]
    print(rows_with_na)
    print("-----")


    # for column in keep_cols:
    #     df[column] = df[column].fillna(-1)
    #     df[column] = df[column].astype("float64")

    y = df["EDUC"]
    X = df.drop(columns=["EDUC", "CPI99"])
    print(f"DataFrame shape: {X.shape}, {y.shape}")

    etl_times["t_etl"] = time.time() - t_etl_start

    return df, X, y, etl_times


def main():
    parser = argparse.ArgumentParser(description="census data science pipelines")
    parser.add_argument("--path", type=str, required=True, help="path to the census dataset.")
    args = parser.parse_args()

    ray.init()

    columns_names = [
        "YEAR0",
        "DATANUM",
        "SERIAL",
        "CBSERIAL",
        "HHWT",
        "CPI99",
        "GQ",
        "QGQ",
        "PERNUM",
        "PERWT",
        "SEX",
        "AGE",
        "EDUC",
        "EDUCD",
        "INCTOT",
        "SEX_HEAD",
        "SEX_MOM",
        "SEX_POP",
        "SEX_SP",
        "SEX_MOM2",
        "SEX_POP2",
        "AGE_HEAD",
        "AGE_MOM",
        "AGE_POP",
        "AGE_SP",
        "AGE_MOM2",
        "AGE_POP2",
        "EDUC_HEAD",
        "EDUC_MOM",
        "EDUC_POP",
        "EDUC_SP",
        "EDUC_MOM2",
        "EDUC_POP2",
        "EDUCD_HEAD",
        "EDUCD_MOM",
        "EDUCD_POP",
        "EDUCD_SP",
        "EDUCD_MOM2",
        "EDUCD_POP2",
        "INCTOT_HEAD",
        "INCTOT_MOM",
        "INCTOT_POP",
        "INCTOT_SP",
        "INCTOT_MOM2",
        "INCTOT_POP2",
    ]
    columns_types = [
        "int64",
        "int64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "float64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "int64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
        "float64",
    ]
    etl_keys = ["t_readcsv", "t_etl"]
    ml_keys = ["t_train_test_split", "t_ml", "t_train", "t_inference"]

    ml_score_keys = ["mse_mean", "cod_mean", "mse_dev", "cod_dev"]

    df, X, y, times = etl(
        args.path,
        columns_names=columns_names,
        columns_types=columns_types,
        etl_keys=etl_keys,
    )

    print_results(results=times, framework="modin_ray")
    ray.shutdown()


if __name__ == "__main__":
    main()