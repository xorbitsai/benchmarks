import os
import argparse
import traceback
import pandas as pd
from pandas.core.frame import DataFrame as PandasDF

def get_query_answer(query: int, base_dir: str) -> PandasDF:
    answer_df = pd.read_csv(
        os.path.join(base_dir, f"{query}.out"),
        sep=",",
        parse_dates=True,
    )
    return answer_df.rename(columns=lambda x: x.strip())


def test_results(q_num: int, answer_path: str, cur_result_path: str):
    answer = get_query_answer(q_num, answer_path)
    cur_result = get_query_answer(q_num, cur_result_path)

    for column_index in range(len(answer.columns)):
        column_name = answer.columns[column_index]
        column_data_type = answer.iloc[:, column_index].dtype

        s1 = cur_result.iloc[:, column_index]
        s2 = answer.iloc[:, column_index]

        if column_data_type.name == "object":
            s1 = s1.astype("string").apply(lambda x: x.strip())
            s2 = s2.astype("string").apply(lambda x: x.strip())

        pd.testing.assert_series_equal(
            left=s1,
            right=s2,
            check_index=False,
            check_names=False,
            check_exact=False,
            rtol=1e-2,
        )


def main():
    parser = argparse.ArgumentParser(description="test results.")
    parser.add_argument(
        "--answer_path", type=str, required=True, help="path to the ground truth answer."
    )
    parser.add_argument(
        "--cur_result_path", type=str, required=True, help="path to the result."
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="whitespace separated TPC-H queries to run.",
    )
    args = parser.parse_args()
    print(f"current result: {args.cur_result_path}")
    print(f"answer path: {args.answer_path}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries

    for query in queries:
        try:
            test_results(query, args.answer_path, args.cur_result_path)
        except Exception as e:
            print(f"q{query} is wrong or missing!")

if __name__ == "__main__":
    main()