import os
import sys
import argparse
import json
import time
import traceback
import datetime
from typing import Dict

import pandas as pd
from pandas.core.frame import DataFrame as PandasDF

import ray
import daft
from daft import DataFrame, col

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common_utils import append_row, ANSWERS_BASE_DIR


dataset_dict = {}


def load_lineitem(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "lineitem" not in dataset_dict or include_io:
        data_path = root + "/lineitem"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["lineitem"] = result
    else:
        result = dataset_dict["lineitem"]
    return result


def load_part(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "part" not in dataset_dict or include_io:
        data_path = root + "/part"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["part"] = result
    else:
        result = dataset_dict["part"]
    return result


def load_orders(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "orders" not in dataset_dict or include_io:
        data_path = root + "/orders"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["orders"] = result
    else:
        result = dataset_dict["orders"]
    return result


def load_customer(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "customer" not in dataset_dict or include_io:
        data_path = root + "/customer"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["customer"] = result
    else:
        result = dataset_dict["customer"]
    return result


def load_nation(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "nation" not in dataset_dict or include_io:
        data_path = root + "/nation"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["nation"] = result
    else:
        result = dataset_dict["nation"]
    return result


def load_region(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "region" not in dataset_dict or include_io:
        data_path = root + "/region"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["region"] = result
    else:
        result = dataset_dict["region"]
    return result


def load_supplier(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "supplier" not in dataset_dict or include_io:
        data_path = root + "/supplier"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["supplier"] = result
    else:
        result = dataset_dict["supplier"]
    return result


def load_partsupp(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    if "partsupp" not in dataset_dict or include_io:
        data_path = root + "/partsupp"
        df = daft.read_parquet(
            data_path,
        )
        result = df
        dataset_dict["partsupp"] = result
    else:
        result = dataset_dict["partsupp"]
    return result


def q01(
    root: str,
    storage_options: Dict,
    include_io: bool = False,
):
    lineitem = load_lineitem(root, storage_options, include_io)

    discounted_price = col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))
    taxed_discounted_price = discounted_price * (1 + col("L_TAX"))
    daft_df = (
        lineitem.where(col("L_SHIPDATE") <= datetime.date(1998, 9, 2))
        .groupby(col("L_RETURNFLAG"), col("L_LINESTATUS"))
        .agg(
            [
                (col("L_QUANTITY").alias("sum_qty"), "sum"),
                (col("L_EXTENDEDPRICE").alias("sum_base_price"), "sum"),
                (discounted_price.alias("sum_disc_price"), "sum"),
                (taxed_discounted_price.alias("sum_charge"), "sum"),
                (col("L_QUANTITY").alias("avg_qty"), "mean"),
                (col("L_EXTENDEDPRICE").alias("avg_price"), "mean"),
                (col("L_DISCOUNT").alias("avg_disc"), "mean"),
                (col("L_QUANTITY").alias("count_order"), "count"),
            ]
        )
        .sort(["L_RETURNFLAG", "L_LINESTATUS"])
    )
    return daft_df


def q02(root: str,
    storage_options: Dict,
    include_io: bool = False,
) -> DataFrame:
    part = load_part(root, storage_options, include_io)
    partsupp = load_partsupp(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    region = load_region(root, storage_options, include_io)

    europe = (
        region.where(col("R_NAME") == "EUROPE")
        .join(nation, left_on=col("R_REGIONKEY"), right_on=col("N_REGIONKEY"))
        .join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(partsupp, left_on=col("S_SUPPKEY"), right_on=col("PS_SUPPKEY"))
    )

    brass = part.where((col("P_SIZE") == 15) & col("P_TYPE").str.endswith("BRASS")).join(
        europe,
        left_on=col("P_PARTKEY"),
        right_on=col("PS_PARTKEY"),
    )
    min_cost = brass.groupby(col("P_PARTKEY")).agg(
        [
            (col("PS_SUPPLYCOST").alias("min"), "min"),
        ]
    )

    daft_df = (
        brass.join(min_cost, on=col("P_PARTKEY"))
        .where(col("PS_SUPPLYCOST") == col("min"))
        .select(
            col("S_ACCTBAL"),
            col("S_NAME"),
            col("N_NAME"),
            col("P_PARTKEY"),
            col("P_MFGR"),
            col("S_ADDRESS"),
            col("S_PHONE"),
            col("S_COMMENT"),
        )
        .sort(by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"], desc=[True, False, False, False])
        .limit(100)
    )
    return daft_df


def q03(root: str,
    storage_options: Dict,
    include_io: bool = False,
) -> DataFrame:
    def decrease(x, y):
        return x * (1 - y)

    customer = load_customer(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    lineitem = load_lineitem(root, storage_options, include_io)

    customer = customer.where(col("C_MKTSEGMENT") == "BUILDING")
    orders = orders.where(col("O_ORDERDATE") < datetime.date(1995, 3, 15))
    lineitem = lineitem.where(col("L_SHIPDATE") > datetime.date(1995, 3, 15))

    daft_df = (
        customer.join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("O_ORDERKEY"), col("O_ORDERDATE"), col("O_SHIPPRIORITY"))
        .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_ORDERKEY"),
            decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
            col("O_ORDERDATE"),
            col("O_SHIPPRIORITY"),
        )
        .groupby(col("O_ORDERKEY"), col("O_ORDERDATE"), col("O_SHIPPRIORITY"))
        .agg([(col("volume").alias("revenue"), "sum")])
        .sort(by=["revenue", "O_ORDERDATE"], desc=[True, False])
        .limit(10)
        .select("O_ORDERKEY", "revenue", "O_ORDERDATE", "O_SHIPPRIORITY")
    )
    return daft_df


def q04(root: str,
    storage_options: Dict,
    include_io: bool = False,
) -> DataFrame:
    orders = load_orders(root, storage_options, include_io)
    lineitem = load_lineitem(root, storage_options, include_io)

    orders = orders.where(
        (col("O_ORDERDATE") >= datetime.date(1993, 8, 1)) & (col("O_ORDERDATE") < datetime.date(1993, 11, 1))
    )

    lineitems = lineitem.where(col("L_COMMITDATE") < col("L_RECEIPTDATE")).select(col("L_ORDERKEY")).distinct()

    daft_df = (
        lineitems.join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .groupby(col("O_ORDERPRIORITY"))
        .agg([(col("L_ORDERKEY").alias("order_count"), "count")])
        .sort(col("O_ORDERPRIORITY"))
    )
    return daft_df


def q05(root: str,
    storage_options: Dict,
    include_io: bool = False,
) -> DataFrame:
    orders = load_orders(root, storage_options, include_io)
    region = load_region(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    lineitem = load_lineitem(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)

    orders = orders.where(
        (col("O_ORDERDATE") >= datetime.date(1994, 1, 1)) & (col("O_ORDERDATE") < datetime.date(1995, 1, 1))
    )
    region = region.where(col("R_NAME") == "ASIA")
    daft_df = (
        region.join(nation, left_on=col("R_REGIONKEY"), right_on=col("N_REGIONKEY"))
        .join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(lineitem, left_on=col("S_SUPPKEY"), right_on=col("L_SUPPKEY"))
        .select(col("N_NAME"), col("L_EXTENDEDPRICE"), col("L_DISCOUNT"), col("L_ORDERKEY"), col("N_NATIONKEY"))
        .join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .join(customer, left_on=[col("O_CUSTKEY"), col("N_NATIONKEY")], right_on=[col("C_CUSTKEY"), col("C_NATIONKEY")])
        .select(col("N_NAME"), (col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))).alias("value"))
        .groupby(col("N_NAME"))
        .agg([(col("value").alias("revenue"), "sum")])
        .sort(col("revenue"), desc=True)
    )
    return daft_df


# def q06(root: str,
#     storage_options: Dict,
#     include_io: bool = False,
# ) -> DataFrame:
#     lineitem = get_df("lineitem")
#     daft_df = lineitem.where(
#         (col("L_SHIPDATE") >= datetime.date(1994, 1, 1))
#         & (col("L_SHIPDATE") < datetime.date(1995, 1, 1))
#         & (col("L_DISCOUNT") >= 0.05)
#         & (col("L_DISCOUNT") <= 0.07)
#         & (col("L_QUANTITY") < 24)
#     ).sum(col("L_EXTENDEDPRICE") * col("L_DISCOUNT"))
#     return daft_df


# def q7(root: str,
#     storage_options: Dict,
#     include_io: bool = False,
# ) -> DataFrame:
#     def decrease(x, y):
#         return x * (1 - y)

#     lineitem = get_df("lineitem").where(
#         (col("L_SHIPDATE") >= datetime.date(1995, 1, 1)) & (col("L_SHIPDATE") <= datetime.date(1996, 12, 31))
#     )
#     nation = get_df("nation").where((col("N_NAME") == "FRANCE") | (col("N_NAME") == "GERMANY"))
#     supplier = get_df("supplier")
#     customer = get_df("customer")
#     orders = get_df("orders")

#     supNation = (
#         nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
#         .join(lineitem, left_on=col("S_SUPPKEY"), right_on=col("L_SUPPKEY"))
#         .select(
#             col("N_NAME").alias("supp_nation"),
#             col("L_ORDERKEY"),
#             col("L_EXTENDEDPRICE"),
#             col("L_DISCOUNT"),
#             col("L_SHIPDATE"),
#         )
#     )

#     daft_df = (
#         nation.join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
#         .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
#         .select(col("N_NAME").alias("cust_nation"), col("O_ORDERKEY"))
#         .join(supNation, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
#         .where(
#             ((col("supp_nation") == "FRANCE") & (col("cust_nation") == "GERMANY"))
#             | ((col("supp_nation") == "GERMANY") & (col("cust_nation") == "FRANCE"))
#         )
#         .select(
#             col("supp_nation"),
#             col("cust_nation"),
#             col("L_SHIPDATE").dt.year().alias("l_year"),
#             decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
#         )
#         .groupby(col("supp_nation"), col("cust_nation"), col("l_year"))
#         .agg([(col("volume").alias("revenue"), "sum")])
#         .sort(by=["supp_nation", "cust_nation", "l_year"])
#     )
#     return daft_df


# def q8(root: str,
#     storage_options: Dict,
#     include_io: bool = False,
# ) -> DataFrame:
#     def decrease(x, y):
#         return x * (1 - y)

#     region = get_df("region").where(col("R_NAME") == "AMERICA")
#     orders = get_df("orders").where(
#         (col("O_ORDERDATE") <= datetime.date(1996, 12, 31)) & (col("O_ORDERDATE") >= datetime.date(1995, 1, 1))
#     )
#     part = get_df("part").where(col("P_TYPE") == "ECONOMY ANODIZED STEEL")
#     nation = get_df("nation")
#     supplier = get_df("supplier")
#     lineitem = get_df("lineitem")
#     customer = get_df("customer")

#     nat = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))

#     line = (
#         lineitem.select(
#             col("L_PARTKEY"),
#             col("L_SUPPKEY"),
#             col("L_ORDERKEY"),
#             decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
#         )
#         .join(part, left_on=col("L_PARTKEY"), right_on=col("P_PARTKEY"))
#         .join(nat, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
#     )

#     daft_df = (
#         nation.join(region, left_on=col("N_REGIONKEY"), right_on=col("R_REGIONKEY"))
#         .select(col("N_NATIONKEY"))
#         .join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
#         .select(col("C_CUSTKEY"))
#         .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
#         .select(col("O_ORDERKEY"), col("O_ORDERDATE"))
#         .join(line, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
#         .select(
#             col("O_ORDERDATE").dt.year().alias("o_year"),
#             col("volume"),
#             (col("N_NAME") == "BRAZIL").if_else(col("volume"), 0.0).alias("case_volume"),
#         )
#         .groupby(col("o_year"))
#         .agg([(col("case_volume").alias("case_volume_sum"), "sum"), (col("volume").alias("volume_sum"), "sum")])
#         .select(col("o_year"), col("case_volume_sum") / col("volume_sum"))
#         .sort(col("o_year"))
#     )

#     return daft_df


# def q9(root: str,
#     storage_options: Dict,
#     include_io: bool = False,
# ) -> DataFrame:
#     lineitem = get_df("lineitem")
#     part = get_df("part")
#     nation = get_df("nation")
#     supplier = get_df("supplier")
#     partsupp = get_df("partsupp")
#     orders = get_df("orders")

#     def expr(x, y, v, w):
#         return x * (1 - y) - (v * w)

#     linepart = part.where(col("P_NAME").str.contains("green")).join(
#         lineitem, left_on=col("P_PARTKEY"), right_on=col("L_PARTKEY")
#     )
#     natsup = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))

#     daft_df = (
#         linepart.join(natsup, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
#         .join(partsupp, left_on=[col("L_SUPPKEY"), col("P_PARTKEY")], right_on=[col("PS_SUPPKEY"), col("PS_PARTKEY")])
#         .join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
#         .select(
#             col("N_NAME"),
#             col("O_ORDERDATE").dt.year().alias("o_year"),
#             expr(col("L_EXTENDEDPRICE"), col("L_DISCOUNT"), col("PS_SUPPLYCOST"), col("L_QUANTITY")).alias("amount"),
#         )
#         .groupby(col("N_NAME"), col("o_year"))
#         .agg([(col("amount"), "sum")])
#         .sort(by=["N_NAME", "o_year"], desc=[False, True])
#     )

#     return daft_df


# def q10(root: str,
#     storage_options: Dict,
#     include_io: bool = False,
# ) -> DataFrame:
#     def decrease(x, y):
#         return x * (1 - y)

#     lineitem = get_df("lineitem").where(col("L_RETURNFLAG") == "R")
#     orders = get_df("orders")
#     nation = get_df("nation")
#     customer = get_df("customer")

#     daft_df = (
#         orders.where(
#             (col("O_ORDERDATE") < datetime.date(1994, 1, 1)) & (col("O_ORDERDATE") >= datetime.date(1993, 10, 1))
#         )
#         .join(customer, left_on=col("O_CUSTKEY"), right_on=col("C_CUSTKEY"))
#         .join(nation, left_on=col("C_NATIONKEY"), right_on=col("N_NATIONKEY"))
#         .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
#         .select(
#             col("O_CUSTKEY"),
#             col("C_NAME"),
#             decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
#             col("C_ACCTBAL"),
#             col("N_NAME"),
#             col("C_ADDRESS"),
#             col("C_PHONE"),
#             col("C_COMMENT"),
#         )
#         .groupby(
#             col("O_CUSTKEY"),
#             col("C_NAME"),
#             col("C_ACCTBAL"),
#             col("C_PHONE"),
#             col("N_NAME"),
#             col("C_ADDRESS"),
#             col("C_COMMENT"),
#         )
#         .agg([(col("volume").alias("revenue"), "sum")])
#         .sort(col("revenue"), desc=True)
#         .select(
#             col("O_CUSTKEY"),
#             col("C_NAME"),
#             col("revenue"),
#             col("C_ACCTBAL"),
#             col("N_NAME"),
#             col("C_ADDRESS"),
#             col("C_PHONE"),
#             col("C_COMMENT"),
#         )
#         .limit(20)
#     )

#     return daft_df


query_to_loaders = {
    1: [load_lineitem],
    2: [load_part, load_partsupp, load_supplier, load_nation, load_region],
    3: [load_lineitem, load_orders, load_customer],
    4: [load_lineitem, load_orders],
    5: [
        load_lineitem,
        load_orders,
        load_customer,
        load_nation,
        load_region,
        load_supplier,
    ],
    # 6: [load_lineitem],
    # 7: [load_lineitem, load_supplier, load_orders, load_customer, load_nation],
    # 8: [
    #     load_part,
    #     load_lineitem,
    #     load_supplier,
    #     load_orders,
    #     load_customer,
    #     load_nation,
    #     load_region,
    # ],
    # 9: [
    #     load_lineitem,
    #     load_orders,
    #     load_part,
    #     load_nation,
    #     load_partsupp,
    #     load_supplier,
    # ],
    # 10: [load_lineitem, load_orders, load_nation, load_customer],
    # 11: [load_partsupp, load_supplier, load_nation],
    # 12: [load_lineitem, load_orders],
    # 13: [load_customer, load_orders],
    # 14: [load_lineitem, load_part],
    # 15: [load_lineitem, load_supplier],
    # 16: [load_part, load_partsupp, load_supplier],
    # 17: [load_lineitem, load_part],
    # 18: [load_lineitem, load_orders, load_customer],
    # 19: [load_lineitem, load_part],
    # 20: [load_lineitem, load_part, load_nation, load_partsupp, load_supplier],
    # 21: [load_lineitem, load_orders, load_supplier, load_nation],
    # 22: [load_customer, load_orders],
}

query_to_runner = {
    1: q01,
    2: q02,
    3: q03,
    4: q04,
    5: q05,
    # 6: q06,
    # 7: q07,
    # 8: q08,
    # 9: q09,
    # 10: q10,
    # 11: q11,
    # 12: q12,
    # 13: q13,
    # 14: q14,
    # 15: q15,
    # 16: q16,
    # 17: q17,
    # 18: q18,
    # 19: q19,
    # 20: q20,
    # 21: q21,
    # 22: q22,
}


def get_query_answer(query: int, base_dir: str = ANSWERS_BASE_DIR) -> PandasDF:
    answer_df = pd.read_csv(
        os.path.join(base_dir, f"q{query}.out"),
        sep="|",
        parse_dates=True,
        infer_datetime_format=True,
    )
    return answer_df.rename(columns=lambda x: x.strip())


def test_results(q_num: int, result_df: PandasDF):
    answer = get_query_answer(q_num)

    for column_index in range(len(answer.columns)):
        column_name = answer.columns[column_index]
        column_data_type = answer.iloc[:, column_index].dtype

        s1 = result_df.iloc[:, column_index]
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


def run_queries(
    path,
    storage_options,
    queries,
    log_time=True,
    include_io=False,
    test_result=False,
    print_result=False
):
    print("Start data loading")
    total_start = time.time()
    for query in queries:
        loaders = query_to_loaders[query]
        for loader in loaders:
            loader(
                path,
                storage_options,
                include_io,
            )
    print(f"Data loading time (s): {time.time() - total_start}")
    total_start = time.time()
    for query in queries:
        try:
            t1 = time.time()
            result = query_to_runner[query](
                path,
                storage_options,
                include_io
            )
            result.collect()
            dur = time.time() - t1
            success = True
            if test_result:
                test_results(query, result)
            if print_result:
                print(result)
        except Exception as e:
            print("".join(traceback.TracebackException.from_exception(e).format()))
            dur = 0.0
            success = False
        finally:
            if log_time:
                append_row("daft", query, dur, daft.__version__, success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
    parser.add_argument(
        "--path", type=str, required=True, help="path to the TPC-H dataset."
    )
    parser.add_argument(
        "--storage_options",
        type=str,
        required=False,
        help="storage options json file.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="whitespace separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help="the endpoint of existing Ray cluster.",
    )
    parser.add_argument(
        "--log_time",
        action="store_true",
        help="log time metrics or not.",
    )
    parser.add_argument(
        "--include_io",
        action="store_true",
        help="include IO or not.",
    )
    parser.add_argument(
        "--test_result",
        action="store_true",
        help="test results with official answers.",
    )
    parser.add_argument(
        "--print_result",
        action="store_true",
        help="print result.",
    )

    args = parser.parse_args()

    # path to TPC-H data in parquet.
    print(f"Path: {args.path}")

    # credentials to access the datasource.
    storage_options = {}
    if args.storage_options is not None:
        with open(args.storage_options, "r") as fp:
            storage_options = json.load(fp)
    print(f"Storage options: {storage_options}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    print(f"Include IO: {args.include_io}")

    ray.init(address="auto")

    def get_daft_version() -> str:
        return daft.get_version()

    @ray.remote(num_cpus=1, scheduling_strategy="SPREAD")
    def warm_up_function():
        import time

        time.sleep(1)
        return get_daft_version()

    num_workers_to_warm = int(ray.cluster_resources()["CPU"])
    tasks = [warm_up_function.remote() for _ in range(num_workers_to_warm)]
    assert ray.get(tasks) == [get_daft_version() for _ in range(num_workers_to_warm)]
    del tasks

    print("Ray cluster warmed up")
    
    try:
        run_queries(
            args.path,
            storage_options,
            queries,
            args.log_time,
            args.include_io,
            args.test_result,
            args.print_result,
        )
    finally:
        ray.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()