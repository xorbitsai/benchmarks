import os
import sys
import argparse
import json
import time
import traceback
import datetime
from typing import Dict

import pandas as pd

import ray
import daft
from daft import DataFrame, col

from common_utils import log_time_fn, parse_common_arguments, print_result_fn

dataset_dict = {}


def load_lineitem(root: str):
    if "lineitem" not in dataset_dict:
        data_path = root + "/lineitem"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["lineitem"] = result
    else:
        result = dataset_dict["lineitem"]
    return result


def load_part(root: str):
    if "part" not in dataset_dict:
        data_path = root + "/part"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["part"] = result
    else:
        result = dataset_dict["part"]
    return result


def load_orders(root: str):
    if "orders" not in dataset_dict:
        data_path = root + "/orders"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["orders"] = result
    else:
        result = dataset_dict["orders"]
    return result


def load_customer(root: str):
    if "customer" not in dataset_dict:
        data_path = root + "/customer"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["customer"] = result
    else:
        result = dataset_dict["customer"]
    return result


def load_nation(root: str):
    if "nation" not in dataset_dict:
        data_path = root + "/nation"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["nation"] = result
    else:
        result = dataset_dict["nation"]
    return result


def load_region(root: str):
    if "region" not in dataset_dict:
        data_path = root + "/region"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["region"] = result
    else:
        result = dataset_dict["region"]
    return result


def load_supplier(root: str):
    if "supplier" not in dataset_dict:
        data_path = root + "/supplier"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["supplier"] = result
    else:
        result = dataset_dict["supplier"]
    return result


def load_partsupp(root: str):
    if "partsupp" not in dataset_dict:
        data_path = root + "/partsupp"
        df = daft.read_parquet(data_path)
        result = df
        dataset_dict["partsupp"] = result
    else:
        result = dataset_dict["partsupp"]
    return result


def q01(root: str):
    lineitem = load_lineitem(root)

    date = datetime.date(1998, 9, 2)

    discounted_price = col("L_EXTENDEDPRICE") * (1 - col("L_DISCOUNT"))
    taxed_discounted_price = discounted_price * (1 + col("L_TAX"))
    daft_df = (
        lineitem.where(col("L_SHIPDATE") <= date)
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


def q02(root: str) -> DataFrame:
    part = load_part(root)
    partsupp = load_partsupp(root)
    supplier = load_supplier(root)
    nation = load_nation(root)
    region = load_region(root)

    size = 15
    p_type = "BRASS"
    region_name = "EUROPE"

    europe = (
        region.where(col("R_NAME") == region_name)
        .join(nation, left_on=col("R_REGIONKEY"), right_on=col("N_REGIONKEY"))
        .join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(partsupp, left_on=col("S_SUPPKEY"), right_on=col("PS_SUPPKEY"))
    )

    brass = part.where((col("P_SIZE") == size) & col("P_TYPE").str.endswith(p_type)).join(
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


def q03(root: str) -> DataFrame:
    def decrease(x, y):
        return x * (1 - y)

    customer = load_customer(root)
    orders = load_orders(root)
    lineitem = load_lineitem(root)
    
    mktsegment = "HOUSEHOLD"
    date = datetime.date(1995, 3, 15)

    customer = customer.where(col("C_MKTSEGMENT") == mktsegment)
    orders = orders.where(col("O_ORDERDATE") < date)
    lineitem = lineitem.where(col("L_SHIPDATE") >= date)

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


def q04(root: str) -> DataFrame:
    orders = load_orders(root)
    lineitem = load_lineitem(root)

    date1 = datetime.date(1993, 8, 1)
    date2 = datetime.date(1993, 11, 1)

    orders = orders.where(
        (col("O_ORDERDATE") >= date1) & (col("O_ORDERDATE") < date2)
    )

    lineitems = lineitem.where(col("L_COMMITDATE") < col("L_RECEIPTDATE")).select(col("L_ORDERKEY")).distinct()

    daft_df = (
        lineitems.join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .groupby(col("O_ORDERPRIORITY"))
        .agg([(col("L_ORDERKEY").alias("order_count"), "count")])
        .sort(col("O_ORDERPRIORITY"))
    )
    return daft_df


def q05(root: str) -> DataFrame:
    orders = load_orders(root)
    region = load_region(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    lineitem = load_lineitem(root)
    customer = load_customer(root)

    region_name = "ASIA"
    date1 = datetime.date(1996, 1, 1)
    date2 = datetime.date(1997, 1, 1)

    orders = orders.where(
        (col("O_ORDERDATE") >= datetime.date(1994, 1, 1)) & (col("O_ORDERDATE") < datetime.date(1995, 1, 1))
    )
    region = region.where(col("R_NAME") == region_name)
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


def q06(root: str) -> DataFrame:
    lineitem = load_lineitem(root)
    date1 = datetime.date(1996, 1, 1)
    date2 = datetime.date(1997, 1, 1)
    daft_df = lineitem.where(
        (col("L_SHIPDATE") >= date1)
        & (col("L_SHIPDATE") < date2)
        & (col("L_DISCOUNT") >= 0.05)
        & (col("L_DISCOUNT") <= 0.07)
        & (col("L_QUANTITY") < 24)
    ).sum(col("L_EXTENDEDPRICE") * col("L_DISCOUNT"))
    return daft_df


def q07(root: str) -> DataFrame:
    linitem = load_lineitem(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    customer = load_customer(root)
    orders = load_orders(root)

    nation1 = "FRANCE"
    nation2 = "GERMANY"
    
    def decrease(x, y):
        return x * (1 - y)

    lineitem = linitem.where(
        (col("L_SHIPDATE") >= datetime.date(1995, 1, 1)) & (col("L_SHIPDATE") <= datetime.date(1996, 12, 31))
    )
    nation = nation.where((col("N_NAME") == nation1) | (col("N_NAME") == nation2))

    supNation = (
        nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))
        .join(lineitem, left_on=col("S_SUPPKEY"), right_on=col("L_SUPPKEY"))
        .select(
            col("N_NAME").alias("supp_nation"),
            col("L_ORDERKEY"),
            col("L_EXTENDEDPRICE"),
            col("L_DISCOUNT"),
            col("L_SHIPDATE"),
        )
    )

    daft_df = (
        nation.join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
        .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("N_NAME").alias("cust_nation"), col("O_ORDERKEY"))
        .join(supNation, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .where(
            ((col("supp_nation") == "FRANCE") & (col("cust_nation") == "GERMANY"))
            | ((col("supp_nation") == "GERMANY") & (col("cust_nation") == "FRANCE"))
        )
        .select(
            col("supp_nation"),
            col("cust_nation"),
            col("L_SHIPDATE").dt.year().alias("l_year"),
            decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
        )
        .groupby(col("supp_nation"), col("cust_nation"), col("l_year"))
        .agg([(col("volume").alias("revenue"), "sum")])
        .sort(by=["supp_nation", "cust_nation", "l_year"])
    )
    return daft_df


def q08(root: str) -> DataFrame:
    lineitem = load_lineitem(root)
    part = load_part(root)
    region = load_region(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    customer = load_customer(root)
    orders = load_orders(root)

    nation_name = "BRAZIL"
    region_name = "AMERICA"
    p_type = "ECONOMY ANODIZED STEEL"
    
    def decrease(x, y):
        return x * (1 - y)

    region = region.where(col("R_NAME") == region_name)
    orders = orders.where(
        (col("O_ORDERDATE") < datetime.date(1997, 1, 1)) & (col("O_ORDERDATE") >= datetime.date(1995, 1, 1))
    )
    part = part.where(col("P_TYPE") == p_type)

    nat = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))

    line = (
        lineitem.select(
            col("L_PARTKEY"),
            col("L_SUPPKEY"),
            col("L_ORDERKEY"),
            decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
        )
        .join(part, left_on=col("L_PARTKEY"), right_on=col("P_PARTKEY"))
        .join(nat, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
    )

    daft_df = (
        nation.join(region, left_on=col("N_REGIONKEY"), right_on=col("R_REGIONKEY"))
        .select(col("N_NATIONKEY"))
        .join(customer, left_on=col("N_NATIONKEY"), right_on=col("C_NATIONKEY"))
        .select(col("C_CUSTKEY"))
        .join(orders, left_on=col("C_CUSTKEY"), right_on=col("O_CUSTKEY"))
        .select(col("O_ORDERKEY"), col("O_ORDERDATE"))
        .join(line, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_ORDERDATE").dt.year().alias("o_year"),
            col("volume"),
            (col("N_NAME") == nation_name).if_else(col("volume"), 0.0).alias("case_volume"),
        )
        .groupby(col("o_year"))
        .agg([(col("case_volume").alias("case_volume_sum"), "sum"), (col("volume").alias("volume_sum"), "sum")])
        .select(col("o_year"), col("case_volume_sum") / col("volume_sum"))
        .sort(col("o_year"))
    )

    return daft_df


def q09(root: str) -> DataFrame:
    lineitem = load_lineitem(root)
    orders = load_orders(root)
    nation = load_nation(root)
    part = load_part(root)
    supplier = load_supplier(root)
    partsupp = load_partsupp(root)

    p_name = "ghost"

    def expr(x, y, v, w):
        return x * (1 - y) - (v * w)

    linepart = part.where(col("P_NAME").str.contains(p_name)).join(
        lineitem, left_on=col("P_PARTKEY"), right_on=col("L_PARTKEY")
    )
    natsup = nation.join(supplier, left_on=col("N_NATIONKEY"), right_on=col("S_NATIONKEY"))

    daft_df = (
        linepart.join(natsup, left_on=col("L_SUPPKEY"), right_on=col("S_SUPPKEY"))
        .join(partsupp, left_on=[col("L_SUPPKEY"), col("P_PARTKEY")], right_on=[col("PS_SUPPKEY"), col("PS_PARTKEY")])
        .join(orders, left_on=col("L_ORDERKEY"), right_on=col("O_ORDERKEY"))
        .select(
            col("N_NAME"),
            col("O_ORDERDATE").dt.year().alias("o_year"),
            expr(col("L_EXTENDEDPRICE"), col("L_DISCOUNT"), col("PS_SUPPLYCOST"), col("L_QUANTITY")).alias("amount"),
        )
        .groupby(col("N_NAME"), col("o_year"))
        .agg([(col("amount"), "sum")])
        .sort(by=["N_NAME", "o_year"], desc=[False, True])
    )

    return daft_df


def q10(root: str) -> DataFrame:
    lineitem = load_lineitem(root)
    orders = load_orders(root)
    nation = load_nation(root)
    customer = load_customer(root)
    
    date1 = datetime.date(1994, 11, 1)
    date2 = datetime.date(1995, 2, 1)
    def decrease(x, y):
        return x * (1 - y)

    daft_df = (
        orders.where(
            (col("O_ORDERDATE") < date2) & (col("O_ORDERDATE") >= date1)
        )
        .join(customer, left_on=col("O_CUSTKEY"), right_on=col("C_CUSTKEY"))
        .join(nation, left_on=col("C_NATIONKEY"), right_on=col("N_NATIONKEY"))
        .join(lineitem, left_on=col("O_ORDERKEY"), right_on=col("L_ORDERKEY"))
        .select(
            col("O_CUSTKEY"),
            col("C_NAME"),
            decrease(col("L_EXTENDEDPRICE"), col("L_DISCOUNT")).alias("volume"),
            col("C_ACCTBAL"),
            col("N_NAME"),
            col("C_ADDRESS"),
            col("C_PHONE"),
            col("C_COMMENT"),
        )
        .groupby(
            col("O_CUSTKEY"),
            col("C_NAME"),
            col("C_ACCTBAL"),
            col("C_PHONE"),
            col("N_NAME"),
            col("C_ADDRESS"),
            col("C_COMMENT"),
        )
        .agg([(col("volume").alias("revenue"), "sum")])
        .sort(col("revenue"), desc=True)
        .select(
            col("O_CUSTKEY"),
            col("C_NAME"),
            col("revenue"),
            col("C_ACCTBAL"),
            col("N_NAME"),
            col("C_ADDRESS"),
            col("C_PHONE"),
            col("C_COMMENT"),
        )
        .limit(20)
    )

    return daft_df


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
    6: [load_lineitem],
    7: [load_lineitem, load_supplier, load_orders, load_customer, load_nation],
    8: [
        load_part,
        load_lineitem,
        load_supplier,
        load_orders,
        load_customer,
        load_nation,
        load_region,
    ],
    9: [
        load_lineitem,
        load_orders,
        load_part,
        load_nation,
        load_partsupp,
        load_supplier,
    ],
    10: [load_lineitem, load_orders, load_nation, load_customer],
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
    6: q06,
    7: q07,
    8: q08,
    9: q09,
    10: q10,
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


def run_queries(
    path,
    queries,
    log_time=True,
    print_result=False,
    include_io=False,
):
    version = daft.__version__
    data_start_time = time.time()
    for query in queries:
        loaders = query_to_loaders[query]
        for loader in loaders:
            loader(path)
    print(f"Total data loading time (s): {time.time() - data_start_time}")
    
    total_start = time.time()
    for query in queries:
        try:
            start_time = time.time()
            result = query_to_runner[query](path)
            result = result.to_pandas()
            without_io_time = time.time() - start_time
            success = True
            if print_result:
                print_result_fn("daft", result, query)
        except Exception as e:
            print("".join(traceback.TracebackException.from_exception(e).format()))
            without_io_time = 0.0
            success = False
        finally:
            pass
        if log_time:
            log_time_fn("daft", query, version=version, without_io_time=without_io_time, success=success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
    # aws settings
    parser.add_argument("--account", type=str, help="AWS access id")
    parser.add_argument("--key", type=str, help="AWS secret access key")
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help="the endpoint of existing Ray cluster.",
    )
    parser = parse_common_arguments(parser)

    args = parser.parse_args()

    # path to TPC-H data in parquet.
    print(f"Path: {args.path}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    print(f"Include IO: {args.include_io}")

    if "s3://" in args.path:
        import boto3
        s3_client = boto3.client(
            's3', 
            aws_access_key_id=args.account,
            aws_secret_access_key=args.key
        )

    ray.init(address="auto")
    
    try:
        run_queries(
            args.path,
            queries,
            args.log_time,
            args.print_result,
            args.include_io,
        )
    finally:
        ray.shutdown()
    ray.shutdown()


if __name__ == "__main__":
    main()