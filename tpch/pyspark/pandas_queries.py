import os
import sys
import argparse
import time
import traceback
from typing import Dict, List, Optional, Set, Union

import pandas as pd
from pandas.core.frame import DataFrame as PandasDF

import pyspark
import pyspark.pandas as ps
from pyspark.sql import SparkSession


TIMINGS_FILE = "time.out"
dataset_dict = {}
spark: SparkSession = None


def append_row(solution: str, q: str, secs: float, version: str, success=True):
    with open(TIMINGS_FILE, "a") as f:
        if f.tell() == 0:
            f.write("solution,version,query_no,duration[s],success\n")
        f.write(f"{solution},{version},{q},{secs},{success}\n")
        print(f"{solution},{version},{q},{secs},{success}")


def load_lineitem(root: str):
    if "lineitem" not in dataset_dict:
        data_path = root + "/lineitem"
        df = spark.read.parquet(data_path)
        df = ps.DataFrame(df)
        dataset_dict["lineitem"] = df
    else:
        df = dataset_dict["lineitem"]
    return df


def load_part(root: str):
    if "part" not in dataset_dict:
        data_path = root + "/part"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["part"] = df
    else:
        df = dataset_dict["part"]
    return df


def load_orders(root: str):
    if "orders" not in dataset_dict:
        data_path = root + "/orders"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["orders"] = df
    else:
        df = dataset_dict["orders"]
    return df


def load_customer(root: str):
    if "customer" not in dataset_dict:
        data_path = root + "/customer"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["customer"] = df
    else:
        df = dataset_dict["customer"]
    return df


def load_nation(root: str):
    if "nation" not in dataset_dict:
        data_path = root + "/nation"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["nation"] = df
    else:
        df = dataset_dict["nation"]
    return df


def load_region(root: str):
    if "region" not in dataset_dict:
        data_path = root + "/region"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["region"] = df
    else:
        df = dataset_dict["region"]
    return df


def load_supplier(root: str):
    if "supplier" not in dataset_dict:
        data_path = root + "/supplier"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["supplier"] = df
    else:
        df = dataset_dict["supplier"]
    return df


def load_partsupp(root: str):
    if "partsupp" not in dataset_dict:
        data_path = root + "/partsupp"
        df = spark.read.parquet(data_path).cache()
        df = ps.DataFrame(df)
        dataset_dict["partsupp"] = df
    else:
        df = dataset_dict["partsupp"]
    return df


def load_lineitem(root: str, include_io: bool = False):
    if "lineitem" not in dataset_dict or include_io:
        data_path = root + "/lineitem"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["lineitem"] = df
    else:
        df = dataset_dict["lineitem"]
    return df


def load_part(root: str, include_io: bool = False):
    if "part" not in dataset_dict or include_io:
        data_path = root + "/part"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["part"] = df
    else:
        df = dataset_dict["part"]
    return df


def load_orders(root: str, include_io: bool = False):
    if "orders" not in dataset_dict or include_io:
        data_path = root + "/orders"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["orders"] = df
    else:
        df = dataset_dict["orders"]
    return df


def load_customer(root: str, include_io: bool = False):
    if "customer" not in dataset_dict or include_io:
        data_path = root + "/customer"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["customer"] = df
    else:
        df = dataset_dict["customer"]
    return df


def load_nation(root: str, include_io: bool = False):
    if "nation" not in dataset_dict or include_io:
        data_path = root + "/nation"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["nation"] = df
    else:
        df = dataset_dict["nation"]
    return df


def load_region(root: str, include_io: bool = False):
    if "region" not in dataset_dict or include_io:
        data_path = root + "/region"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["region"] = df
    else:
        df = dataset_dict["region"]
    return df


def load_supplier(root: str, include_io: bool = False):
    if "supplier" not in dataset_dict or include_io:
        data_path = root + "/supplier"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["supplier"] = df
    else:
        df = dataset_dict["supplier"]
    return df


def load_partsupp(root: str, include_io: bool = False):
    if "partsupp" not in dataset_dict or include_io:
        data_path = root + "/partsupp"
        df = spark.read.parquet(data_path).persist(StorageLevel.MEMORY_ONLY)
        df = ps.DataFrame(df)
        dataset_dict["partsupp"] = df
    else:
        df = dataset_dict["partsupp"]
    return df


def q01(root: str, include_io: bool = False):
    lineitem = load_lineitem(root, include_io)
    date = pd.Timestamp("1998-09-02")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "L_ORDERKEY",
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "L_DISCOUNT",
            "L_TAX",
            "L_RETURNFLAG",
            "L_LINESTATUS",
            "L_SHIPDATE",
        ],
    ]
    sel = lineitem_filtered.L_SHIPDATE <= date
    lineitem_filtered = lineitem_filtered[sel]
    lineitem_filtered["AVG_QTY"] = lineitem_filtered.L_QUANTITY
    lineitem_filtered["AVG_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE
    lineitem_filtered["DISC_PRICE"] = lineitem_filtered.L_EXTENDEDPRICE * (
        1 - lineitem_filtered.L_DISCOUNT
    )
    lineitem_filtered["CHARGE"] = (
        lineitem_filtered.L_EXTENDEDPRICE
        * (1 - lineitem_filtered.L_DISCOUNT)
        * (1 + lineitem_filtered.L_TAX)
    )
    total = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"]).agg(
        {
            "L_QUANTITY": "sum",
            "L_EXTENDEDPRICE": "sum",
            "DISC_PRICE": "sum",
            "CHARGE": "sum",
            "AVG_QTY": "mean",
            "AVG_PRICE": "mean",
            "L_DISCOUNT": "mean",
            "L_ORDERKEY": "count",
        }
    )

    total = total.reset_index().sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    return total


def q02(root: str, include_io: bool = False):
    nation = load_nation(root, include_io)
    region = load_region(root, include_io)
    supplier = load_supplier(root, include_io)
    part = load_part(root, include_io)
    partsupp = load_partsupp(root, include_io)

    nation_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME", "N_REGIONKEY"]]
    region_filtered = region[(region["R_NAME"] == "EUROPE")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    r_n_merged = nation_filtered.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    r_n_merged = r_n_merged.loc[:, ["N_NATIONKEY", "N_NAME"]]
    supplier_filtered = supplier.loc[
        :,
        [
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_NATIONKEY",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    s_r_n_merged = r_n_merged.merge(
        supplier_filtered, left_on="N_NATIONKEY", right_on="S_NATIONKEY", how="inner"
    )
    s_r_n_merged = s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_SUPPKEY",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
        ],
    ]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY", "PS_SUPPLYCOST"]]
    ps_s_r_n_merged = s_r_n_merged.merge(
        partsupp_filtered, left_on="S_SUPPKEY", right_on="PS_SUPPKEY", how="inner"
    )
    ps_s_r_n_merged = ps_s_r_n_merged.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_PARTKEY",
            "PS_SUPPLYCOST",
        ],
    ]
    part_filtered = part.loc[:, ["P_PARTKEY", "P_MFGR", "P_SIZE", "P_TYPE"]]
    part_filtered = part_filtered[
        (part_filtered["P_SIZE"] == 15) & (part_filtered["P_TYPE"].str.endswith("BRASS"))
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_MFGR"]]
    merged_df = part_filtered.merge(
        ps_s_r_n_merged, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    merged_df = merged_df.loc[
        :,
        [
            "N_NAME",
            "S_NAME",
            "S_ADDRESS",
            "S_PHONE",
            "S_ACCTBAL",
            "S_COMMENT",
            "PS_SUPPLYCOST",
            "P_PARTKEY",
            "P_MFGR",
        ],
    ]
    min_values = merged_df.groupby("P_PARTKEY", as_index=False)["PS_SUPPLYCOST"].min()
    min_values.columns = ["P_PARTKEY_CPY", "MIN_SUPPLYCOST"]
    merged_df = merged_df.merge(
        min_values,
        left_on=["P_PARTKEY", "PS_SUPPLYCOST"],
        right_on=["P_PARTKEY_CPY", "MIN_SUPPLYCOST"],
        how="inner",
    )
    total = merged_df.loc[
        :,
        [
            "S_ACCTBAL",
            "S_NAME",
            "N_NAME",
            "P_PARTKEY",
            "P_MFGR",
            "S_ADDRESS",
            "S_PHONE",
            "S_COMMENT",
        ],
    ]
    total = total.sort_values(
        by=["S_ACCTBAL", "N_NAME", "S_NAME", "P_PARTKEY"],
        ascending=[False, True, True, True],
    )
    return total


# @time_collector
# def q03(lineitem: ps.DataFrame, orders: ps.DataFrame, customer: ps.DataFrame):
#     date = pd.Timestamp("1995-03-04")
#     lineitem_filtered = lineitem.loc[
#         :, ["L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
#     ]
#     orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY"]]
#     customer_filtered = customer.loc[:, ["C_MKTSEGMENT", "C_CUSTKEY"]]
#     lsel = lineitem_filtered.L_SHIPDATE > date
#     osel = orders_filtered.O_ORDERDATE < date
#     csel = customer_filtered.C_MKTSEGMENT == "HOUSEHOLD"
#     flineitem = lineitem_filtered[lsel]
#     forders = orders_filtered[osel]
#     fcustomer = customer_filtered[csel]
#     jn1 = fcustomer.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
#     jn2 = jn1.merge(flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
#     jn2["TMP"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
#     total = (
#         jn2.groupby(["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False)["TMP"]
#         .sum()
#         .sort_values(["TMP"], ascending=False)
#     )
#     res = total.loc[:, ["L_ORDERKEY", "TMP", "O_ORDERDATE", "O_SHIPPRIORITY"]]
#     return res.head(10)


def q04(root: str, include_io: bool = False):
    lineitem = load_lineitem(root, include_io)
    orders = load_orders(root, include_io)

    date1 = pd.Timestamp("1993-10-01")
    date2 = pd.Timestamp("1993-07-01")
    lsel = lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE
    osel = (orders.O_ORDERDATE < date1) & (orders.O_ORDERDATE >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"].count()
    return total


# @time_collector
# def q05(
#     lineitem: ps.DataFrame,
#     orders: ps.DataFrame,
#     customer: ps.DataFrame,
#     nation: ps.DataFrame,
#     region: ps.DataFrame,
#     supplier: ps.DataFrame,
# ):
#     date1 = pd.Timestamp("1996-01-01")
#     date2 = pd.Timestamp("1997-01-01")
#     rsel = region.R_NAME == "ASIA"
#     osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
#     forders = orders[osel]
#     fregion = region[rsel]
#     jn1 = fregion.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
#     jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
#     jn3 = jn2.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
#     jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
#     jn5 = supplier.merge(
#         jn4, left_on=["S_SUPPKEY", "S_NATIONKEY"], right_on=["L_SUPPKEY", "N_NATIONKEY"]
#     )
#     jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
#     gb = jn5.groupby("N_NAME", as_index=False)["TMP"].sum()
#     total = gb.sort_values("TMP", ascending=False)
#     return total


# @time_collector
# def q06(lineitem: ps.DataFrame):
#     date1 = pd.Timestamp("1996-01-01")
#     date2 = pd.Timestamp("1997-01-01")
#     lineitem_filtered = lineitem.loc[
#         :, ["L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
#     ]
#     sel = (
#         (lineitem_filtered.L_SHIPDATE >= date1)
#         & (lineitem_filtered.L_SHIPDATE < date2)
#         & (lineitem_filtered.L_DISCOUNT >= 0.08)
#         & (lineitem_filtered.L_DISCOUNT <= 0.1)
#         & (lineitem_filtered.L_QUANTITY < 24)
#     )
#     flineitem = lineitem_filtered[sel]
#     total = (flineitem.L_EXTENDEDPRICE * flineitem.L_DISCOUNT).sum()
#     return total


# @time_collector
# def q07(lineitem, supplier, orders, customer, nation):
#     """This version is faster than q07_old. Keeping the old one for reference"""
#     lineitem_filtered = lineitem[
#         (lineitem["L_SHIPDATE"] >= pd.Timestamp("1995-01-01"))
#         & (lineitem["L_SHIPDATE"] < pd.Timestamp("1997-01-01"))
#     ]
#     lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].dt.year
#     lineitem_filtered["VOLUME"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
#         1.0 - lineitem_filtered["L_DISCOUNT"]
#     )
#     lineitem_filtered = lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY", "L_YEAR", "VOLUME"]]
#     supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
#     orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
#     customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
#     n1 = nation[(nation["N_NAME"] == "FRANCE")].loc[:, ["N_NATIONKEY", "N_NAME"]]
#     n2 = nation[(nation["N_NAME"] == "GERMANY")].loc[:, ["N_NATIONKEY", "N_NAME"]]

#     # ----- do nation 1 -----
#     N1_C = customer_filtered.merge(n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     N1_C = N1_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
#         columns={"N_NAME": "CUST_NATION"}
#     )
#     N1_C_O = N1_C.merge(orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner")
#     N1_C_O = N1_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

#     N2_S = supplier_filtered.merge(n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     N2_S = N2_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
#         columns={"N_NAME": "SUPP_NATION"}
#     )
#     N2_S_L = N2_S.merge(lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner")
#     N2_S_L = N2_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

#     total1 = N1_C_O.merge(N2_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner")
#     total1 = total1.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

#     # ----- do nation 2 -----
#     N2_C = customer_filtered.merge(n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     N2_C = N2_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
#         columns={"N_NAME": "CUST_NATION"}
#     )
#     N2_C_O = N2_C.merge(orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner")
#     N2_C_O = N2_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

#     N1_S = supplier_filtered.merge(n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     N1_S = N1_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
#         columns={"N_NAME": "SUPP_NATION"}
#     )
#     N1_S_L = N1_S.merge(lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner")
#     N1_S_L = N1_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

#     total2 = N2_C_O.merge(N1_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner")
#     total2 = total2.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

#     # concat results
#     total = ps.concat([total1, total2])

#     total = (
#         total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False)
#         .agg(REVENUE=ps.NamedAgg(column="VOLUME", aggfunc="sum"))
#         .sort_values(by=["SUPP_NATION", "CUST_NATION", "L_YEAR"], ascending=[True, True, True])
#     )
#     # skip sort when Mars groupby does sort already
#     # total = total.sort_values(
#     #     by=["SUPP_NATION", "CUST_NATION", "L_YEAR"], ascending=[True, True, True]
#     # )
#     return total


# @time_collector
# def q08(part, lineitem, supplier, orders, customer, nation, region):
#     part_filtered = part[(part["P_TYPE"] == "ECONOMY ANODIZED STEEL")]
#     part_filtered = part_filtered.loc[:, ["P_PARTKEY"]]
#     lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_SUPPKEY", "L_ORDERKEY"]]
#     lineitem_filtered["VOLUME"] = lineitem["L_EXTENDEDPRICE"] * (1.0 - lineitem["L_DISCOUNT"])
#     total = part_filtered.merge(
#         lineitem_filtered, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
#     )
#     total = total.loc[:, ["L_SUPPKEY", "L_ORDERKEY", "VOLUME"]]
#     supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
#     total = total.merge(supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner")
#     total = total.loc[:, ["L_ORDERKEY", "VOLUME", "S_NATIONKEY"]]
#     orders_filtered = orders[
#         (orders["O_ORDERDATE"] >= pd.Timestamp("1995-01-01"))
#         & (orders["O_ORDERDATE"] < pd.Timestamp("1997-01-01"))
#     ]
#     orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].dt.year
#     orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_YEAR"]]
#     total = total.merge(orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner")
#     total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_CUSTKEY", "O_YEAR"]]
#     customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
#     total = total.merge(customer_filtered, left_on="O_CUSTKEY", right_on="C_CUSTKEY", how="inner")
#     total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "C_NATIONKEY"]]
#     n1_filtered = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
#     n2_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME"]].rename(columns={"N_NAME": "NATION"})
#     total = total.merge(n1_filtered, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "N_REGIONKEY"]]
#     total = total.merge(n2_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     total = total.loc[:, ["VOLUME", "O_YEAR", "N_REGIONKEY", "NATION"]]
#     region_filtered = region[(region["R_NAME"] == "AMERICA")]
#     region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
#     total = total.merge(region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner")
#     total = total.loc[:, ["VOLUME", "O_YEAR", "NATION"]]

#     def udf(df):
#         demonimator = df["VOLUME"].sum()
#         df = df[df["NATION"] == "BRAZIL"]
#         numerator = df["VOLUME"].sum()
#         return numerator / demonimator

#     total = total.groupby("O_YEAR", as_index=False).apply(udf)
#     # add reset_index here to turn Series to a DataFrame, same as the q13
#     total = total.reset_index()
#     total.columns = ["O_YEAR", "MKT_SHARE"]
#     total = total.sort_values(by=["O_YEAR"], ascending=[True])
#     return total


# @time_collector
# def q09(lineitem, orders, part, nation, partsupp, supplier):
#     psel = part.P_NAME.str.contains("ghost")
#     fpart = part[psel]
#     jn1 = lineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
#     jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
#     jn3 = jn2.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
#     jn4 = partsupp.merge(
#         jn3, left_on=["PS_PARTKEY", "PS_SUPPKEY"], right_on=["L_PARTKEY", "L_SUPPKEY"]
#     )
#     jn5 = jn4.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
#     jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1 - jn5.L_DISCOUNT) - (
#         (1 * jn5.PS_SUPPLYCOST) * jn5.L_QUANTITY
#     )
#     jn5["O_YEAR"] = jn5.O_ORDERDATE.dt.year
#     gb = jn5.groupby(["N_NAME", "O_YEAR"], as_index=False)["TMP"].sum()
#     total = gb.sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
#     return total


# @time_collector
# def q10(lineitem, orders, customer, nation):
#     date1 = pd.Timestamp("1994-11-01")
#     date2 = pd.Timestamp("1995-02-01")
#     osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
#     lsel = lineitem.L_RETURNFLAG == "R"
#     forders = orders[osel]
#     flineitem = lineitem[lsel]
#     jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
#     jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
#     jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
#     jn3["TMP"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
#     gb = jn3.groupby(
#         [
#             "C_CUSTKEY",
#             "C_NAME",
#             "C_ACCTBAL",
#             "C_PHONE",
#             "N_NAME",
#             "C_ADDRESS",
#             "C_COMMENT",
#         ],
#         as_index=False,
#     )["TMP"].sum()
#     total = gb.sort_values("TMP", ascending=False)
#     return total.head(20)


# @time_collector
# def q11(partsupp, supplier, nation):
#     partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
#     partsupp_filtered["TOTAL_COST"] = partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
#     supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
#     ps_supp_merge = partsupp_filtered.merge(
#         supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
#     )
#     ps_supp_merge = ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
#     nation_filtered = nation[(nation["N_NAME"] == "GERMANY")]
#     nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
#     ps_supp_n_merge = ps_supp_merge.merge(
#         nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
#     )
#     ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
#     sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001
#     total = ps_supp_n_merge.groupby(["PS_PARTKEY"], as_index=False).agg(
#         VALUE=ps.NamedAgg(column="TOTAL_COST", aggfunc="sum")
#     )
#     total = total[total["VALUE"] > sum_val]
#     total = total.sort_values("VALUE", ascending=False)
#     return total


# @time_collector
# def q12(lineitem, orders):
#     date1 = pd.Timestamp("1994-01-01")
#     date2 = pd.Timestamp("1995-01-01")
#     sel = (
#         (lineitem.L_RECEIPTDATE < date2)
#         & (lineitem.L_COMMITDATE < date2)
#         & (lineitem.L_SHIPDATE < date2)
#         & (lineitem.L_SHIPDATE < lineitem.L_COMMITDATE)
#         & (lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE)
#         & (lineitem.L_RECEIPTDATE >= date1)
#         & ((lineitem.L_SHIPMODE == "MAIL") | (lineitem.L_SHIPMODE == "SHIP"))
#     )
#     flineitem = lineitem[sel]
#     jn = flineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")

#     def g1(x):
#         return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

#     def g2(x):
#         return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

#     total = jn.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
#     total = total.reset_index()  # reset index to keep consistency with pandas
#     total = total.sort_values("L_SHIPMODE")
#     return total


# @time_collector
# def q13(customer, orders):
#     customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
#     orders_filtered = orders[~orders["O_COMMENT"].str.contains(r"special[\S|\s]*requests")]
#     orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
#     c_o_merged = customer_filtered.merge(
#         orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
#     )
#     c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]
#     count_df = c_o_merged.groupby(["C_CUSTKEY"], as_index=False).agg(
#         C_COUNT=ps.NamedAgg(column="O_ORDERKEY", aggfunc="count")
#     )
#     total = count_df.groupby(["C_COUNT"], as_index=False).size().reset_index()
#     total.columns = ["C_COUNT", "CUSTDIST"]
#     total = total.sort_values(by=["CUSTDIST", "C_COUNT"], ascending=[False, False])
#     return total


# @time_collector
# def q14(lineitem: ps.DataFrame, part: ps.DataFrame):
#     startDate = pd.Timestamp("1994-03-01")
#     endDate = pd.Timestamp("1994-04-01")
#     p_type_like = "PROMO"
#     part_filtered = part.loc[:, ["P_PARTKEY", "P_TYPE"]]
#     lineitem_filtered = lineitem.loc[
#         :, ["L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE", "L_PARTKEY"]
#     ]
#     sel = (lineitem_filtered.L_SHIPDATE >= startDate) & (lineitem_filtered.L_SHIPDATE < endDate)
#     flineitem = lineitem_filtered[sel]
#     jn = flineitem.merge(part_filtered, left_on="L_PARTKEY", right_on="P_PARTKEY")
#     jn["TMP"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
#     total = jn[jn.P_TYPE.str.startswith(p_type_like)].TMP.sum() * 100 / jn.TMP.sum()
#     return total


# @time_collector
# def q15(lineitem, supplier):
#     lineitem_filtered = lineitem[
#         (lineitem["L_SHIPDATE"] >= pd.Timestamp("1996-01-01"))
#         & (lineitem["L_SHIPDATE"] < (pd.Timestamp("1996-01-01") + pd.DateOffset(months=3)))
#     ]
#     lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
#         1.0 - lineitem_filtered["L_DISCOUNT"]
#     )
#     lineitem_filtered = lineitem_filtered.loc[:, ["L_SUPPKEY", "REVENUE_PARTS"]]
#     # .agg(TOTAL_REVENUE=pd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))\
#     revenue_table = lineitem_filtered.groupby("L_SUPPKEY", as_index=False).sum()
#     revenue_table.columns = ["L_SUPPKEY", "TOTAL_REVENUE"]
#     revenue_table = revenue_table.rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})

#     max_revenue = revenue_table["TOTAL_REVENUE"].max()
#     revenue_table = revenue_table[revenue_table["TOTAL_REVENUE"] == max_revenue]
#     supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE"]]
#     total = supplier_filtered.merge(
#         revenue_table, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
#     )
#     total = total.loc[:, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]]
#     return total


# @time_collector
# def q16(part, partsupp, supplier):
#     part_filtered = part[
#         (part["P_BRAND"] != "Brand#45")
#         & (~part["P_TYPE"].str.contains("^MEDIUM POLISHED"))
#         & part["P_SIZE"].isin([49, 14, 23, 45, 19, 3, 36, 9])
#     ]
#     part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_BRAND", "P_TYPE", "P_SIZE"]]
#     partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
#     total = part_filtered.merge(
#         partsupp_filtered, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
#     )
#     total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
#     supplier_filtered = supplier[supplier["S_COMMENT"].str.contains(r"Customer(\S|\s)*Complaints")]
#     supplier_filtered = supplier_filtered.loc[:, ["S_SUPPKEY"]].drop_duplicates()
#     # left merge to select only PS_SUPPKEY values not in supplier_filtered
#     total = total.merge(supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left")
#     total = total[total["S_SUPPKEY"].isna()]
#     total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
#     total = total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)["PS_SUPPKEY"].nunique()
#     total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
#     total = total.sort_values(
#         by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
#         ascending=[False, True, True, True],
#     )
#     return total


# @time_collector
# def q17(lineitem, part):
#     left = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY", "L_EXTENDEDPRICE"]]
#     right = part[((part["P_BRAND"] == "Brand#23") & (part["P_CONTAINER"] == "MED BOX"))]
#     right = right.loc[:, ["P_PARTKEY"]]
#     line_part_merge = left.merge(right, left_on="L_PARTKEY", right_on="P_PARTKEY", how="inner")
#     line_part_merge = line_part_merge.loc[:, ["L_QUANTITY", "L_EXTENDEDPRICE", "P_PARTKEY"]]
#     lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY"]]
#     lineitem_avg = lineitem_filtered.groupby(["L_PARTKEY"], as_index=False).mean()
#     lineitem_avg.columns = ["L_PARTKEY", "avg"]
#     lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
#     lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
#     total = line_part_merge.merge(
#         lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
#     )
#     total = total[total["L_QUANTITY"] < total["avg"]]
#     total = ps.DataFrame({"avg_yearly": [total["L_EXTENDEDPRICE"].sum() / 7.0]})
#     return total


# @time_collector
# def q18(lineitem, orders, customer):
#     gb1 = lineitem.groupby("L_ORDERKEY", as_index=False)["L_QUANTITY"].sum()
#     fgb1 = gb1[gb1.L_QUANTITY > 300]
#     jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
#     jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
#     gb2 = jn2.groupby(
#         ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
#         as_index=False,
#     )["L_QUANTITY"].sum()
#     total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
#     return total.head(100)


# @time_collector
# def q19(lineitem, part):
#     Brand31 = "Brand#31"
#     Brand43 = "Brand#43"
#     SMBOX = "SM BOX"
#     SMCASE = "SM CASE"
#     SMPACK = "SM PACK"
#     SMPKG = "SM PKG"
#     MEDBAG = "MED BAG"
#     MEDBOX = "MED BOX"
#     MEDPACK = "MED PACK"
#     MEDPKG = "MED PKG"
#     LGBOX = "LG BOX"
#     LGCASE = "LG CASE"
#     LGPACK = "LG PACK"
#     LGPKG = "LG PKG"
#     DELIVERINPERSON = "DELIVER IN PERSON"
#     AIR = "AIR"
#     AIRREG = "AIRREG"
#     lsel = (
#         (
#             ((lineitem.L_QUANTITY <= 36) & (lineitem.L_QUANTITY >= 26))
#             | ((lineitem.L_QUANTITY <= 25) & (lineitem.L_QUANTITY >= 15))
#             | ((lineitem.L_QUANTITY <= 14) & (lineitem.L_QUANTITY >= 4))
#         )
#         & (lineitem.L_SHIPINSTRUCT == DELIVERINPERSON)
#         & ((lineitem.L_SHIPMODE == AIR) | (lineitem.L_SHIPMODE == AIRREG))
#     )
#     psel = (part.P_SIZE >= 1) & (
#         (
#             (part.P_SIZE <= 5)
#             & (part.P_BRAND == Brand31)
#             & (
#                 (part.P_CONTAINER == SMBOX)
#                 | (part.P_CONTAINER == SMCASE)
#                 | (part.P_CONTAINER == SMPACK)
#                 | (part.P_CONTAINER == SMPKG)
#             )
#         )
#         | (
#             (part.P_SIZE <= 10)
#             & (part.P_BRAND == Brand43)
#             & (
#                 (part.P_CONTAINER == MEDBAG)
#                 | (part.P_CONTAINER == MEDBOX)
#                 | (part.P_CONTAINER == MEDPACK)
#                 | (part.P_CONTAINER == MEDPKG)
#             )
#         )
#         | (
#             (part.P_SIZE <= 15)
#             & (part.P_BRAND == Brand43)
#             & (
#                 (part.P_CONTAINER == LGBOX)
#                 | (part.P_CONTAINER == LGCASE)
#                 | (part.P_CONTAINER == LGPACK)
#                 | (part.P_CONTAINER == LGPKG)
#             )
#         )
#     )
#     flineitem = lineitem[lsel]
#     fpart = part[psel]
#     jn = flineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
#     jnsel = (
#         (jn.P_BRAND == Brand31)
#         & (
#             (jn.P_CONTAINER == SMBOX)
#             | (jn.P_CONTAINER == SMCASE)
#             | (jn.P_CONTAINER == SMPACK)
#             | (jn.P_CONTAINER == SMPKG)
#         )
#         & (jn.L_QUANTITY >= 4)
#         & (jn.L_QUANTITY <= 14)
#         & (jn.P_SIZE <= 5)
#         | (jn.P_BRAND == Brand43)
#         & (
#             (jn.P_CONTAINER == MEDBAG)
#             | (jn.P_CONTAINER == MEDBOX)
#             | (jn.P_CONTAINER == MEDPACK)
#             | (jn.P_CONTAINER == MEDPKG)
#         )
#         & (jn.L_QUANTITY >= 15)
#         & (jn.L_QUANTITY <= 25)
#         & (jn.P_SIZE <= 10)
#         | (jn.P_BRAND == Brand43)
#         & (
#             (jn.P_CONTAINER == LGBOX)
#             | (jn.P_CONTAINER == LGCASE)
#             | (jn.P_CONTAINER == LGPACK)
#             | (jn.P_CONTAINER == LGPKG)
#         )
#         & (jn.L_QUANTITY >= 26)
#         & (jn.L_QUANTITY <= 36)
#         & (jn.P_SIZE <= 15)
#     )
#     jn = jn[jnsel]
#     total = (jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)).sum()
#     return total


# @time_collector
# def q20(lineitem, part, nation, partsupp, supplier):
#     date1 = pd.Timestamp("1996-01-01")
#     date2 = pd.Timestamp("1997-01-01")
#     psel = part.P_NAME.str.startswith("azure")
#     nsel = nation.N_NAME == "JORDAN"
#     lsel = (lineitem.L_SHIPDATE >= date1) & (lineitem.L_SHIPDATE < date2)
#     fpart = part[psel]
#     fnation = nation[nsel]
#     flineitem = lineitem[lsel]
#     jn1 = fpart.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
#     jn2 = jn1.merge(
#         flineitem,
#         left_on=["PS_PARTKEY", "PS_SUPPKEY"],
#         right_on=["L_PARTKEY", "L_SUPPKEY"],
#     )
#     gb = jn2.groupby(["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"], as_index=False)[
#         "L_QUANTITY"
#     ].sum()
#     gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
#     fgb = gb[gbsel]
#     jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
#     jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
#     jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
#     total = jn4.sort_values("S_NAME").drop_duplicates()
#     return total


# @time_collector
# def q21(lineitem, orders, supplier, nation):
#     lineitem_filtered = lineitem.loc[
#         :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
#     ]

#     # Keep all rows that have another row in linetiem with the same orderkey and different suppkey
#     lineitem_orderkeys = (
#         lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
#         .groupby("L_ORDERKEY", as_index=False)["L_SUPPKEY"]
#         .nunique()
#     )
#     lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
#     lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] > 1]
#     lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

#     # Keep all rows that have l_receiptdate > l_commitdate
#     lineitem_filtered = lineitem_filtered[
#         lineitem_filtered["L_RECEIPTDATE"] > lineitem_filtered["L_COMMITDATE"]
#     ]
#     lineitem_filtered = lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]

#     # Merge Filter + Exists
#     lineitem_filtered = lineitem_filtered.merge(lineitem_orderkeys, on="L_ORDERKEY", how="inner")

#     # Not Exists: Check the exists condition isn't still satisfied on the output.
#     lineitem_orderkeys = lineitem_filtered.groupby("L_ORDERKEY", as_index=False)[
#         "L_SUPPKEY"
#     ].nunique()
#     lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
#     lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] == 1]
#     lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

#     # Merge Filter + Not Exists
#     lineitem_filtered = lineitem_filtered.merge(lineitem_orderkeys, on="L_ORDERKEY", how="inner")

#     orders_filtered = orders.loc[:, ["O_ORDERSTATUS", "O_ORDERKEY"]]
#     orders_filtered = orders_filtered[orders_filtered["O_ORDERSTATUS"] == "F"]
#     orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY"]]
#     total = lineitem_filtered.merge(
#         orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
#     )
#     total = total.loc[:, ["L_SUPPKEY"]]

#     supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY", "S_NAME"]]
#     total = total.merge(supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner")
#     total = total.loc[:, ["S_NATIONKEY", "S_NAME"]]
#     nation_filtered = nation.loc[:, ["N_NAME", "N_NATIONKEY"]]
#     nation_filtered = nation_filtered[nation_filtered["N_NAME"] == "SAUDI ARABIA"]
#     total = total.merge(nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner")
#     total = total.loc[:, ["S_NAME"]]
#     total = total.groupby("S_NAME", as_index=False).size().reset_index()
#     total.columns = ["S_NAME", "NUMWAIT"]
#     total = total.sort_values(by=["NUMWAIT", "S_NAME"], ascending=[False, True])
#     return total


# @time_collector
# def q22(customer, orders):
#     customer_filtered = customer.loc[:, ["C_ACCTBAL", "C_CUSTKEY"]]
#     customer_filtered["CNTRYCODE"] = customer["C_PHONE"].str.slice(0, 2)
#     customer_filtered = customer_filtered[
#         (customer["C_ACCTBAL"] > 0.00)
#         & customer_filtered["CNTRYCODE"].isin(["13", "31", "23", "29", "30", "18", "17"])
#     ]
#     avg_value = customer_filtered["C_ACCTBAL"].mean()
#     customer_filtered = customer_filtered[customer_filtered["C_ACCTBAL"] > avg_value]
#     # Select only the keys that don't match by performing a left join and only selecting columns with an na value
#     orders_filtered = orders.loc[:, ["O_CUSTKEY"]].drop_duplicates()
#     customer_keys = customer_filtered.loc[:, ["C_CUSTKEY"]].drop_duplicates()
#     customer_selected = customer_keys.merge(
#         orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
#     )
#     customer_selected = customer_selected[customer_selected["O_CUSTKEY"].isna()]
#     customer_selected = customer_selected.loc[:, ["C_CUSTKEY"]]
#     customer_selected = customer_selected.merge(customer_filtered, on="C_CUSTKEY", how="inner")
#     customer_selected = customer_selected.loc[:, ["CNTRYCODE", "C_ACCTBAL"]]
#     agg1 = customer_selected.groupby(["CNTRYCODE"], as_index=False).size().reset_index()
#     agg1.columns = ["CNTRYCODE", "NUMCUST"]
#     agg2 = customer_selected.groupby(["CNTRYCODE"], as_index=False).sum()
#     agg2.columns = ["CNTRYCODE", "TOTACCTBAL"]
#     total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
#     total = total.sort_values(by=["CNTRYCODE"], ascending=[True])
#     return total


query_to_loaders = {
    1: [load_lineitem],
    # 2: [load_part, load_partsupp, load_supplier, load_nation, load_region],
    # 3: [load_lineitem, load_orders, load_customer],
    4: [load_lineitem, load_orders],
    # 5: [
    #     load_lineitem,
    #     load_orders,
    #     load_customer,
    #     load_nation,
    #     load_region,
    #     load_supplier,
    # ],
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
    # 2: q02,
    # 3: q03,
    4: q04,
    # 5: q05,
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
    queries,
    log_time=True,
    include_io=False,
    test_result=False,
    print_result=False,
):
    print("Start data loading")
    total_start = time.time()
    for query in queries:
        loaders = query_to_loaders[query]
        for loader in loaders:
            loader(path, include_io=False)
    print(f"Data loading time (s): {time.time() - total_start}")
    total_start = time.time()
    for query in queries:
        try:
            t1 = time.time()
            result = query_to_runner[query](
                path,
                include_io,
            )
            if print_result:
                print(result)
            dur = time.time() - t1
            success = True
            if test_result:
                test_results(query, result)
        except Exception as e:
            print("".join(traceback.TracebackException.from_exception(e).format()))
            dur = 0.0
            success = False
        finally:
            if log_time:
                append_row("pyspark", query, dur, pyspark.__version__, success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    global spark
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
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
    parser.add_argument("--master", type=str, help="Spark master URI")

    # aws settings
    parser.add_argument("--account", type=str, help="AWS access id")
    parser.add_argument("--key", type=str, help="AWS secret access key")
    parser.add_argument("--endpoint", type=str, help="AWS region endpoint related to your S3")

    # parser.add_argument("--executor_cores", type=str, help='Number of cores for each Spark executor')
    # parser.add_argument("--executor_memory", type=str, help='Memory size for each Spark executor')
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
    print(f"Path: {args.path}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    print(f"Include IO: {args.include_io}")
    path: str = args.path
    master_ip = args.master

    account = args.account
    key = args.key

    spark = SparkSession\
        .builder\
        .appName("PySpark tpch query")\
        .master(args.master) \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # spark = (
    #     SparkSession.builder.appName("PySpark tpch query")
    #     .config(
    #         "spark.jars.packages",
    #         "org.apache.spark:spark-hadoop-cloud_2.12:3.2.2",
    #     )
    #     .config("spark.executor.cores", args.executor_cores)
    #     .config("spark.executor.memory", args.executor_memory)
    #     .getOrCreate()
    # )

    if "s3://" in path:
        conf = spark.sparkContext._jsc.hadoopConfiguration()
        conf.set("fs.s3a.access.key", account)
        conf.set("fs.s3a.secret.key", key)
        conf.set("fs.s3a.endpoint", args.endpoint)

        path = path.replace("s3://", "s3a://")
    
    run_queries(path, 
                args.queries, 
                args.log_time, 
                args.include_io, 
                args.test_result, 
                args.print_result)


if __name__ == "__main__":
    main()
