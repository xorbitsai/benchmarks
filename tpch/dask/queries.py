import os
import sys
import argparse
import json
import time
import traceback
from datetime import datetime
from typing import Dict

import pandas
from pandas.core.frame import DataFrame as PandasDF

import dask
import dask.dataframe as pd
from dask.distributed import Client, wait

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from common_utils import append_row, ANSWERS_BASE_DIR

dataset_dict = {}


def load_lineitem(root: str, storage_options: Dict, include_io: bool = False):
    if "lineitem" not in dataset_dict or include_io:
        data_path = root + "/lineitem"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        df.L_SHIPDATE = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
        df.L_RECEIPTDATE = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
        df.L_COMMITDATE = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
        dataset_dict["lineitem"] = df
        result = df
    else:
        result = dataset_dict["lineitem"]
    return result


def load_part(root: str, storage_options: Dict, include_io: bool = False):
    if "part" not in dataset_dict or include_io:
        data_path = root + "/part"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["part"] = df
        result = df
    else:
        result = dataset_dict["part"]
    return result


def load_orders(root: str, storage_options: Dict, include_io: bool = False):
    if "orders" not in dataset_dict or include_io:
        data_path = root + "/orders"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        df.O_ORDERDATE = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
        dataset_dict["orders"] = df
        result = df
    else:
        result = dataset_dict["orders"]
    return result


def load_customer(root: str, storage_options: Dict, include_io: bool = False):
    if "customer" not in dataset_dict or include_io:
        data_path = root + "/customer"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["customer"] = df
        result = df
    else:
        result = dataset_dict["customer"]
    return result


def load_nation(root: str, storage_options: Dict, include_io: bool = False):
    if "nation" not in dataset_dict or include_io:
        data_path = root + "/nation"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["nation"] = df
        result = df
    else:
        result = dataset_dict["nation"]
    return result


def load_region(root: str, storage_options: Dict, include_io: bool = False):
    if "region" not in dataset_dict or include_io:
        data_path = root + "/region"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["region"] = df
        result = df
    else:
        result = dataset_dict["region"]
    return result


def load_supplier(root: str, storage_options: Dict, include_io: bool = False):
    if "supplier" not in dataset_dict or include_io:
        data_path = root + "/supplier"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["supplier"] = df
        result = df
    else:
        result = dataset_dict["supplier"]
    return result


def load_partsupp(root: str, storage_options: Dict, include_io: bool = False):
    if "partsupp" not in dataset_dict or include_io:
        data_path = root + "/partsupp"
        df = pd.read_parquet(
            data_path,
            storage_options=storage_options,
        )
        dataset_dict["partsupp"] = df
        result = df
    else:
        result = dataset_dict["partsupp"]
    return result


def q01(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)

    date = datetime.strptime("1998-09-02", "%Y-%m-%d")
    lineitem_filtered = lineitem.loc[
        :,
        [
            "L_QUANTITY",
            "L_EXTENDEDPRICE",
            "L_DISCOUNT",
            "L_TAX",
            "L_RETURNFLAG",
            "L_LINESTATUS",
            "L_SHIPDATE",
            "L_ORDERKEY",
        ],
    ]
    sel = lineitem_filtered.L_SHIPDATE <= date
    lineitem_filtered = lineitem_filtered[sel].copy()
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
    gb = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"])

    total = gb.agg(
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

    total = total.compute().reset_index().sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
    return total


def q02(root: str, storage_options: Dict, include_io: bool = False):
    part = load_part(root, storage_options, include_io)
    partsupp = load_partsupp(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    region = load_region(root, storage_options, include_io)

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
        (part_filtered["P_SIZE"] == 15)
        & (part_filtered["P_TYPE"].str.endswith("BRASS"))
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

    min_values = merged_df.groupby("P_PARTKEY")["PS_SUPPLYCOST"].min().reset_index()

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

    total = total.compute().sort_values(
        by=[
            "S_ACCTBAL",
            "N_NAME",
            "S_NAME",
            "P_PARTKEY",
        ],
        ascending=[
            False,
            True,
            True,
            True,
        ],
    )

    total = total.head(100)

    return total


def q03(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)

    date = datetime.strptime("1995-03-04", "%Y-%m-%d")
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    orders_filtered = orders.loc[
        :, ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY"]
    ]
    customer_filtered = customer.loc[:, ["C_MKTSEGMENT", "C_CUSTKEY"]]
    lsel = lineitem_filtered.L_SHIPDATE > date
    osel = orders_filtered.O_ORDERDATE < date
    csel = customer_filtered.C_MKTSEGMENT == "HOUSEHOLD"
    flineitem = lineitem_filtered[lsel]
    forders = orders_filtered[osel]
    fcustomer = customer_filtered[csel]
    jn1 = fcustomer.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn2["TMP"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
    total = (
        jn2.groupby(["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"])["TMP"]
        .sum()
        .compute()
        .reset_index()
        .sort_values(["TMP"], ascending=False)
    )

    res = total.loc[:, ["L_ORDERKEY", "TMP", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    res = res.head(10)
    return res


def q04(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)

    date1 = datetime.strptime("1993-11-01", "%Y-%m-%d")
    date2 = datetime.strptime("1993-08-01", "%Y-%m-%d")
    lsel = lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE
    osel = (orders.O_ORDERDATE < date1) & (orders.O_ORDERDATE >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    forders = forders[["O_ORDERKEY", "O_ORDERPRIORITY"]]
    jn = forders.merge(
        flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY"
    ).drop_duplicates(subset=["O_ORDERKEY"])[["O_ORDERPRIORITY", "O_ORDERKEY"]]
    total = (
        jn.groupby("O_ORDERPRIORITY")["O_ORDERKEY"]
        .count()
        .reset_index()
        .sort_values(["O_ORDERPRIORITY"])
    )
    total = total.compute()
    return total


def q05(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    region = load_region(root, storage_options, include_io)

    date1 = datetime.strptime("1996-01-01", "%Y-%m-%d")
    date2 = datetime.strptime("1997-01-01", "%Y-%m-%d")

    rsel = region.R_NAME == "ASIA"
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)

    forders = orders[osel]
    fregion = region[rsel]
    jn1 = fregion.merge(nation, left_on="R_REGIONKEY", right_on="N_REGIONKEY")
    jn2 = jn1.merge(customer, left_on="N_NATIONKEY", right_on="C_NATIONKEY")
    jn3 = jn2.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn4 = jn3.merge(lineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")

    jn5 = supplier.merge(
        jn4, left_on=["S_SUPPKEY", "S_NATIONKEY"], right_on=["L_SUPPKEY", "N_NATIONKEY"]
    )
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME")["TMP"].sum()

    total = gb.compute().reset_index().sort_values("TMP", ascending=False)
    return total


def q06(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)

    date1 = datetime.strptime("1996-01-01", "%Y-%m-%d")
    date2 = datetime.strptime("1997-01-01", "%Y-%m-%d")

    lineitem_filtered = lineitem.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    sel = (
        (lineitem_filtered.L_SHIPDATE >= date1)
        & (lineitem_filtered.L_SHIPDATE < date2)
        & (lineitem_filtered.L_DISCOUNT >= 0.08)
        & (lineitem_filtered.L_DISCOUNT <= 0.1)
        & (lineitem_filtered.L_QUANTITY < 24)
    )
    flineitem = lineitem_filtered[sel]
    total = (flineitem.L_EXTENDEDPRICE * flineitem.L_DISCOUNT).sum()
    total = total.compute()
    return total


def q07(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)

    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= datetime.strptime("1995-01-01", "%Y-%m-%d"))
        & (lineitem["L_SHIPDATE"] < datetime.strptime("1997-01-01", "%Y-%m-%d"))
    ]
    lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].apply(
        lambda x: x.year
    )
    lineitem_filtered["VOLUME"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_YEAR", "VOLUME"]
    ]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    n1 = nation[(nation["N_NAME"] == "FRANCE")].loc[:, ["N_NATIONKEY", "N_NAME"]]
    n2 = nation[(nation["N_NAME"] == "GERMANY")].loc[:, ["N_NATIONKEY", "N_NAME"]]

    # ----- do nation 1 -----
    N1_C = customer_filtered.merge(
        n1, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_C = N1_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N1_C_O = N1_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N1_C_O = N1_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N2_S = supplier_filtered.merge(
        n2, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_S = N2_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N2_S_L = N2_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N2_S_L = N2_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total1 = N1_C_O.merge(
        N2_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total1 = total1.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # ----- do nation 2 ----- (same as nation 1 section but with nation 2)
    N2_C = customer_filtered.merge(
        n2, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N2_C = N2_C.drop(columns=["C_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "CUST_NATION"}
    )
    N2_C_O = N2_C.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="inner"
    )
    N2_C_O = N2_C_O.drop(columns=["C_CUSTKEY", "O_CUSTKEY"])

    N1_S = supplier_filtered.merge(
        n1, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    N1_S = N1_S.drop(columns=["S_NATIONKEY", "N_NATIONKEY"]).rename(
        columns={"N_NAME": "SUPP_NATION"}
    )
    N1_S_L = N1_S.merge(
        lineitem_filtered, left_on="S_SUPPKEY", right_on="L_SUPPKEY", how="inner"
    )
    N1_S_L = N1_S_L.drop(columns=["S_SUPPKEY", "L_SUPPKEY"])

    total2 = N2_C_O.merge(
        N1_S_L, left_on="O_ORDERKEY", right_on="L_ORDERKEY", how="inner"
    )
    total2 = total2.drop(columns=["O_ORDERKEY", "L_ORDERKEY"])

    # concat results
    total = pd.concat([total1, total2])
    total = total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"]).VOLUME.agg("sum")
    total.columns = ["SUPP_NATION", "CUST_NATION", "L_YEAR", "REVENUE"]

    total = (
        total.compute()
        .reset_index()
        .sort_values(
            by=["SUPP_NATION", "CUST_NATION", "L_YEAR"],
            ascending=[
                True,
                True,
                True,
            ],
        )
    )
    return total


def q08(root: str, storage_options: Dict, include_io: bool = False):
    part = load_part(root, storage_options, include_io)
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    region = load_region(root, storage_options, include_io)

    part_filtered = part[(part["P_TYPE"] == "ECONOMY ANODIZED STEEL")]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY"]]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_SUPPKEY", "L_ORDERKEY"]]
    lineitem_filtered["VOLUME"] = lineitem["L_EXTENDEDPRICE"] * (
        1.0 - lineitem["L_DISCOUNT"]
    )
    total = part_filtered.merge(
        lineitem_filtered, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY", "L_ORDERKEY", "VOLUME"]]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["L_ORDERKEY", "VOLUME", "S_NATIONKEY"]]
    orders_filtered = orders[
        (orders["O_ORDERDATE"] >= datetime.strptime("1995-01-01", "%Y-%m-%d"))
        & (orders["O_ORDERDATE"] < datetime.strptime("1997-01-01", "%Y-%m-%d"))
    ]
    orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].apply(lambda x: x.year)
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_YEAR"]]
    total = total.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_CUSTKEY", "O_YEAR"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    total = total.merge(
        customer_filtered, left_on="O_CUSTKEY", right_on="C_CUSTKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "C_NATIONKEY"]]
    n1_filtered = nation.loc[:, ["N_NATIONKEY", "N_REGIONKEY"]]
    n2_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME"]].rename(
        columns={"N_NAME": "NATION"}
    )
    total = total.merge(
        n1_filtered, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "N_REGIONKEY"]]
    total = total.merge(
        n2_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "N_REGIONKEY", "NATION"]]
    region_filtered = region[(region["R_NAME"] == "AMERICA")]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    total = total.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "NATION"]]

    def udf(df):
        demonimator = df["VOLUME"].sum()
        df = df[df["NATION"] == "BRAZIL"]
        numerator = df["VOLUME"].sum()
        return numerator / demonimator

    total = total.groupby("O_YEAR").apply(udf)
    total = (
        total.compute()
        .reset_index()
        .sort_values(
            by=[
                "O_YEAR",
            ],
            ascending=[
                True,
            ],
        )
    )
    total.columns = ["O_YEAR", "MKT_SHARE"]
    return total


def q09(root: str, storage_options: Dict, include_io: bool = False):
    part = load_part(root, storage_options, include_io)
    partsupp = load_partsupp(root, storage_options, include_io)
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)

    psel = part.P_NAME.str.contains("ghost")
    fpart = part[psel]
    jn1 = lineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn2 = jn1.merge(supplier, left_on="L_SUPPKEY", right_on="S_SUPPKEY")
    jn3 = jn2.merge(nation, left_on="S_NATIONKEY", right_on="N_NATIONKEY")
    jn4 = partsupp.merge(
        jn3, left_on=["PS_PARTKEY", "PS_SUPPKEY"], right_on=["L_PARTKEY", "L_SUPPKEY"]
    )
    jn5 = jn4.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn5["TMP"] = jn5.L_EXTENDEDPRICE * (1 - jn5.L_DISCOUNT) - (
        (1 * jn5.PS_SUPPLYCOST) * jn5.L_QUANTITY
    )
    jn5["O_YEAR"] = jn5.O_ORDERDATE.apply(lambda x: x.year)
    gb = jn5.groupby(["N_NAME", "O_YEAR"])["TMP"].sum()
    total = (
        gb.compute()
        .reset_index()
        .sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
    )
    return total


def q10(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)

    t1 = time.time()
    date1 = datetime.strptime("1994-11-01", "%Y-%m-%d")
    date2 = datetime.strptime("1995-02-01", "%Y-%m-%d")
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    lsel = lineitem.L_RETURNFLAG == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["TMP"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
    gb = jn3.groupby(
        [
            "C_CUSTKEY",
            "C_NAME",
            "C_ACCTBAL",
            "C_PHONE",
            "N_NAME",
            "C_ADDRESS",
            "C_COMMENT",
        ],
    )["TMP"].sum()
    total = gb.compute().reset_index().sort_values("TMP", ascending=False)
    total = total.head(20)

    return total


def q11(root: str, storage_options: Dict, include_io: bool = False):
    partsupp = load_partsupp(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)

    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
    )
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
    nation_filtered = nation[(nation["N_NAME"] == "GERMANY")]
    nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * 0.0001

    total = ps_supp_n_merge.groupby(["PS_PARTKEY"]).TOTAL_COST.agg("sum").reset_index()
    total = total.rename(columns={"TOTAL_COST": "VALUE"})
    total = total[total["VALUE"] > sum_val]
    total = total.compute().sort_values("VALUE", ascending=False)
    
    return total


def q12(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)

    date1 = datetime.strptime("1994-01-01", "%Y-%m-%d")
    date2 = datetime.strptime("1995-01-01", "%Y-%m-%d")
    sel = (
        (lineitem.L_RECEIPTDATE < date2)
        & (lineitem.L_COMMITDATE < date2)
        & (lineitem.L_SHIPDATE < date2)
        & (lineitem.L_SHIPDATE < lineitem.L_COMMITDATE)
        & (lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE)
        & (lineitem.L_RECEIPTDATE >= date1)
        & ((lineitem.L_SHIPMODE == "MAIL") | (lineitem.L_SHIPMODE == "SHIP"))
    )
    flineitem = lineitem[sel]
    jn = flineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    gb = jn.groupby("L_SHIPMODE")["O_ORDERPRIORITY"]

    def g1(x):
        return x.apply(lambda s: ((s == "1-URGENT") | (s == "2-HIGH")).sum())

    def g2(x):
        return x.apply(lambda s: ((s != "1-URGENT") & (s != "2-HIGH")).sum())

    g1_agg = pd.Aggregation("g1", g1, lambda s0: s0.sum())
    g2_agg = pd.Aggregation("g2", g2, lambda s0: s0.sum())
    total = gb.agg([g1_agg, g2_agg])
    total = total.compute().reset_index().sort_values("L_SHIPMODE")

    return total


def q13(root: str, storage_options: Dict, include_io: bool = False):
    customer = load_customer(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)

    customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
    orders_filtered = orders[
        ~orders["O_COMMENT"].str.contains("special(\\S|\\s)*requests")
    ]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]

    count_df = c_o_merged.groupby(["C_CUSTKEY"]).O_ORDERKEY.agg("count").reset_index()
    count_df = count_df.rename(columns={"O_ORDERKEY": "C_COUNT"})

    total = count_df.groupby(["C_COUNT"]).size().reset_index()
    total.columns = ["C_COUNT", "CUSTDIST"]
    total = total.compute().sort_values(
        by=["CUSTDIST", "C_COUNT"],
        ascending=[
            False,
            False,
        ],
    )

    return total


def q14(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    part = load_part(root, storage_options, include_io)

    startDate = datetime.strptime("1994-03-01", "%Y-%m-%d")
    endDate = datetime.strptime("1994-04-01", "%Y-%m-%d")
    p_type_like = "PROMO"
    part_filtered = part.loc[:, ["P_PARTKEY", "P_TYPE"]]
    lineitem_filtered = lineitem.loc[
        :, ["L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE", "L_PARTKEY"]
    ]
    sel = (lineitem_filtered.L_SHIPDATE >= startDate) & (
        lineitem_filtered.L_SHIPDATE < endDate
    )
    flineitem = lineitem_filtered[sel]
    jn = flineitem.merge(part_filtered, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn["TMP"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
    total = jn[jn.P_TYPE.str.startswith(p_type_like)].TMP.sum() * 100 / jn.TMP.sum()

    return total

def q15(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)

    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= datetime.strptime("1996-01-01", "%Y-%m-%d"))
        & (lineitem["L_SHIPDATE"] < (datetime.strptime("1996-04-01", "%Y-%m-%d")))
    ]  # + pd.DateOffset(months=3)))]
    lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[:, ["L_SUPPKEY", "REVENUE_PARTS"]]
    revenue_table = (
        lineitem_filtered.groupby("L_SUPPKEY")["REVENUE_PARTS"].agg("sum").reset_index()
    )
    revenue_table = revenue_table.rename(
        columns={"REVENUE_PARTS": "TOTAL_REVENUE", "L_SUPPKEY": "SUPPLIER_NO"}
    )
    max_revenue = revenue_table["TOTAL_REVENUE"].max()
    revenue_table = revenue_table[revenue_table["TOTAL_REVENUE"] == max_revenue]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE"]]
    total = supplier_filtered.merge(
        revenue_table, left_on="S_SUPPKEY", right_on="SUPPLIER_NO", how="inner"
    )
    total = total.loc[
        :, ["S_SUPPKEY", "S_NAME", "S_ADDRESS", "S_PHONE", "TOTAL_REVENUE"]
    ]
    
    return total


def q16(root: str, storage_options: Dict, include_io: bool = False):
    part = load_part(root, storage_options, include_io)
    partsupp = load_partsupp(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)

    part_filtered = part[
        (part["P_BRAND"] != "Brand#45")
        & (~part["P_TYPE"].str.contains("^MEDIUM POLISHED"))
        & part["P_SIZE"].isin([49, 14, 23, 45, 19, 3, 36, 9])
    ]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY", "P_BRAND", "P_TYPE", "P_SIZE"]]
    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    total = part_filtered.merge(
        partsupp_filtered, left_on="P_PARTKEY", right_on="PS_PARTKEY", how="inner"
    )
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    supplier_filtered = supplier[
        supplier["S_COMMENT"].str.contains("Customer(\\S|\\s)*Complaints")
    ]
    supplier_filtered = supplier_filtered.loc[:, ["S_SUPPKEY"]].drop_duplicates()
    # left merge to select only ps_suppkey values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left"
    )
    total = total[total["S_SUPPKEY"].isna()]
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    total = (
        total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"])["PS_SUPPKEY"]
        .nunique()
        .reset_index()
    )
    total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
    total = total.compute().sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )

    return total


def q17(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    part = load_part(root, storage_options, include_io)

    left = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY", "L_EXTENDEDPRICE"]]
    right = part[((part["P_BRAND"] == "Brand#23") & (part["P_CONTAINER"] == "MED BOX"))]
    right = right.loc[:, ["P_PARTKEY"]]
    line_part_merge = left.merge(
        right, left_on="L_PARTKEY", right_on="P_PARTKEY", how="inner"
    )
    line_part_merge = line_part_merge.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "P_PARTKEY"]
    ]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY"]]
    lineitem_avg = (
        lineitem_filtered.groupby(["L_PARTKEY"])
        .L_QUANTITY.agg("mean")
        .reset_index()
        .rename(columns={"L_QUANTITY": "avg"})
    )
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = pandas.DataFrame(
        {"avg_yearly": [(total["L_EXTENDEDPRICE"].sum() / 7.0).compute()]}
    )

    return total


def q18(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    customer = load_customer(root, storage_options, include_io)

    gb1 = lineitem.groupby("L_ORDERKEY")["L_QUANTITY"].sum().reset_index()
    fgb1 = gb1[gb1.L_QUANTITY > 300]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
    )["L_QUANTITY"].sum()
    total = (
        gb2.compute()
        .reset_index()
        .sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    )
    total = total.head(100)

    return total


def q19(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    part = load_part(root, storage_options, include_io)

    t1 = time.time()
    Brand31 = "Brand#31"
    Brand43 = "Brand#43"
    SMBOX = "SM BOX"
    SMCASE = "SM CASE"
    SMPACK = "SM PACK"
    SMPKG = "SM PKG"
    MEDBAG = "MED BAG"
    MEDBOX = "MED BOX"
    MEDPACK = "MED PACK"
    MEDPKG = "MED PKG"
    LGBOX = "LG BOX"
    LGCASE = "LG CASE"
    LGPACK = "LG PACK"
    LGPKG = "LG PKG"
    DELIVERINPERSON = "DELIVER IN PERSON"
    AIR = "AIR"
    AIRREG = "AIRREG"
    lsel = (
        (
            ((lineitem.L_QUANTITY <= 36) & (lineitem.L_QUANTITY >= 26))
            | ((lineitem.L_QUANTITY <= 25) & (lineitem.L_QUANTITY >= 15))
            | ((lineitem.L_QUANTITY <= 14) & (lineitem.L_QUANTITY >= 4))
        )
        & (lineitem.L_SHIPINSTRUCT == DELIVERINPERSON)
        & ((lineitem.L_SHIPMODE == AIR) | (lineitem.L_SHIPMODE == AIRREG))
    )
    psel = (part.P_SIZE >= 1) & (
        (
            (part.P_SIZE <= 5)
            & (part.P_BRAND == Brand31)
            & (
                (part.P_CONTAINER == SMBOX)
                | (part.P_CONTAINER == SMCASE)
                | (part.P_CONTAINER == SMPACK)
                | (part.P_CONTAINER == SMPKG)
            )
        )
        | (
            (part.P_SIZE <= 10)
            & (part.P_BRAND == Brand43)
            & (
                (part.P_CONTAINER == MEDBAG)
                | (part.P_CONTAINER == MEDBOX)
                | (part.P_CONTAINER == MEDPACK)
                | (part.P_CONTAINER == MEDPKG)
            )
        )
        | (
            (part.P_SIZE <= 15)
            & (part.P_BRAND == Brand43)
            & (
                (part.P_CONTAINER == LGBOX)
                | (part.P_CONTAINER == LGCASE)
                | (part.P_CONTAINER == LGPACK)
                | (part.P_CONTAINER == LGPKG)
            )
        )
    )
    flineitem = lineitem[lsel]
    fpart = part[psel]
    jn = flineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jnsel = (
        (
            (jn.P_BRAND == Brand31)
            & (
                (jn.P_CONTAINER == SMBOX)
                | (jn.P_CONTAINER == SMCASE)
                | (jn.P_CONTAINER == SMPACK)
                | (jn.P_CONTAINER == SMPKG)
            )
            & (jn.L_QUANTITY >= 4)
            & (jn.L_QUANTITY <= 14)
            & (jn.P_SIZE <= 5)
        )
        | (
            (jn.P_BRAND == Brand43)
            & (
                (jn.P_CONTAINER == MEDBAG)
                | (jn.P_CONTAINER == MEDBOX)
                | (jn.P_CONTAINER == MEDPACK)
                | (jn.P_CONTAINER == MEDPKG)
            )
            & (jn.L_QUANTITY >= 15)
            & (jn.L_QUANTITY <= 25)
            & (jn.P_SIZE <= 10)
        )
        | (
            (jn.P_BRAND == Brand43)
            & (
                (jn.P_CONTAINER == LGBOX)
                | (jn.P_CONTAINER == LGCASE)
                | (jn.P_CONTAINER == LGPACK)
                | (jn.P_CONTAINER == LGPKG)
            )
            & (jn.L_QUANTITY >= 26)
            & (jn.L_QUANTITY <= 36)
            & (jn.P_SIZE <= 15)
        )
    )
    jn = jn[jnsel]
    total = (jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)).sum()

    return total


def q20(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    part = load_part(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)
    partsupp = load_partsupp(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)

    date1 = datetime.strptime("1996-01-01", "%Y-%m-%d")
    date2 = datetime.strptime("1997-01-01", "%Y-%m-%d")
    psel = part.P_NAME.str.startswith("azure")
    nsel = nation.N_NAME == "JORDAN"
    lsel = (lineitem.L_SHIPDATE >= date1) & (lineitem.L_SHIPDATE < date2)
    fpart = part[psel]
    fnation = nation[nsel]
    flineitem = lineitem[lsel]
    jn1 = fpart.merge(partsupp, left_on="P_PARTKEY", right_on="PS_PARTKEY")
    jn2 = jn1.merge(
        flineitem,
        left_on=["PS_PARTKEY", "PS_SUPPKEY"],
        right_on=["L_PARTKEY", "L_SUPPKEY"],
    )
    gb = (
        jn2.groupby(["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"])["L_QUANTITY"]
        .sum()
        .reset_index()
    )
    gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
    jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
    total = jn4.compute().sort_values("S_NAME").drop_duplicates()

    return total


def q21(root: str, storage_options: Dict, include_io: bool = False):
    lineitem = load_lineitem(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)
    supplier = load_supplier(root, storage_options, include_io)
    nation = load_nation(root, storage_options, include_io)

    t1 = time.time()
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
    ]

    # Exists
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
        .groupby("L_ORDERKEY")["L_SUPPKEY"]
        .nunique()
        .reset_index()
    )
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] > 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Filter
    lineitem_filtered = lineitem_filtered[
        lineitem_filtered["L_RECEIPTDATE"] > lineitem_filtered["L_COMMITDATE"]
    ]
    lineitem_filtered = lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]

    # Merge Filter + Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    # Not Exists: Check the exists condition isn't still satisfied on the output.
    lineitem_orderkeys = (
        lineitem_filtered.groupby("L_ORDERKEY")["L_SUPPKEY"].nunique().reset_index()
    )
    lineitem_orderkeys.columns = ["L_ORDERKEY", "nunique_col"]
    lineitem_orderkeys = lineitem_orderkeys[lineitem_orderkeys["nunique_col"] == 1]
    lineitem_orderkeys = lineitem_orderkeys.loc[:, ["L_ORDERKEY"]]

    # Merge Filter + Not Exists
    lineitem_filtered = lineitem_filtered.merge(
        lineitem_orderkeys, on="L_ORDERKEY", how="inner"
    )

    orders_filtered = orders.loc[:, ["O_ORDERSTATUS", "O_ORDERKEY"]]
    orders_filtered = orders_filtered[orders_filtered["O_ORDERSTATUS"] == "F"]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY"]]
    total = lineitem_filtered.merge(
        orders_filtered, left_on="L_ORDERKEY", right_on="O_ORDERKEY", how="inner"
    )
    total = total.loc[:, ["L_SUPPKEY"]]

    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY", "S_NAME"]]
    total = total.merge(
        supplier_filtered, left_on="L_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    total = total.loc[:, ["S_NATIONKEY", "S_NAME"]]
    nation_filtered = nation.loc[:, ["N_NAME", "N_NATIONKEY"]]
    nation_filtered = nation_filtered[nation_filtered["N_NAME"] == "SAUDI ARABIA"]
    total = total.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["S_NAME"]]
    total = total.groupby("S_NAME").size().reset_index()
    total.columns = ["S_NAME", "NUMWAIT"]
    total = total.compute().sort_values(
        by=[
            "NUMWAIT",
            "S_NAME",
        ],
        ascending=[
            False,
            True,
        ],
    )

    return total


def q22(root: str, storage_options: Dict, include_io: bool = False):
    customer = load_customer(root, storage_options, include_io)
    orders = load_orders(root, storage_options, include_io)

    customer_filtered = customer.loc[:, ["C_ACCTBAL", "C_CUSTKEY"]]
    customer_filtered["CNTRYCODE"] = customer["C_PHONE"].str.slice(0, 2)
    customer_filtered = customer_filtered[
        (customer["C_ACCTBAL"] > 0.00)
        & customer_filtered["CNTRYCODE"].isin(
            ["13", "31", "23", "29", "30", "18", "17"]
        )
    ]
    avg_value = customer_filtered["C_ACCTBAL"].mean()
    customer_filtered = customer_filtered[customer_filtered["C_ACCTBAL"] > avg_value]
    # Select only the keys that don't match by performing a left join and only
    # selecting columns with an N/A value
    orders_filtered = orders.loc[:, ["O_CUSTKEY"]].drop_duplicates()
    customer_keys = customer_filtered.loc[:, ["C_CUSTKEY"]].drop_duplicates()
    customer_selected = customer_keys.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    customer_selected = customer_selected[customer_selected["O_CUSTKEY"].isna()]
    customer_selected = customer_selected.loc[:, ["C_CUSTKEY"]]
    customer_selected = customer_selected.merge(
        customer_filtered, on="C_CUSTKEY", how="inner"
    )
    customer_selected = customer_selected.loc[:, ["CNTRYCODE", "C_ACCTBAL"]]
    agg1 = customer_selected.groupby(["CNTRYCODE"]).size().reset_index()
    agg1.columns = ["CNTRYCODE", "NUMCUST"]
    agg2 = customer_selected.groupby(["CNTRYCODE"]).C_ACCTBAL.agg("sum").reset_index()
    agg2 = agg2.rename(columns={"C_ACCTBAL": "TOTACCTBAL"})
    total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
    total = total.compute().sort_values(
        by=[
            "CNTRYCODE",
        ],
        ascending=[
            True,
        ],
    )
    
    return total


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
    11: [load_partsupp, load_supplier, load_nation],
    12: [load_lineitem, load_orders],
    13: [load_customer, load_orders],
    14: [load_lineitem, load_part],
    15: [load_lineitem, load_supplier],
    16: [load_part, load_partsupp, load_supplier],
    17: [load_lineitem, load_part],
    18: [load_lineitem, load_orders, load_customer],
    19: [load_lineitem, load_part],
    20: [load_lineitem, load_part, load_nation, load_partsupp, load_supplier],
    21: [load_lineitem, load_orders, load_supplier, load_nation],
    22: [load_customer, load_orders],
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
    11: q11,
    12: q12,
    13: q13,
    14: q14,
    15: q15,
    16: q16,
    17: q17,
    18: q18,
    19: q19,
    20: q20,
    21: q21,
    22: q22,
}


def get_query_answer(query: int, base_dir: str = ANSWERS_BASE_DIR) -> PandasDF:
    answer_df = pandas.read_csv(
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

        pandas.testing.assert_series_equal(
            left=s1,
            right=s2,
            check_index=False,
            check_names=False,
            check_exact=False,
            rtol=1e-2,
        )


def run_queries(path, 
        storage_options, 
        client, 
        queries,
        log_time=True,
        include_io=False,
        test_result=False,
        print_result=False
    ):
    total_start = time.time()
    print("Start data loading")
    for query in queries:
        loaders = query_to_loaders[query]
        for loader in loaders:
            loader(path, storage_options, include_io)
    print(f"Data loading time (s): {time.time() - total_start}")

    total_start = time.time()
    print("Start data persisting")
    for table_name in dataset_dict:
        df = dataset_dict[table_name]
        start = time.time()
        df = client.persist(df)
        wait(df)
        dataset_dict[table_name] = df
        print(f"{table_name} persisting time (s): {time.time() - start}")
    print(f"Total data persisting time (s): {time.time() - total_start}")

    total_start = time.time()
    for query in queries:
        try:
            t1 = time.time()
            result = query_to_runner[query](path, storage_options, include_io)
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
                append_row("dask", query, dur, dask.__version__, success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the TPC-H dataset."
    )
    parser.add_argument(
        "--storage_options",
        type=str,
        required=False,
        help="Path to the storage options json file.",
    )
    parser.add_argument(
        "--queries",
        type=int,
        nargs="+",
        required=False,
        help="Comma separated TPC-H queries to run.",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        required=False,
        help="The endpoint of existing Dask cluster."
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

    if args.endpoint == "local" or args.endpoint is None:
        from dask.distributed import LocalCluster
        client = LocalCluster()
    elif args.endpoint:
        client = Client(args.endpoint)
    
    run_queries(args.path, 
        storage_options, 
        client, queries, 
        args.log_time, 
        args.include_io, 
        args.test_result, 
        args.print_result
    )


if __name__ == "__main__":
    main()
