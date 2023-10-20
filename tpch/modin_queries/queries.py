import os
import json
import time
import argparse
import traceback
from typing import Dict

import modin
import ray
import modin.pandas as pd

from common_utils import log_time_fn, parse_common_arguments, print_result_fn

dataset_dict = {}


def load_lineitem(root: str, storage_options: Dict):
    if "lineitem" not in dataset_dict:
        data_path = root + "/lineitem"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        df.L_SHIPDATE = pd.to_datetime(df.L_SHIPDATE, format="%Y-%m-%d")
        df.L_RECEIPTDATE = pd.to_datetime(df.L_RECEIPTDATE, format="%Y-%m-%d")
        df.L_COMMITDATE = pd.to_datetime(df.L_COMMITDATE, format="%Y-%m-%d")
        result = df
        dataset_dict["lineitem"] = result
    else:
        result = dataset_dict["lineitem"]
    return result


def load_part(root: str, storage_options: Dict):
    if "part" not in dataset_dict:
        data_path = root + "/part"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["part"] = result
    else:
        result = dataset_dict["part"]
    return result


def load_orders(root: str, storage_options: Dict):
    if "orders" not in dataset_dict:
        data_path = root + "/orders"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        df.O_ORDERDATE = pd.to_datetime(df.O_ORDERDATE, format="%Y-%m-%d")
        result = df
        dataset_dict["orders"] = result
    else:
        result = dataset_dict["orders"]
    return result


def load_customer(root: str, storage_options: Dict):
    if "customer" not in dataset_dict:
        data_path = root + "/customer"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["customer"] = result
    else:
        result = dataset_dict["customer"]
    return result


def load_nation(root: str, storage_options: Dict):
    if "nation" not in dataset_dict:
        data_path = root + "/nation"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["nation"] = result
    else:
        result = dataset_dict["nation"]
    return result


def load_region(root: str, storage_options: Dict):
    if "region" not in dataset_dict:
        data_path = root + "/region"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["region"] = result
    else:
        result = dataset_dict["region"]
    return result


def load_supplier(root: str, storage_options: Dict):
    if "supplier" not in dataset_dict:
        data_path = root + "/supplier"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["supplier"] = result
    else:
        result = dataset_dict["supplier"]
    return result


def load_partsupp(root: str, storage_options: Dict):
    if "partsupp" not in dataset_dict:
        data_path = root + "/partsupp"
        df = pd.read_parquet(data_path, storage_options=storage_options)
        result = df
        dataset_dict["partsupp"] = result
    else:
        result = dataset_dict["partsupp"]
    return result


def q01(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)

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
    gb = lineitem_filtered.groupby(["L_RETURNFLAG", "L_LINESTATUS"], as_index=False)
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
    total = (
        total.sort_values(["L_RETURNFLAG", "L_LINESTATUS"])
            .rename(columns={
                "L_QUANTITY": "SUM_QTY",
                "L_EXTENDEDPRICE": "SUM_BASE_PRICE",
                "DISC_PRICE": "SUM_DISC_PRICE",
                "CHARGE": "SUM_CHARGE",
                "L_DISCOUNT": "AVG_DISC",
                "L_ORDERKEY": "COUNT_ORDER"
            })
    )

    return total


def q02(root: str, storage_options: Dict):
    part = load_part(root, storage_options)
    partsupp = load_partsupp(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)
    region = load_region(root, storage_options)

    size = 15
    p_type = "BRASS"
    region_name = "EUROPE"

    nation_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME", "N_REGIONKEY"]]
    region_filtered = region[(region["R_NAME"] == region_name)]
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
        (part_filtered["P_SIZE"] == size)
        & (part_filtered["P_TYPE"].str.endswith(p_type))
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
    total = total.head(100)

    return total


def q03(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    customer = load_customer(root, storage_options)

    mktsegment = "HOUSEHOLD"
    date = pd.Timestamp("1995-03-04")
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE"]
    ]
    orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY", "O_ORDERDATE", "O_SHIPPRIORITY"]]
    customer_filtered = customer.loc[:, ["C_MKTSEGMENT", "C_CUSTKEY"]]
    lsel = lineitem_filtered.L_SHIPDATE > date
    osel = orders_filtered.O_ORDERDATE < date
    csel = customer_filtered.C_MKTSEGMENT == mktsegment
    flineitem = lineitem_filtered[lsel]
    forders = orders_filtered[osel]
    fcustomer = customer_filtered[csel]
    jn1 = fcustomer.merge(forders, left_on="C_CUSTKEY", right_on="O_CUSTKEY")
    jn2 = jn1.merge(flineitem, left_on="O_ORDERKEY", right_on="L_ORDERKEY")
    jn2["REVENUE"] = jn2.L_EXTENDEDPRICE * (1 - jn2.L_DISCOUNT)
    total = (
        jn2.groupby(["L_ORDERKEY", "O_ORDERDATE", "O_SHIPPRIORITY"], as_index=False)["REVENUE"]
        .sum()
        .sort_values(["REVENUE"], ascending=False)
    )

    total = total[:10].loc[
        :, ["L_ORDERKEY", "REVENUE", "O_ORDERDATE", "O_SHIPPRIORITY"]
    ]

    return total


def q04(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)

    date2 = pd.Timestamp("1993-8-01")
    date1 = date2 + pd.DateOffset(months=3)
    lsel = lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE
    osel = (orders.O_ORDERDATE < date1) & (orders.O_ORDERDATE >= date2)
    flineitem = lineitem[lsel]
    forders = orders[osel]
    jn = forders[forders["O_ORDERKEY"].isin(flineitem["L_ORDERKEY"])]
    total = (
        jn.groupby("O_ORDERPRIORITY", as_index=False)["O_ORDERKEY"]
        .count()
        .sort_values(["O_ORDERPRIORITY"])
        .rename(columns={"O_ORDERKEY": "ORDER_COUNT"})
    )

    return total


def q05(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    customer = load_customer(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)
    region = load_region(root, storage_options)

    region_name = "ASIA"
    date1 = pd.Timestamp("1996-01-01")
    date2 = date1 + pd.DateOffset(years=1)
    rsel = region.R_NAME == region_name
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
    jn5["REVENUE"] = jn5.L_EXTENDEDPRICE * (1.0 - jn5.L_DISCOUNT)
    gb = jn5.groupby("N_NAME", as_index=False)["REVENUE"].sum()
    total = gb.sort_values("REVENUE", ascending=False)

    return total


def q06(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)

    date1 = pd.Timestamp("1996-01-01")
    date2 = date1 + pd.DateOffset(years=1)
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
    result_value = (flineitem.L_EXTENDEDPRICE * flineitem.L_DISCOUNT).sum()
    result_df = pd.DataFrame({"REVENUE": [result_value]})

    return result_df


def q07(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    customer = load_customer(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)

    nation1 = "FRANCE"
    nation2 = "GERMANY"
    lineitem_filtered = lineitem.loc[
        (lineitem["L_SHIPDATE"] >= pd.Timestamp("1995-01-01"))
        & (lineitem["L_SHIPDATE"] < pd.Timestamp("1997-01-01"))
    ]
    lineitem_filtered["L_YEAR"] = lineitem_filtered["L_SHIPDATE"].dt.year
    lineitem_filtered["VOLUME"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_YEAR", "VOLUME"]
    ]
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    orders_filtered = orders.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    customer_filtered = customer.loc[:, ["C_CUSTKEY", "C_NATIONKEY"]]
    n1 = nation[(nation["N_NAME"] == nation1)].loc[:, ["N_NATIONKEY", "N_NAME"]]
    n2 = nation[(nation["N_NAME"] == nation2)].loc[:, ["N_NATIONKEY", "N_NAME"]]

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

    total = (
        total.groupby(["SUPP_NATION", "CUST_NATION", "L_YEAR"], as_index=False)
            .agg(REVENUE=pd.NamedAgg(column="VOLUME", aggfunc="sum"))
            .sort_values(
                by=["SUPP_NATION", "CUST_NATION", "L_YEAR"],
                ascending=[True, True, True]
            )
    )

    return total


def q08(root: str, storage_options: Dict):
    part = load_part(root, storage_options)
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    customer = load_customer(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)
    region = load_region(root, storage_options)

    nation_name = "BRAZIL"
    region_name = "AMERICA"
    p_type = "ECONOMY ANODIZED STEEL"
    part_filtered = part[(part["P_TYPE"] == p_type)]
    part_filtered = part_filtered.loc[:, ["P_PARTKEY"]]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_SUPPKEY", "L_ORDERKEY"]]
    lineitem_filtered["VOLUME"] = lineitem["L_EXTENDEDPRICE"] * (1.0 - lineitem["L_DISCOUNT"])
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
        (orders["O_ORDERDATE"] >= pd.Timestamp("1995-01-01"))
        & (orders["O_ORDERDATE"] < pd.Timestamp("1997-01-01"))
    ]
    orders_filtered["O_YEAR"] = orders_filtered["O_ORDERDATE"].dt.year
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
    n2_filtered = nation.loc[:, ["N_NATIONKEY", "N_NAME"]] \
        .rename(columns={"N_NAME": "NATION"})
    total = total.merge(
        n1_filtered, left_on="C_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "S_NATIONKEY", "O_YEAR", "N_REGIONKEY"]]
    total = total.merge(
        n2_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "N_REGIONKEY", "NATION"]]
    region_filtered = region[(region["R_NAME"] == region_name)]
    region_filtered = region_filtered.loc[:, ["R_REGIONKEY"]]
    total = total.merge(
        region_filtered, left_on="N_REGIONKEY", right_on="R_REGIONKEY", how="inner"
    )
    total = total.loc[:, ["VOLUME", "O_YEAR", "NATION"]]

    def udf(df):
        demonimator = df["VOLUME"].sum()
        df = df[df["NATION"] == nation_name]
        numerator = df["VOLUME"].sum()
        return numerator / demonimator

    total = total.groupby("O_YEAR").apply(udf).reset_index()
    total.columns = ["O_YEAR", "MKT_SHARE"]
    total = total.sort_values(by=["O_YEAR"], ascending=[True])

    return total


def q09(root: str, storage_options: Dict):
    part = load_part(root, storage_options)
    partsupp = load_partsupp(root, storage_options)
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)

    p_name = "ghost"
    psel = part.P_NAME.str.contains(p_name)
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
    jn5["O_YEAR"] = jn5.O_ORDERDATE.dt.year
    gb = jn5.groupby(["N_NAME", "O_YEAR"], as_index=False)["TMP"].sum()
    total = gb.sort_values(["N_NAME", "O_YEAR"], ascending=[True, False])
    total = total.rename(columns={"TMP": "SUM_PROFIT"})

    return total


def q10(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    nation = load_nation(root, storage_options)
    customer = load_customer(root, storage_options)

    date1 = pd.Timestamp("1994-11-01")
    date2 = date1 + pd.DateOffset(months=3)
    osel = (orders.O_ORDERDATE >= date1) & (orders.O_ORDERDATE < date2)
    lsel = lineitem.L_RETURNFLAG == "R"
    forders = orders[osel]
    flineitem = lineitem[lsel]
    jn1 = flineitem.merge(forders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    jn3 = jn2.merge(nation, left_on="C_NATIONKEY", right_on="N_NATIONKEY")
    jn3["REVENUE"] = jn3.L_EXTENDEDPRICE * (1.0 - jn3.L_DISCOUNT)
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
        as_index=False,
    )["REVENUE"].sum()
    total = gb.sort_values("REVENUE", ascending=False)
    total = total.head(20)
    total = total[
        [
            "C_CUSTKEY",
            "C_NAME",
            "REVENUE",
            "C_ACCTBAL",
            "N_NAME",
            "C_ADDRESS",
            "C_PHONE",
            "C_COMMENT",
        ]
    ]

    return total


def q11(root: str, storage_options: Dict):
    partsupp = load_partsupp(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)

    nation_name = "GERMANY"
    fraction = 0.0001

    partsupp_filtered = partsupp.loc[:, ["PS_PARTKEY", "PS_SUPPKEY"]]
    partsupp_filtered["TOTAL_COST"] = (
        partsupp["PS_SUPPLYCOST"] * partsupp["PS_AVAILQTY"]
    )
    supplier_filtered = supplier.loc[:, ["S_SUPPKEY", "S_NATIONKEY"]]
    ps_supp_merge = partsupp_filtered.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="inner"
    )
    ps_supp_merge = ps_supp_merge.loc[:, ["PS_PARTKEY", "S_NATIONKEY", "TOTAL_COST"]]
    nation_filtered = nation[(nation["N_NAME"] == nation_name)]
    nation_filtered = nation_filtered.loc[:, ["N_NATIONKEY"]]
    ps_supp_n_merge = ps_supp_merge.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    ps_supp_n_merge = ps_supp_n_merge.loc[:, ["PS_PARTKEY", "TOTAL_COST"]]
    sum_val = ps_supp_n_merge["TOTAL_COST"].sum() * fraction
    total = ps_supp_n_merge.groupby(["PS_PARTKEY"], as_index=False).agg(
        VALUE=pd.NamedAgg(column="TOTAL_COST", aggfunc="sum")
    )
    total = total[total["VALUE"] > sum_val]
    total = total.sort_values("VALUE", ascending=False)

    return total


def q12(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)

    shipmode1 = "MAIL"
    shipmode2 = "SHIP"
    date1 = pd.Timestamp("1994-01-01")
    date2 = date1 + pd.DateOffset(years=1)
    sel = (
        (lineitem.L_RECEIPTDATE < date2)
        & (lineitem.L_COMMITDATE < date2)
        & (lineitem.L_SHIPDATE < date2)
        & (lineitem.L_SHIPDATE < lineitem.L_COMMITDATE)
        & (lineitem.L_COMMITDATE < lineitem.L_RECEIPTDATE)
        & (lineitem.L_RECEIPTDATE >= date1)
        & ((lineitem.L_SHIPMODE == shipmode1) | (lineitem.L_SHIPMODE == shipmode2))
    )
    flineitem = lineitem[sel]
    jn = flineitem.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")

    def g1(x):
        return ((x == "1-URGENT") | (x == "2-HIGH")).sum()

    def g2(x):
        return ((x != "1-URGENT") & (x != "2-HIGH")).sum()

    total = jn.groupby("L_SHIPMODE", as_index=False)["O_ORDERPRIORITY"].agg((g1, g2))
    total = (
        total.sort_values("L_SHIPMODE")
        .rename(columns={"g1": "HIGH_LINE_COUNT", "g2": "LOW_LINE_COUNT"})
    )
    return total


def q13(root: str, storage_options: Dict):
    customer = load_customer(root, storage_options)
    orders = load_orders(root, storage_options)

    word1 = "special"
    word2 = "requests"
    customer_filtered = customer.loc[:, ["C_CUSTKEY"]]
    orders_filtered = orders[
        ~orders["O_COMMENT"].str.contains(f"{word1}(\\S|\\s)*{word2}")
    ]
    orders_filtered = orders_filtered.loc[:, ["O_ORDERKEY", "O_CUSTKEY"]]
    c_o_merged = customer_filtered.merge(
        orders_filtered, left_on="C_CUSTKEY", right_on="O_CUSTKEY", how="left"
    )
    c_o_merged = c_o_merged.loc[:, ["C_CUSTKEY", "O_ORDERKEY"]]
    count_df = c_o_merged.groupby(["C_CUSTKEY"], as_index=False).agg(
        C_COUNT=pd.NamedAgg(column="O_ORDERKEY", aggfunc="count")
    )
    total = count_df.groupby(["C_COUNT"], as_index=False).size()
    total.columns = ["C_COUNT", "CUSTDIST"]
    total = total.sort_values(
        by=["CUSTDIST", "C_COUNT"],
        ascending=[False, False],
    )

    return total


def q14(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    part = load_part(root, storage_options)

    startDate = pd.Timestamp("1994-03-01")
    endDate = startDate + pd.DateOffset(months=1)
    p_type_like = "PROMO"
    part_filtered = part.loc[:, ["P_PARTKEY", "P_TYPE"]]
    lineitem_filtered = lineitem.loc[
        :, ["L_EXTENDEDPRICE", "L_DISCOUNT", "L_SHIPDATE", "L_PARTKEY"]
    ]
    sel = (lineitem_filtered.L_SHIPDATE >= startDate) \
        & (lineitem_filtered.L_SHIPDATE < endDate)
    flineitem = lineitem_filtered[sel]
    jn = flineitem.merge(part_filtered, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jn["PROMO_REVENUE"] = jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)
    total = (
        jn[jn.P_TYPE.str.startswith(p_type_like)].PROMO_REVENUE.sum() * 100 / jn.PROMO_REVENUE.sum()
    )

    result_df = pd.DataFrame({"PROMO_REVENUE": [total]})
    return result_df


def q15(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    supplier = load_supplier(root, storage_options)

    lineitem_filtered = lineitem[
        (lineitem["L_SHIPDATE"] >= pd.Timestamp("1996-01-01"))
        & (lineitem["L_SHIPDATE"] < (pd.Timestamp("1996-01-01") + pd.DateOffset(months=3)))
    ]
    lineitem_filtered["REVENUE_PARTS"] = lineitem_filtered["L_EXTENDEDPRICE"] * (
        1.0 - lineitem_filtered["L_DISCOUNT"]
    )
    lineitem_filtered = lineitem_filtered.loc[:, ["L_SUPPKEY", "REVENUE_PARTS"]]
    revenue_table = (
        lineitem_filtered.groupby("L_SUPPKEY", as_index=False)
        .agg(TOTAL_REVENUE=pd.NamedAgg(column="REVENUE_PARTS", aggfunc="sum"))
        .rename(columns={"L_SUPPKEY": "SUPPLIER_NO"})
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


def q16(root: str, storage_options: Dict):
    part = load_part(root, storage_options)
    partsupp = load_partsupp(root, storage_options)
    supplier = load_supplier(root, storage_options)

    brand = "Brand#45"
    p_type = "MEDIUM POLISHED"
    size1 = 49
    size2 = 14
    size3 = 23
    size4 = 45
    size5 = 19
    size6 = 3
    size7 = 36
    size8 = 9
    part_filtered = part[
        (part["P_BRAND"] != brand)
        & (~part["P_TYPE"].str.contains(f"^{p_type}"))
        & part["P_SIZE"].isin([size1, size2, size3, size4, size5, size6, size7, size8])
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
    # left merge to select only PS_SUPPKEY values not in supplier_filtered
    total = total.merge(
        supplier_filtered, left_on="PS_SUPPKEY", right_on="S_SUPPKEY", how="left"
    )
    total = total[total["S_SUPPKEY"].isna()]
    total = total.loc[:, ["P_BRAND", "P_TYPE", "P_SIZE", "PS_SUPPKEY"]]
    total = total.groupby(["P_BRAND", "P_TYPE", "P_SIZE"], as_index=False)["PS_SUPPKEY"] \
                .nunique()
    total.columns = ["P_BRAND", "P_TYPE", "P_SIZE", "SUPPLIER_CNT"]
    total = total.sort_values(
        by=["SUPPLIER_CNT", "P_BRAND", "P_TYPE", "P_SIZE"],
        ascending=[False, True, True, True],
    )

    return total


def q17(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    part = load_part(root, storage_options)

    brand = "Brand#23"
    container = "MED BOX"

    left = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY", "L_EXTENDEDPRICE"]]
    right = part[((part["P_BRAND"] == brand) & (part["P_CONTAINER"] == container))]
    right = right.loc[:, ["P_PARTKEY"]]
    line_part_merge = left.merge(
        right, left_on="L_PARTKEY", right_on="P_PARTKEY", how="inner"
    )
    line_part_merge = line_part_merge.loc[
        :, ["L_QUANTITY", "L_EXTENDEDPRICE", "P_PARTKEY"]
    ]
    lineitem_filtered = lineitem.loc[:, ["L_PARTKEY", "L_QUANTITY"]]
    lineitem_avg = lineitem_filtered.groupby(["L_PARTKEY"], as_index=False).agg(
        avg=pd.NamedAgg(column="L_QUANTITY", aggfunc="mean")
    )
    lineitem_avg["avg"] = 0.2 * lineitem_avg["avg"]
    lineitem_avg = lineitem_avg.loc[:, ["L_PARTKEY", "avg"]]
    total = line_part_merge.merge(
        lineitem_avg, left_on="P_PARTKEY", right_on="L_PARTKEY", how="inner"
    )
    total = total[total["L_QUANTITY"] < total["avg"]]
    total = pd.DataFrame({"AVG_YEARLY": [total["L_EXTENDEDPRICE"].sum() / 7.0]})

    return total


def q18(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    customer = load_customer(root, storage_options)

    quantity = 300
    gb1 = lineitem.groupby("L_ORDERKEY", as_index=False)["L_QUANTITY"].sum()
    fgb1 = gb1[gb1.L_QUANTITY > quantity]
    jn1 = fgb1.merge(orders, left_on="L_ORDERKEY", right_on="O_ORDERKEY")
    jn2 = jn1.merge(customer, left_on="O_CUSTKEY", right_on="C_CUSTKEY")
    gb2 = jn2.groupby(
        ["C_NAME", "C_CUSTKEY", "O_ORDERKEY", "O_ORDERDATE", "O_TOTALPRICE"],
        as_index=False,
    )["L_QUANTITY"].sum()
    total = gb2.sort_values(["O_TOTALPRICE", "O_ORDERDATE"], ascending=[False, True])
    total = total.head(100)

    return total


def q19(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    part = load_part(root, storage_options)

    quantity1 = 4
    quantity2 = 15
    quantity3 = 26
    brand1 = "Brand#31"
    brand2 = "Brand#43"

    lsel = (
        (
            (
                (lineitem.L_QUANTITY <= quantity3 + 10)
                & (lineitem.L_QUANTITY >= quantity3)
            )
            | (
                (lineitem.L_QUANTITY <= quantity2 + 10)
                & (lineitem.L_QUANTITY >= quantity2)
            )
            | (
                (lineitem.L_QUANTITY <= quantity1 + 10)
                & (lineitem.L_QUANTITY >= quantity1)
            )
        )
        & (lineitem.L_SHIPINSTRUCT == "DELIVER IN PERSON")
        & ((lineitem.L_SHIPMODE == "AIR") | (lineitem.L_SHIPMODE == "AIRREG"))
    )
    psel = (part.P_SIZE >= 1) & (
        (
            (part.P_SIZE <= 5)
            & (part.P_BRAND == brand1)
            & (
                (part.P_CONTAINER == "SM BOX")
                | (part.P_CONTAINER == "SM CASE")
                | (part.P_CONTAINER == "SM PACK")
                | (part.P_CONTAINER == "SM PKG")
            )
        )
        | (
            (part.P_SIZE <= 10)
            & (part.P_BRAND == brand2)
            & (
                (part.P_CONTAINER == "MED BAG")
                | (part.P_CONTAINER == "MED BOX")
                | (part.P_CONTAINER == "MED PACK")
                | (part.P_CONTAINER == "MED PKG")
            )
        )
        | (
            (part.P_SIZE <= 15)
            & (part.P_BRAND == brand2)
            & (
                (part.P_CONTAINER == "LG BOX")
                | (part.P_CONTAINER == "LG CASE")
                | (part.P_CONTAINER == "LG PACK")
                | (part.P_CONTAINER == "LG PKG")
            )
        )
    )
    flineitem = lineitem[lsel]
    fpart = part[psel]
    jn = flineitem.merge(fpart, left_on="L_PARTKEY", right_on="P_PARTKEY")
    jnsel = (
        (jn.P_BRAND == brand1)
        & (
            (jn.P_CONTAINER == "SM BOX")
            | (jn.P_CONTAINER == "SM CASE")
            | (jn.P_CONTAINER == "SM PACK")
            | (jn.P_CONTAINER == "SM PKG")
        )
        & (jn.L_QUANTITY >= quantity1)
        & (jn.L_QUANTITY <= quantity1 + 10)
        & (jn.P_SIZE <= 5)
        | (jn.P_BRAND == brand2)
        & (
            (jn.P_CONTAINER == "MED BAG")
            | (jn.P_CONTAINER == "MED BOX")
            | (jn.P_CONTAINER == "MED PACK")
            | (jn.P_CONTAINER == "MED PKG")
        )
        & (jn.L_QUANTITY >= quantity2)
        & (jn.L_QUANTITY <= quantity2 + 10)
        & (jn.P_SIZE <= 10)
        | (jn.P_BRAND == brand2)
        & (
            (jn.P_CONTAINER == "LG BOX")
            | (jn.P_CONTAINER == "LG CASE")
            | (jn.P_CONTAINER == "LG PACK")
            | (jn.P_CONTAINER == "LG PKG")
        )
        & (jn.L_QUANTITY >= quantity3)
        & (jn.L_QUANTITY <= quantity3 + 10)
        & (jn.P_SIZE <= 15)
    )
    jn = jn[jnsel]
    result_value = (jn.L_EXTENDEDPRICE * (1.0 - jn.L_DISCOUNT)).sum()
    result_df = pd.DataFrame({"REVENUE": [result_value]})

    return result_df


def q20(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    part = load_part(root, storage_options)
    nation = load_nation(root, storage_options)
    partsupp = load_partsupp(root, storage_options)
    supplier = load_supplier(root, storage_options)

    p_name = "azure"
    date1 = pd.Timestamp("1996-01-01")
    date2 = date1 + pd.DateOffset(years=1)
    psel = part.P_NAME.str.startswith(p_name)
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
    gb = jn2.groupby(["PS_PARTKEY", "PS_SUPPKEY", "PS_AVAILQTY"], as_index=False)[
        "L_QUANTITY"
    ].sum()
    gbsel = gb.PS_AVAILQTY > (0.5 * gb.L_QUANTITY)
    fgb = gb[gbsel]
    jn3 = fgb.merge(supplier, left_on="PS_SUPPKEY", right_on="S_SUPPKEY")
    jn4 = fnation.merge(jn3, left_on="N_NATIONKEY", right_on="S_NATIONKEY")
    jn4 = jn4.loc[:, ["S_NAME", "S_ADDRESS"]]
    total = jn4.sort_values("S_NAME").drop_duplicates()

    return total


def q21(root: str, storage_options: Dict):
    lineitem = load_lineitem(root, storage_options)
    orders = load_orders(root, storage_options)
    supplier = load_supplier(root, storage_options)
    nation = load_nation(root, storage_options)

    nation_name = "SAUDI ARABIA"
    lineitem_filtered = lineitem.loc[
        :, ["L_ORDERKEY", "L_SUPPKEY", "L_RECEIPTDATE", "L_COMMITDATE"]
    ]

    # Exists
    lineitem_orderkeys = (
        lineitem_filtered.loc[:, ["L_ORDERKEY", "L_SUPPKEY"]]
        .groupby("L_ORDERKEY", as_index=False)["L_SUPPKEY"]
        .nunique()
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
        lineitem_filtered.groupby("L_ORDERKEY", as_index=False)["L_SUPPKEY"]
        .nunique()
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
    nation_filtered = nation_filtered[nation_filtered["N_NAME"] == nation_name]
    total = total.merge(
        nation_filtered, left_on="S_NATIONKEY", right_on="N_NATIONKEY", how="inner"
    )
    total = total.loc[:, ["S_NAME"]]
    total = total.groupby("S_NAME", as_index=False).size()
    total.columns = ["S_NAME", "NUMWAIT"]
    total = total.sort_values(by=["NUMWAIT", "S_NAME"], ascending=[False, True])
    total = total.head(100)

    return total


def q22(root: str, storage_options: Dict):
    customer = load_customer(root, storage_options)
    orders = load_orders(root, storage_options)

    I1 = "13"
    I2 = "31"
    I3 = "23"
    I4 = "29"
    I5 = "30"
    I6 = "18"
    I7 = "17"
    customer_filtered = customer.loc[:, ["C_ACCTBAL", "C_CUSTKEY"]]
    customer_filtered["CNTRYCODE"] = customer["C_PHONE"].str.slice(0, 2)
    customer_filtered = customer_filtered[
        (customer["C_ACCTBAL"] > 0.00)
        & customer_filtered["CNTRYCODE"].isin([I1, I2, I3, I4, I5, I6, I7])
    ]
    avg_value = customer_filtered["C_ACCTBAL"].mean()
    customer_filtered = customer_filtered[customer_filtered["C_ACCTBAL"] > avg_value]
    # Select only the keys that don't match by performing a left join and only selecting columns with an na value
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
    agg1 = customer_selected.groupby(["CNTRYCODE"], as_index=False).size()
    agg1.columns = ["CNTRYCODE", "NUMCUST"]
    agg2 = customer_selected.groupby(["CNTRYCODE"], as_index=False).agg(
        TOTACCTBAL=pd.NamedAgg(column="C_ACCTBAL", aggfunc="sum")
    )
    total = agg1.merge(agg2, on="CNTRYCODE", how="inner")
    total = total.sort_values(by=["CNTRYCODE"], ascending=[True])

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


def run_queries(
    path,
    storage_options,
    queries,
    log_time=True,
    print_result=False,
    include_io=False,
):
    version = modin.__version__
    data_start_time = time.time()
    for query in queries:
        loaders = query_to_loaders[query]
        for loader in loaders:
            loader(path, storage_options)
    print(f"Total data loading time (s): {time.time() - data_start_time}")

    total_start = time.time()
    for query in queries:
        try:
            start_time = time.time()
            result = query_to_runner[query](path, storage_options)
            without_io_time = time.time() - start_time
            success = True
            if print_result:
                print_result_fn("modin_ray", result, query)
        except Exception as e:
            print("".join(traceback.TracebackException.from_exception(e).format()))
            without_io_time = 0.0
            success = False
        finally:
            pass
        if log_time:
            log_time_fn("modin_ray", query, version=version, without_io_time=without_io_time, success=success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
    parser.add_argument(
        "--storage_options",
        type=str,
        required=False,
        help="storage options json file.",
    )
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
    run_queries(args.path, storage_options, queries, args.log_time, args.print_result, args.include_io)


if __name__ == "__main__":
    main()
