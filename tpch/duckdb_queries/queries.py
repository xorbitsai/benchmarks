import os
import time
import argparse
import traceback

import pandas as pd
pd.set_option('display.max_columns', None)

import duckdb
from duckdb import DuckDBPyRelation

from common_utils import log_time_fn, parse_common_arguments, print_result_fn

dataset_dict = {}


def create_table(path: str, talbe_name: str):
    duckdb.sql(
        f"create temp table if not exists {talbe_name} as select * from read_parquet('{path}/*.parquet');"
    )
    return talbe_name


def load_lineitem(root: str):
    if "lineitem" not in dataset_dict:
        data_path = root + "/lineitem"
        df = create_table(data_path, "LINEITEM")
        dataset_dict["lineitem"] = df
    else:
        df = dataset_dict["lineitem"]
    return df


def load_part(root: str):
    if "part" not in dataset_dict:
        data_path = root + "/part"
        df = create_table(data_path, "PART")
        dataset_dict["part"] = df
    else:
        df = dataset_dict["part"]
    return df


def load_orders(root: str):
    if "orders" not in dataset_dict:
        data_path = root + "/orders"
        df = create_table(data_path, "ORDERS")
        dataset_dict["orders"] = df
    else:
        df = dataset_dict["orders"]
    return df


def load_customer(root: str):
    if "customer" not in dataset_dict:
        data_path = root + "/customer"
        df = create_table(data_path, "CUSTOMER")
        dataset_dict["customer"] = df
    else:
        df = dataset_dict["customer"]
    return df


def load_nation(root: str):
    if "nation" not in dataset_dict:
        data_path = root + "/nation"
        df = create_table(data_path, "NATION")
        dataset_dict["nation"] = df
    else:
        df = dataset_dict["nation"]
    return df


def load_region(root: str):
    if "region" not in dataset_dict:
        data_path = root + "/region"
        df = create_table(data_path, "REGION")
        dataset_dict["region"] = df
    else:
        df = dataset_dict["region"]
    return df


def load_supplier(root: str):
    if "supplier" not in dataset_dict:
        data_path = root + "/supplier"
        df = create_table(data_path, "SUPPLIER")
        dataset_dict["supplier"] = df
    else:
        df = dataset_dict["supplier"]
    return df


def load_partsupp(root: str):
    if "partsupp" not in dataset_dict:
        data_path = root + "/partsupp"
        df = create_table(data_path, "PARTSUPP")
        dataset_dict["partsupp"] = df
    else:
        df = dataset_dict["partsupp"]
    return df


def q01(root: str):
    lineitem = load_lineitem(root)

    total = duckdb.sql(
        """SELECT
                L_RETURNFLAG,
                L_LINESTATUS,
                SUM(L_QUANTITY) AS SUM_QTY,
                SUM(L_EXTENDEDPRICE) AS SUM_BASE_PRICE,
                SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS SUM_DISC_PRICE,
                SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT) * (1 + L_TAX)) AS SUM_CHARGE,
                AVG(L_QUANTITY) AS AVG_QTY,
                AVG(L_EXTENDEDPRICE) AS AVG_PRICE,
                AVG(L_DISCOUNT) AS AVG_DISC,
                COUNT(*) AS COUNT_ORDER
            FROM
                LINEITEM
            WHERE
                L_SHIPDATE <= DATE '1998-12-01' - INTERVAL '90' DAY
            GROUP BY
                L_RETURNFLAG,
                L_LINESTATUS
            ORDER BY
                L_RETURNFLAG,
                L_LINESTATUS"""
    )
    return total


def q02(root: str):
    nation = load_nation(root)
    region = load_region(root)
    supplier = load_supplier(root)
    part = load_part(root)
    partsupp = load_partsupp(root)

    SIZE = 15
    TYPE = "BRASS"
    REGION = "EUROPE"
    total = duckdb.sql(
        f"""SELECT
                S_ACCTBAL,
                S_NAME,
                N_NAME,
                P_PARTKEY,
                P_MFGR,
                S_ADDRESS,
                S_PHONE,
                S_COMMENT
            FROM
                PART,
                SUPPLIER,
                PARTSUPP,
                NATION,
                REGION
            WHERE
                P_PARTKEY = PS_PARTKEY
                AND S_SUPPKEY = PS_SUPPKEY
                AND P_SIZE = {SIZE}
                AND P_TYPE LIKE '%{TYPE}'
                AND S_NATIONKEY = N_NATIONKEY
                AND N_REGIONKEY = R_REGIONKEY
                AND R_NAME = '{REGION}'
                AND PS_SUPPLYCOST = (
                    SELECT
                    MIN(PS_SUPPLYCOST)
                    FROM
                    PARTSUPP, SUPPLIER,
                    NATION, REGION
                    WHERE
                    P_PARTKEY = PS_PARTKEY
                    AND S_SUPPKEY = PS_SUPPKEY
                    AND S_NATIONKEY = N_NATIONKEY
                    AND N_REGIONKEY = R_REGIONKEY
                    AND R_NAME = '{REGION}'
                    )
            ORDER BY
                S_ACCTBAL DESC,
                N_NAME,
                S_NAME,
                P_PARTKEY"""
    )

    return total


def q03(root: str):
    lineitem = load_lineitem(root)
    customer = load_customer(root)
    orders = load_orders(root)

    MKTSEGMENT = "HOUSEHOLD"
    DATE = "1995-03-04"
    total = duckdb.sql(
        f"""SELECT
            L_ORDERKEY,
            SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS REVENUE,
            O_ORDERDATE,
            O_SHIPPRIORITY
        FROM
            CUSTOMER,
            orders,
            LINEITEM
        WHERE
            C_MKTSEGMENT = '{MKTSEGMENT}'
            AND C_CUSTKEY = O_CUSTKEY
            AND L_ORDERKEY = O_ORDERKEY
            AND O_ORDERDATE < DATE '{DATE}'
            AND L_SHIPDATE > DATE '{DATE}'
        GROUP BY
            L_ORDERKEY,
            O_ORDERDATE,
            O_SHIPPRIORITY
        ORDER BY
            REVENUE DESC,
            O_ORDERDATE
        LIMIT 10"""
    )
    return total


def q04(root: str):
    line_item = load_lineitem(root)
    orders = load_orders(root)
    
    DATE = "1993-08-01"
    total = duckdb.sql(
        f"""SELECT
                O_ORDERPRIORITY,
                COUNT(*) AS ORDER_COUNT
            FROM
                orders
            WHERE
                O_ORDERDATE >= DATE '{DATE}'
                AND O_ORDERDATE < DATE '{DATE}' + INTERVAL '3' MONTH
                AND EXISTS (
                    SELECT
                        *
                    FROM
                        LINEITEM
                    WHERE
                        L_ORDERKEY = O_ORDERKEY
                        AND L_COMMITDATE < L_RECEIPTDATE
                )
            GROUP BY
                O_ORDERPRIORITY
            ORDER BY
                O_ORDERPRIORITY"""
    )

    return total


def q05(root: str):
    orders = load_orders(root)
    region = load_region(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    lineitem = load_lineitem(root)
    customer = load_customer(root)

    REGION = "ASIA"
    DATE = "1996-01-01"
    total = duckdb.sql(
        f"""SELECT
                N_NAME,
                SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS REVENUE
            FROM
                CUSTOMER,
                orders,
                LINEITEM,
                SUPPLIER,
                NATION,
                REGION
            WHERE
                C_CUSTKEY = O_CUSTKEY
                AND L_ORDERKEY = O_ORDERKEY
                AND L_SUPPKEY = S_SUPPKEY
                AND C_NATIONKEY = S_NATIONKEY
                AND S_NATIONKEY = N_NATIONKEY
                AND N_REGIONKEY = R_REGIONKEY
                AND R_NAME = '{REGION}'
                AND O_ORDERDATE >= DATE '{DATE}'
                AND O_ORDERDATE < DATE '{DATE}' + INTERVAL '1' YEAR
            GROUP BY
                N_NAME
            ORDER BY
                REVENUE DESC"""
    )
    return total


def q06(root: str):
    lineitem = load_lineitem(root)

    DATE = "1996-01-01"
    total = duckdb.sql(
        f"""SELECT
                SUM(L_EXTENDEDPRICE * L_DISCOUNT) AS REVENUE
            FROM
                LINEITEM
            WHERE
                L_SHIPDATE >= DATE '{DATE}'
                AND L_SHIPDATE < DATE '{DATE}' + INTERVAL '1' YEAR
                AND L_DISCOUNT BETWEEN .08 AND .1
                AND L_QUANTITY < 24"""
    )
    return total


def q07(root: str):
    orders = load_orders(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    lineitem = load_lineitem(root)
    customer = load_customer(root)
    
    NATION1 = "FRANCE"
    NATION2 = "GERMANY"
    total = duckdb.sql(
        f"""SELECT
                SUPP_NATION,
                CUST_NATION,
                L_YEAR, SUM(VOLUME) AS REVENUE
            FROM (
                SELECT
                    N1.N_NAME AS SUPP_NATION,
                    N2.N_NAME AS CUST_NATION,
                    EXTRACT(year FROM L_SHIPDATE) AS L_YEAR,
                    L_EXTENDEDPRICE * (1 - L_DISCOUNT) AS VOLUME
                FROM
                    SUPPLIER,
                    LINEITEM,
                    ORDERS,
                    CUSTOMER,
                    NATION N1,
                    NATION N2
                WHERE
                    S_SUPPKEY = L_SUPPKEY
                    AND O_ORDERKEY = L_ORDERKEY
                    AND C_CUSTKEY = O_CUSTKEY
                    AND S_NATIONKEY = N1.N_NATIONKEY
                    AND C_NATIONKEY = N2.N_NATIONKEY
                    AND (
                    (N1.N_name = '{NATION1}' AND N2.N_NAME = '{NATION2}')
                    OR (N1.N_NAME = '{NATION2}' AND N2.N_NAME = '{NATION1}')
                    )
                    AND L_SHIPDATE BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                ) AS SHIPPING
            GROUP BY
                SUPP_NATION,
                CUST_NATION,
                L_YEAR
            ORDER BY
                SUPP_NATION,
                CUST_NATION,
                L_YEAR"""
    )
    return total


def q08(root: str):
    part = load_part(root)
    region = load_region(root)
    orders = load_orders(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    lineitem = load_lineitem(root)
    customer = load_customer(root)
    
    NATION = "BRAZIL"
    REGION = "AMERICA"
    TYPE = "ECONOMY ANODIZED STEEL"
    total = duckdb.sql(
        F"""SELECT
                O_YEAR,
                SUM(CASE
                    WHEN NAtion = '{NATION}'
                    THEN VOLUME
                    ELSE 0
                END) / SUM(VOLUME) AS MKT_SHARE
            FROM (
                SELECT
                    EXTRACT(YEAR FROM O_ORDERDATE) AS O_YEAR,
                    L_EXTENDEDPRICE * (1-L_DISCOUNT) AS VOLUME,
                    N2.N_NAME AS NATION
                FROM
                    PART,
                    SUPPLIER,
                    LINEITEM,
                    ORDERS,
                    CUSTOMER,
                    NATION N1,
                    NATION N2,
                    REGION
                WHERE
                    P_PARTKEY = L_PARTKEY
                    AND S_SUPPKEY = L_SUPPKEY
                    AND L_ORDERKEY = O_ORDERKEY
                    AND O_CUSTKEY = C_CUSTKEY
                    AND C_NATIONKEY = N1.N_NATIONKEY
                    AND N1.N_REGIONKEY = R_REGIONKEY
                    AND R_NAME = '{REGION}'
                    AND S_NATIONKEY = N2.N_NATIONKEY
                    AND O_ORDERDATE BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'
                    AND P_TYPE = '{TYPE}'
                ) AS ALL_NATIONS
            GROUP BY
                O_YEAR
            ORDER BY
                O_YEAR"""
    )

    return total


def q09(root: str):
    part = load_part(root)
    orders = load_orders(root)
    nation = load_nation(root)
    supplier = load_supplier(root)
    lineitem = load_lineitem(root)
    partsupp = load_partsupp(root)

    NAME = "ghost"

    total = duckdb.sql(
        f"""SELECT
                NATION,
                O_YEAR,
                SUM(AMOUNT) AS SUM_PROFIT
            FROM
                (
                    SELECT
                        N_NAME AS NATION,
                        year(O_ORDERDATE) AS O_YEAR,
                        L_EXTENDEDPRICE * (1 - L_DISCOUNT) - PS_SUPPLYCOST * L_QUANTITY AS AMOUNT
                    FROM
                        PART,
                        SUPPLIER,
                        LINEITEM,
                        PARTSUPP,
                        orders,
                        NATION
                    WHERE
                        S_SUPPKEY = L_SUPPKEY
                        AND PS_SUPPKEY = L_SUPPKEY
                        AND PS_PARTKEY = L_PARTKEY
                        AND P_PARTKEY = L_PARTKEY
                        AND O_ORDERKEY = L_ORDERKEY
                        AND S_NATIONKEY = N_NATIONKEY
                        AND P_NAME LIKE '%{NAME}%'
                ) AS PROFIT
            GROUP BY
                NATION,
                O_YEAR
            ORDER BY
                NATION,
                O_YEAR DESC"""
    )
    return total


def q10(root: str):
    orders = load_orders(root)
    nation = load_nation(root)
    lineitem = load_lineitem(root)
    customer = load_customer(root)

    DATE = "1994-11-01"
    total = duckdb.sql(
        f"""SELECT
                C_CUSTKEY,
                C_NAME,
                SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS REVENUE,
                C_ACCTBAL,
                N_NAME,
                C_ADDRESS,
                C_PHONE,
                C_COMMENT
            FROM
                CUSTOMER,
                ORDERS,
                LINEITEM,
                NATION
            WHERE
                C_CUSTKEY = O_CUSTKEY
                AND L_ORDERKEY = O_ORDERKEY
                AND O_ORDERDATE >= DATE '{DATE}'
                AND O_ORDERDATE < DATE '{DATE}' + INTERVAL '3' MONTH
                AND L_RETURNFLAG = 'R'
                AND C_NATIONKEY = N_NATIONKEY
            GROUP BY
                C_CUSTKEY,
                C_NAME,
                C_ACCTBAL,
                C_PHONE,
                N_NAME,
                C_ADDRESS,
                C_COMMENT
            ORDER BY
                REVENUE DESC
            LIMIT 20"""
    )
    return total

# todo: result is empty
def q11(root: str):
    partsupp = load_partsupp(root)
    supplier = load_supplier(root)
    nation = load_nation(root)
   
    NATION = "GERMANY"
    FRACTION = 0.0001

    total = duckdb.sql(
        f"""SELECT
                PS_PARTKEY,
                SUM(PS_SUPPLYCOST * PS_AVAILQTY) AS VALUE
            FROM
                PARTSUPP,
                SUPPLIER,
                NATION
            WHERE
                PS_SUPPKEY = S_SUPPKEY
                AND S_NATIONKEY = N_NATIONKEY
                AND N_NAME = '{NATION}'
            GROUP BY
                PS_PARTKEY HAVING
                        SUM(PS_SUPPLYCOST * PS_AVAILQTY) > (
                    SELECT
                        SUM(PS_SUPPLYCOST * PS_AVAILQTY) * {FRACTION}
                    FROM
                        PARTSUPP,
                        SUPPLIER,
                        NATION
                    WHERE
                        PS_SUPPKEY = S_SUPPKEY
                        AND S_NATIONKEY = N_NATIONKEY
                        AND N_NAME = '{NATION}'
                    )
                ORDER BY
                    VALUE DESC"""
    )

    return total


def q12(root):
    lineitem = load_lineitem(root)
    orders = load_orders(root)

    SHIPMODE1 = "MAIL"
    SHIPMODE2 = "SHIP"
    DATE = "1994-01-01"
    total = duckdb.sql(
        f"""SELECT
                L_SHIPMODE,
                SUM(CASE
                    WHEN O_ORDERPRIORITY = '1-URGENT'
                        OR O_ORDERPRIORITY = '2-HIGH'
                        THEN 1
                    ELSE 0
                END) AS HIGH_LINE_COUNT,
                SUM(CASE
                    WHEN O_ORDERPRIORITY <> '1-URGENT'
                        AND O_ORDERPRIORITY <> '2-HIGH'
                        THEN 1
                    ELSE 0
                END) AS LOW_LINE_COUNT
            FROM
                orders,
                LINEITEM
            WHERE
                O_ORDERKEY = L_ORDERKEY
                AND L_SHIPMODE IN ('{SHIPMODE1}', '{SHIPMODE2}')
                AND L_COMMITDATE < L_RECEIPTDATE
                AND L_SHIPDATE < L_COMMITDATE
                AND L_RECEIPTDATE >= DATE '{DATE}'
                AND L_RECEIPTDATE < DATE '{DATE}' + INTERVAL '1' YEAR
            GROUP BY
                L_SHIPMODE
            ORDER BY
                L_SHIPMODE"""
    )

    return total


def q13(root: str):
    customer = load_customer(root)
    orders = load_orders(root)

    WORD1 = "special"
    WORD2 = "requests"
    total = duckdb.sql(
        F"""SELECT
                C_COUNT, COUNT(*) AS CUSTDIST
            FROM (
                SELECT
                    C_CUSTKEY,
                    COUNT(O_ORDERKEY)
                FROM
                    CUSTOMER LEFT OUTER JOIN orders ON
                    C_CUSTKEY = O_CUSTKEY
                    AND O_COMMENT NOT LIKE '%{WORD1}%{WORD2}%'
                GROUP BY
                    C_CUSTKEY
                )AS C_orders (C_CUSTKEY, C_COUNT)
            GROUP BY
                C_COUNT
            ORDER BY
                CUSTDIST DESC,
                C_COUNT DESC"""
    )
    return total


def q14(root):
    lineitem = load_lineitem(root)
    part = load_part(root)
    
    DATE = "1994-03-01"
    total = duckdb.sql(
        f"""SELECT
                100.00 * SUM(CASE
                    WHEN P_TYPE LIKE 'PROMO%'
                        THEN L_EXTENDEDPRICE * (1 - L_DISCOUNT)
                    ELSE 0
                END) / SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS PROMO_REVENUE
            FROM
                LINEITEM,
                PART
            WHERE
                L_PARTKEY = P_PARTKEY
                AND L_SHIPDATE >= DATE '{DATE}'
                AND L_SHIPDATE < DATE '{DATE}' + INTERVAL '1' MONTH"""
    )
    return total


def q15(root):
    lineitem = load_lineitem(root)
    supplier = load_supplier(root)

    DATE = "1996-01-01"
    _ = duckdb.execute(
        f"""CREATE TEMP VIEW REVENUE (SUPPLIER_NO, TOTAL_REVENUE) AS
                SELECT
                    L_SUPPKEY,
                    CAST(SUM(L_EXTENDEDPRICE * (1 - L_DISCOUNT)) AS DECIMAL(12,2))
                FROM
                    LINEITEM
                WHERE
                    L_SHIPDATE >= DATE '{DATE}'
                    AND L_SHIPDATE < DATE '{DATE}' + INTERVAL '3' MONTH
                GROUP BY
                    L_SUPPKEY"""
    )
    total = duckdb.sql(
        """SELECT
                S_SUPPKEY,
                S_NAME,
                S_ADDRESS,
                S_PHONE,
                TOTAL_REVENUE
            FROM
                SUPPLIER,
                REVENUE
            WHERE
                S_SUPPKEY = SUPPLIER_NO
                AND TOTAL_REVENUE = (
                    SELECT
                        MAX(TOTAL_REVENUE)
                    FROM
                        REVENUE
                )
            ORDER BY
                S_SUPPKEY"""
    )
    # _ = duckdb.execute(
    #     "DROP VIEW REVENUE"
    # )
    return total


def q16(root):
    part = load_part(root)
    partsupp = load_partsupp(root)
    supplier = load_supplier(root)

    BRAND = "Brand#45"
    TYPE = "MEDIUM POLISHED"
    SIZE1 = 49
    SIZE2 = 14
    SIZE3 = 23
    SIZE4 = 45
    SIZE5 = 19
    SIZE6 = 3
    SIZE7 = 36
    SIZE8 = 9
    total = duckdb.sql(
        F"""SELECT
                P_BRAND,
                P_TYPE,
                P_SIZE,
                COUNT(DISTINCT PS_SUPPKEY) AS SUPPLIER_CNT
            FROM
                PARTSUPP,
                PART
            WHERE
                P_PARTKEY = PS_PARTKEY
                AND P_BRAND <> '{BRAND}'
                AND P_TYPE NOT LIKE '{TYPE}%'
                AND P_SIZE IN ({SIZE1}, {SIZE2}, {SIZE3}, {SIZE4}, {SIZE5}, {SIZE6}, {SIZE7}, {SIZE8})
                AND PS_SUPPKEY NOT IN (
                    SELECT
                        S_SUPPKEY
                    FROM
                        SUPPLIER
                    WHERE
                        S_COMMENT LIKE '%CUSTOMER%COMPLAINTS%'
                )
            GROUP BY
                P_BRAND,
                P_TYPE,
                P_SIZE
            ORDER BY
                SUPPLIER_CNT DESC,
                P_BRAND,
                P_TYPE,
                P_SIZE"""
    )
    return total


def q17(root):
    lineitem = load_lineitem(root)
    part = load_part(root)

    BRAND = "Brand#23"
    CONTAINER = "MED BOX"
    total = duckdb.sql(
        f"""SELECT
                SUM(L_EXTENDEDPRICE) / 7.0 AS AVG_YEARLY
            FROM
                LINEITEM,
                PART
            WHERE
                P_PARTKEY = L_PARTKEY
                AND P_BRAND = '{BRAND}'
                AND P_CONTAINER = '{CONTAINER}'
                AND L_QUANTITY < (
                    SELECT
                        0.2 * AVG(L_QUANTITY)
                    FROM
                        LINEITEM
                    WHERE
                        L_PARTKEY = P_PARTKEY
                )"""
    )
    return total


def q18(root):
    lineitem = load_lineitem(root)
    orders = load_orders(root)
    customer = load_customer(root)

    QUANTITY = 300
    total = duckdb.sql(
        f"""SELECT
                C_NAME,
                C_CUSTKEY,
                O_ORDERKEY,
                O_ORDERDATE,
                O_TOTALPRICE,
                SUM(L_QUANTITY)
            FROM
                CUSTOMER,
                orders,
                LINEITEM
            WHERE
                O_ORDERKEY IN (
                    SELECT
                        L_ORDERKEY
                    FROM
                        LINEITEM
                    GROUP BY
                        L_ORDERKEY HAVING
                            SUM(L_QUANTITY) > {QUANTITY}
                )
                AND C_CUSTKEY = O_CUSTKEY
                AND O_ORDERKEY = L_ORDERKEY
            GROUP BY
                C_NAME,
                C_CUSTKEY,
                O_ORDERKEY,
                O_ORDERDATE,
                O_TOTALPRICE
            ORDER BY
                O_TOTALPRICE DESC,
                O_ORDERDATE
            LIMIT 100"""
    )
    return total


def q19(root):
    lineitem = load_lineitem(root)
    part = load_part(root)

    BRAND = "Brand#31"
    QUANTITY = 4
    total = duckdb.sql(
        f"""SELECT
                SUM(L_EXTENDEDPRICE* (1 - L_DISCOUNT)) AS REVENUE
            FROM
                LINEITEM,
                PART
            WHERE
                (
                    P_PARTKEY = L_PARTKEY
                    AND P_BRAND = '{BRAND}'
                    AND P_CONTAINER IN ('SM CASE', 'SM BOX', 'SM PACK', 'SM PKG')
                    AND L_QUANTITY >= {QUANTITY} AND L_QUANTITY <= {QUANTITY} + 10
                    AND P_SIZE BETWEEN 1 AND 5
                    AND L_SHIPMODE IN ('AIR', 'AIR REG')
                    AND L_SHIPINSTRUCT = 'DELIVER IN PERSON'
                )
                OR
                (
                    P_PARTKEY = L_PARTKEY
                    AND P_BRAND = 'BRAND#43'
                    AND P_CONTAINER IN ('MED BAG', 'MED BOX', 'MED PKG', 'MED PACK')
                    AND L_QUANTITY >= 15 AND L_QUANTITY <= 25
                    AND P_SIZE BETWEEN 1 AND 10
                    AND L_SHIPMODE IN ('AIR', 'AIR REG')
                    AND L_SHIPINSTRUCT = 'DELIVER IN PERSON'
                )
                OR
                (
                    P_PARTKEY = L_PARTKEY
                    AND P_BRAND = 'BRAND#43'
                    AND P_CONTAINER IN ('LG CASE', 'LG BOX', 'LG PACK', 'LG PKG')
                    AND L_QUANTITY >= 26 AND L_QUANTITY <= 36
                    AND P_SIZE BETWEEN 1 AND 15
                    AND L_SHIPMODE IN ('AIR', 'AIR REG')
                    AND L_SHIPINSTRUCT = 'DELIVER IN PERSON'
                )"""
    )

    return total


def q20(root):
    lineitem = load_lineitem(root)
    part = load_part(root)
    nation = load_nation(root)
    partsupp = load_partsupp(root)
    supplier = load_supplier(root)

    NAME = "azure"
    DATE = "1996-01-01"
    total = duckdb.sql(
        f"""SELECT
                S_NAME,
                S_ADDRESS
            FROM
                SUPPLIER,
                NATION
            WHERE
                S_SUPPKEY IN (
                    SELECT
                        PS_SUPPKEY
                    FROM
                        PARTSUPP
                    WHERE
                        PS_PARTKEY IN (
                            SELECT
                                P_PARTKEY
                            FROM
                                PART
                            WHERE
                                P_NAME LIKE '{NAME}%'
                        )
                        AND PS_AVAILQTY > (
                            SELECT
                                0.5 * SUM(L_QUANTITY)
                            FROM
                                LINEITEM
                            WHERE
                                L_PARTKEY = PS_PARTKEY
                                AND L_SUPPKEY = PS_SUPPKEY
                                AND L_SHIPDATE >= DATE '{DATE}'
                                AND L_SHIPDATE < DATE '{DATE}' + INTERVAL '1' YEAR
                        )
                )
                AND S_NATIONKEY = N_NATIONKEY
                AND N_NAME = 'JORDAN'
            ORDER BY
                S_NAME"""
    )

    return total


def q21(root):
    lineitem = load_lineitem(root)
    orders = load_orders(root)
    nation = load_nation(root)
    supplier = load_supplier(root)

    NATION = "SAUDI ARABIA"
    total = duckdb.sql(
        f"""SELECT
                S_NAME,
                COUNT(*) AS NUMWAIT
            FROM
                SUPPLIER,
                LINEITEM L1,
                orders,
                NATION
            WHERE
                S_SUPPKEY = L1.L_SUPPKEY
                AND O_ORDERKEY = L1.L_ORDERKEY
                AND O_ordersTATUS = 'F'
                AND L1.L_RECEIPTDATE > L1.L_COMMITDATE
                AND EXISTS (
                    SELECT
                        *
                    FROM
                        LINEITEM L2
                    WHERE
                        L2.L_ORDERKEY = L1.L_ORDERKEY
                        AND L2.L_SUPPKEY <> L1.L_SUPPKEY
                )
                AND NOT EXISTS (
                    SELECT
                        *
                    FROM
                        LINEITEM L3
                    WHERE
                        L3.L_ORDERKEY = L1.L_ORDERKEY
                        AND L3.L_SUPPKEY <> L1.L_SUPPKEY
                        AND L3.L_RECEIPTDATE > L3.L_COMMITDATE
                )
                AND S_NATIONKEY = N_NATIONKEY
                AND N_NAME = '{NATION}'
            GROUP BY
                S_NAME
            ORDER BY
                NUMWAIT DESC,
                S_NAME"""
    )
    return total


def q22(root):
    customer = load_customer(root)
    orders = load_orders(root)

    I1 = 13
    I2 = 31
    I3 = 23
    I4 = 29
    I5 = 30
    I6 = 18
    I7 = 17
    total = duckdb.sql(
        F"""SELECT
                CNTRYCODE,
                COUNT(*) AS NUMCUST,
                SUM(C_ACCTBAL) AS TOTACCTBAL
            FROM (
                SELECT
                    SUBSTRING(C_PHONE FROM 1 FOR 2) AS CNTRYCODE,
                    C_ACCTBAL
                FROM
                    CUSTOMER
                WHERE
                    SUBSTRING(C_PHONE FROM 1 FOR 2) IN
                        ('{I1}','{I2}','{I3}','{I4}','{I5}','{I6}','{I7}')
                    AND C_ACCTBAL > (
                        SELECT
                            AVG(C_ACCTBAL)
                        FROM
                            CUSTOMER
                        WHERE
                            C_ACCTBAL > 0.00
                            AND SUBSTRING (C_PHONE FROM 1 FOR 2) IN
                                ('{I1}','{I2}','{I3}','{I4}','{I5}','{I6}','{I7}')
                    )
                    AND NOT EXISTS (
                        SELECT
                            *
                        FROM
                            orders
                        WHERE
                            O_CUSTKEY = C_CUSTKEY
                    )
                ) AS CUSTSALE
            GROUP BY
                CNTRYCODE
            ORDER BY
                CNTRYCODE"""
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


def run_queries(
    path,
    queries,
    log_time=True,
    print_result=False,
    include_io=True
):
    print(f"print_result: {print_result}")
    version = duckdb.__version__
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
            result = result.df()
            without_io_time = time.time() - start_time
            success = True
            if print_result:
                print_result_fn("duckdb", result, query)
        except Exception as e:
            print("".join(traceback.TracebackException.from_exception(e).format()))
            without_io_time = 0.0
            success = False
        finally:
            pass
        if log_time:
            log_time_fn("duckdb", query, version=version, without_io_time=without_io_time, success=success)
    print(f"Total query execution time (s): {time.time() - total_start}")


def main():
    parser = argparse.ArgumentParser(description="TPC-H benchmark.")
    # aws settings
    parser.add_argument("--account", type=str, help="AWS access id")
    parser.add_argument("--key", type=str, help="AWS secret access key")
    parser.add_argument("--endpoint", type=str, help="AWS region endpoint related to your S3")
    parser = parse_common_arguments(parser)  
    args = parser.parse_args()
    path: str = args.path
    print(f"Path: {args.path}")

    queries = list(range(1, 23))
    if args.queries is not None:
        queries = args.queries
    print(f"Queries to run: {queries}")
    print(f"Include IO: {args.include_io}")

    if "s3://" in path:
        import boto3
        s3_client = boto3.client(
            's3', 
            aws_access_key_id=args.account,
            aws_secret_access_key=args.key
        )
    
    run_queries(path, 
                queries, 
                args.log_time, 
                args.print_result,
                args.include_io)


if __name__ == "__main__":
    main()