# TPC-H Benchmarks
We compared Xorbits to Dask, Pandas API on Spark, and Modin on Ray with TPC-H benchmarks at scale
factor 100 (~100 GB dataset) and 1000 (~1 TB dataset). The cluster for TPC-H SF 100 consists of an
r6i.large instance as the supervisor, and 5 r6i.4xlarge instances as the workers. The cluster for
TPC-H SF 1000 consists of an r6i.large instance as the supervisor, and 16 r6i.8xlarge instances.



## Software versions
- Xorbits: 0.1.0
- Dask: 2022.12.1
- Modin: 0.18.0
- Spark: 3.3.1

## SF1000: Xorbits
Although Xorbits is able to pass all the queries in a row, Dask, Pandas API on Spark and Modin 
failed on most of the queries. Thus, we are not able to compare the performance difference now, and
we plan to try again later.

![image info](https://xorbits.io/res/xorbits_1t.png)

|     |  Xorbits  |
|:---:|:---------:|
| Q01 |  141.810  |
| Q02 |   35.000  |
| Q03 |  194.630  |
| Q04 |  225.570  |
| Q05 |  185.560  |
| Q06 |  101.430  |
| Q07 |  157.150  |
| Q08 |  143.060  |
| Q09 |  249.270  |
| Q10 |  131.940  |
| Q11 |   31.823  |
| Q12 |   89.139  |
| Q13 |   40.157  |
| Q14 |   76.638  |
| Q15 |  108.130  |
| Q16 |   43.952  |
| Q17 |  304.620  |
| Q18 |  126.880  |
| Q19 |   69.102  |
| Q20 |  103.000  |
| Q21 |  454.020  |
| Q22 |  162.460  |
|Total|  3012.88  |

## SF100: Xorbits vs. Dask
Dask is a well-known "Pandas-like" framework for scaling Python workloads. The graph below 
illustrates the computing times of Xorbits and Dask for the TPC-H queries (excluding I/O). Q21 was 
excluded since Dask ran out of memory. Across all queries, Xorbits was found to be 7.3x faster than
Dask.

![image info](https://xorbits.io/res/benchmark_dask.png)

|       |   Dask   | Xorbits | Speed up ratio |
|-------|:--------:|:-------:|:--------------:|
| Q01   |   8.33   |  14.64  |      0.57      |
| Q02   |   34.44  |   8.02  |      4.30      |
| Q03   |   43.19  |  37.27  |      1.16      |
| Q04   |  154.41  |  15.91  |      9.70      |
| Q05   |  294.43  |  29.04  |      10.14     |
| Q06   |   18.85  |   8.83  |      2.13      |
| Q07   |   89.56  |  25.29  |      3.54      |
| Q08   |  136.45  |  29.43  |      4.64      |
| Q09   |  626.76  |  43.31  |      14.47     |
| Q10   |  379.44  |  36.13  |      10.50     |
| Q11   |   33.73  |   5.10  |      6.61      |
| Q12   |  174.45  |  13.30  |      13.12     |
| Q13   |   26.47  |  13.05  |      2.03      |
| Q14   |   27.46  |  11.97  |      2.30      |
| Q15   |   3.62   |  11.96  |      0.30      |
| Q16   |  125.70  |  13.42  |      9.37      |
| Q17   |  266.15  |  49.25  |      5.40      |
| Q18   |  485.49  |  17.75  |      27.36     |
| Q19   |   54.10  |   9.95  |      5.44      |
| Q20   |  120.96  |  15.38  |      7.87      |
| Q22   |   32.37  |  17.76  |      1.82      |
| Total | 3,136.36 |  426.74 |      7.35      |

## SF100:  Xorbits vs. Pandas API on Spark
Spark is a well-known framework for fast, large-scale data processing. The graph below illustrates
the computing times of Xorbits and Spark Pandas API for the TPC-H queries (excluding I/O). Across
all queries, the two systems have roughly similar performance, but Xorbits provided much better API
compatibility. Spark Pandas API failed on Q1, Q4, Q7, Q21, and ran out of memory on Q20.

![image info](https://xorbits.io/res/benchmark_spark.png)

|       | Spark Pandas API | Xorbits | Speed up ratio |
|-------|:----------------:|:-------:|:--------------:|
| Q02   |       18.02      |   8.02  |      2.25      |
| Q03   |       13.96      |  37.27  |      0.37      |
| Q05   |       16.00      |  29.04  |      0.55      |
| Q06   |      134.00      |   8.83  |      15.17     |
| Q08   |       24.71      |  29.43  |      0.84      |
| Q09   |       27.15      |  43.31  |      0.63      |
| Q10   |       10.29      |  36.13  |      0.28      |
| Q11   |       5.82       |   5.10  |      1.14      |
| Q13   |       29.23      |  13.05  |      2.24      |
| Q14   |       17.12      |  11.97  |      1.43      |
| Q15   |       14.26      |  11.96  |      1.19      |
| Q16   |       5.86       |  13.42  |      0.44      |
| Q17   |       22.64      |  49.25  |      0.46      |
| Q18   |       9.51       |  17.75  |      0.54      |
| Q20   |       20.29      |  15.38  |      1.32      |
| Q21   |       64.18      |  78.80  |      0.81      |
| Q22   |       23.98      |  17.76  |      1.35      |
| Total |      457.03      |  426.44 |      1.07      |

## SF100:  Xorbits vs. Modin
Modin is another "Pandas-like" framework that claims to scale Pandas by "changing a single line of
code." The graph below illustrates the computing times of Xorbits and Modin for the TPC-H queries
(excluding I/O). Since Modin hanged at the first query, we tried running queries individually to 
lower the memory usage. However, Modin still ran out of memory for most of the queries that involve
heavy data shuffles, making the performance difference less obvious. Xorbits was still found to be
3.2x faster than Modin.

![image info](https://xorbits.io/res/benchmark_modin.png)

|       |  Xorbits  | Modin on Ray | Speed up ratio |
|:-----:|:---------:|:------------:|:--------------:|
|  Q06  |    8.83   |     28.92    |      3.27      |
|  Q11  |    5.10   |     4.86     |      0.95      |
|  Q13  |   13.05   |     83.74    |      6.41      |
|  Q15  |   11.96   |     79.10    |      6.61      |
|  Q16  |   13.42   |     27.62    |      2.05      |
|  Q17  |   49.25   |    127.02    |      2.57      |
|  Q22  |   17.76   |     33.13    |      1.86      |
| Total |   119.37  |    384.39    |      3.22      |

## Run

### Environment Setup

Each folder contains a `deploy.md` file which descripes how to set up the environment via Docker container. For distributed engines like Dask, Ray, Spark, and Xorbits, we need an cluster endpoint that the client can connect with.

### Run the queries

Note that we should run the queries **under** the `tpch` folder while using the `-m` flags to launch a specific query. 

Here are some arguments that we can specify when launching the query scirpt:

* `--path`: the path of the TPC-H datasets. If the `path` is on S3, we should also specify `storage_options` that contains AWS accounts and keys.

* `--queries`: the queries that we want to run. Query number should be seperated by whitespace, for example: `--queries 1 2`. If `queries` is not specified, all the 22 queries will be executed.

* `--log_time`: log the execution time.

* `--print_result`: print the query result and save into files.

For example, lanching the pandas script should be like:

```
python -u -m pandas_queries.queries \
    --path /path/to/tpch/SF10 \
    --log_time \
    --print_result \
    --queries 1 \
```