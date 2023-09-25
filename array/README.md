# Array Benchmarks

We compared Xorbits to Dask. Workloads including matrix multiplication, QR decomposition, linear regression, etc.

## Run

To run a specific workload, you can specify `--workloads` and `--size` parameters.

```python
python workloads.py --endpoint ${address} \
    --workloads matmul \
    --size xl
```