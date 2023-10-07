# Start a Dask scheduler
```shell
docker run -d \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    -e AWS_DEFAULT_OUTPUT=${AWS_DEFAULT_OUTPUT} \
    --network host --name dask_scheduler \
    ghcr.io/dask/dask:"${TAG}" dask-scheduler

docker exec dask_scheduler pip install pyarrow==8.0.0 s3fs
```

# Start a Dask worker
```shell
docker run -d \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    -e AWS_DEFAULT_OUTPUT=${AWS_DEFAULT_OUTPUT} \
    --network host --name dask_worker \
    ghcr.io/dask/dask:"${TAG}" dask-worker ${SCHEDULER_HOSTNAME}:8786 --nworkers auto

docker exec dask_worker pip install pyarrow==8.0.0 s3fs
```
