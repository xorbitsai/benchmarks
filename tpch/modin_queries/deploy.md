# Start a Ray head
```shell
docker run -d \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    -e AWS_DEFAULT_OUTPUT=${AWS_DEFAULT_OUTPUT} \
    -e __MODIN_AUTOIMPORT_PANDAS__=1 \
    --network host --name ray_head \
    continuumio/miniconda3:4.12.0 tail -f /dev/null

docker exec ray_head pip install modin[ray] fsspec s3fs pyarrow==8.0.0

docker exec ray_head ray start --head --port=6379
```

# Start a Ray worker
```shell
docker run -d \
    -e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} \
    -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY} \
    -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION} \
    -e AWS_DEFAULT_OUTPUT=${AWS_DEFAULT_OUTPUT} \
    -e __MODIN_AUTOIMPORT_PANDAS__=1 \
    --network host --name ray_worker continuumio/miniconda3:4.12.0 tail -f /dev/null

docker exec ray_worker pip install modin[ray] fsspec s3fs pyarrow==8.0.0

docker exec ray_worker ray start --address=${RAY_HEAD_HOSTNAME}:6379
```