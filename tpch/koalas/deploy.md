# Start Spark master

```shell
docker run --privileged \
           --env SPARK_MASTER_HOST="$master_host" \
           -d \
           --network host \
           --name spark-master xprobe/spark:spark-master-3.3.1-3 tail -f /dev/null

docker exec -d spark-master bin/spark-class org.apache.spark.deploy.master.Master
```

# Start Spark worker

```shell
docker run --privileged \
           -d --network host \
           --name spark-worker xprobe/spark:spark-worker-3.3.1-3 tail -f /dev/null
           
docker exec -d spark-worker bin/spark-class org.apache.spark.deploy.worker.Worker "spark://$master_host:7077"
```