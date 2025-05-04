# Distributed Image Processing with Spark & Hadoop

This project sets up a distributed image processing pipeline using Docker, Hadoop (HDFS), and Apache Spark. The system allows users to store images in HDFS and run image-processing jobs via a Spark-based application.

## 🐳 Getting Started

### 1. Build and Start the Docker Containers

Ensure Docker and Docker Compose are installed. Then run:

```bash
docker-compose up --build
```

This will start all necessary services including:
	•	Hadoop NameNode and DataNode
	•	Spark Master and Workers
	•	image-app (custom Spark-based image processor)

# Access the NameNode container
docker exec -it namenode bash

# Inside the container, create the target directory in HDFS
hadoop fs -mkdir -p /data/images

# Put local images (assuming they're mounted to /local_images) into HDFS
hadoop fs -put /local_images/* /data/images/

# Verify upload
hadoop fs -ls /data/images

# Start inage app
docker exec -it image-app

# Multi-node/Single-node analysis
```bash
# full commands
# build and start in detached mode
docker-compose up --build -d

# stop image app
docker-compose stop image-app

# init namenode
docker exec -it namenode bash
# and insde the shell:
hadoop fs -mkdir -p /data/images
hadoop fs -put /local_images/* /data/images/
hadoop fs -ls /data/images    # you should see your 21 cat/dog images
exit

# start image-app with params
docker-compose up --no-deps --build -d image-app

# run main.py with 4 partitions(can't find the metrics file now)
docker-compose exec image-app python main.py --partitions 4 --output-csv metrics.csv

# scale up to 3 workers(to be verified)
docker-compose up --no-deps -d --scale spark-worker=3

# again run the main.py
docker-compose exec image-app python main.py --partitions 12 --output-csv metrics.csv
```