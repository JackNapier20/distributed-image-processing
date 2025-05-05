# Distributed Image Processing with Spark & Hadoop

This project sets up a distributed image processing pipeline using Docker, Hadoop (HDFS), and Apache Spark. The system allows users to store images in HDFS and run image-processing jobs via a Spark-based application.

## Contributers 
- [Isha](https://github.com/isha-234)
- [Lavanika](https://github.com/lavanika)
- [Sri Ram](https://github.com/srirambandi)
- [Siva](https://github.com/siva)


## Getting Started

### 1. Build and Start the Docker Containers

Ensure Docker and Docker Compose are installed. Then run:

```bash
docker-compose up --build
```

This will start all necessary services including:
	•	Hadoop NameNode and DataNode
	•	Spark Master and Workers
	•	image-app (custom Spark-based image processor)

This will also give an error saying - 
```commandline
 java.io.IOException: Input Pattern hdfs://namenode:9000/data/images/* matches 0 files
```

It just means that the local images have not been loaded to HDFS. We do this in the next step.

On another terminal, run the following commands - 
#### Access the NameNode container
```
docker exec -it namenode bash
```
#### Inside the container, create the target directory in HDFS
```
hadoop fs -mkdir -p /data/images
```
#### Put local images (assuming they are mounted to /local_images) into HDFS

```
hadoop fs -put /local_images/* /data/images/
```

#### Verify upload
```
hadoop fs -ls /data/images

```
In another terminal, run-
#### Start image app
```
docker start -i image-app

```
or you can run docker compose up command in another terminal
```commandline
docker compose up
```
### 2. Multi-node/Single-node analysis

```bash
# full commands
# tar our image files, so its easy to upload to HDFS
tar -cf images/catsdogs.tar images/*.jpg

# build and start in detached mode
docker-compose up --build -d

# stop image app
docker-compose stop image-app

# start image-app with params
docker-compose up --no-deps --build -d image-app

# run main.py with 4 partitions
docker-compose exec image-app python main.py --partitions 4

# scale up to 3 workers(to be verified)
docker-compose up --no-deps -d --scale spark-worker=3

# again run the main.py
docker-compose exec image-app python main.py --partitions 12
```