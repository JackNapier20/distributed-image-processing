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