# Distributed Image Processing with Spark & Hadoop

This project sets up a distributed image processing pipeline using Docker, Hadoop (HDFS), and Apache Spark. The system allows users to store images in HDFS and run image classification inference jobs via a Spark-based application, with various spark workers.

## Contributers 
- [Isha](https://github.com/isha-234)
- [Lavanika](https://github.com/lava-nika)
- [Sri Ram](https://github.com/srirambandi)
- [Sivaraaman](https://github.com/JackNapier20)


## Running the experiments
### Dataset setup
 - Fetch the images from the [dataset](https://www.kaggle.com/c/dogs-vs-cats/data).
 - Move images from the `test1` directory in the dataset to `\images` director of the project.
 - Compress the files to a `tar` file with the following command.
```bash
# tar our image files, so its easy to upload to HDFS
tar -cf images/catsdogs.tar images/*.jpg
```

### Multi-node/Single-node analysis
```bash
# setup the docker containers and images with specified number of spark workers
docker compose up -d --scale spark-worker=1
# run the experiment with specified number of partitions and batch size
docker compose exec image-app python main.py --partitions 8 --batch-size 8
```

