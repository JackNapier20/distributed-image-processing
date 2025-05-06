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
### Testcases :

We tested the above commands for 1 to 3 worker nodes.
After each run we can see 10 different samples of the model output.
```commandline
Namenode is up and running!
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
25/05/06 02:21:55 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
TAR file is uploaded to hdfs://namenode:9000/datasets/catsdogs/catsdogs.tar
Loaded 512 images from hdfs TAR                                                 
Repartitioned into 4 slices
Total number of batches: 35                                                     
BENCHMARK METRICS :                                                             
images,partitions | batch_size | time(sec)| throughput(image/sec)
512 | 4 | 8 | 23.11,22.2
Sample predictions:
112.jpg                    CNN=cat,  ResNet50=dog
113.jpg                    CNN=dog,  ResNet50=dog
114.jpg                    CNN=cat,  ResNet50=dog
115.jpg                    CNN=dog,  ResNet50=dog
116.jpg                    CNN=cat,  ResNet50=dog
130.jpg                    CNN=cat,  ResNet50=dog
131.jpg                    CNN=dog,  ResNet50=dog
132.jpg                    CNN=cat,  ResNet50=dog
133.jpg                    CNN=cat,  ResNet50=dog
134.jpg                    CNN=cat,  ResNet50=dog

```
