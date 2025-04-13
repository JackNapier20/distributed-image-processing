from pyspark.sql import SparkSession
import cv2
import numpy as np
import time

time.sleep(20)

# Start Spark
spark = SparkSession.builder \
    .appName("ImageSanityCheck") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

sc = spark.sparkContext

# Load binary files using directly
# image_rdd = sc.binaryFiles("file:///app/images/*")

# Load binary files using Hadoop
image_rdd = sc.binaryFiles("hdfs://namenode:9000/images/*")
print("Number of files loaded: ", image_rdd.count())

def decode_image(data):
    path, content = data
    try:
        img_array = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        shape = img.shape if img is not None else "decode error"
    except Exception as e:
        shape = str(e)
    return (path, shape)

results = image_rdd.map(decode_image).collect()

for path, shape in results:
    print(f"{path.split('/')[-1]}: {shape}")

input("Press Enter to exit...")