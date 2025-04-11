# from pyspark.sql import SparkSession
# import cv2
# import numpy as np

# spark = SparkSession.builder \
#     .appName("ImageClassifier") \
#     .master("spark://spark-master:7077") \
#     .getOrCreate()

# sc = spark.sparkContext

# # Load images from mounted volume (or HDFS later)
# image_rdd = sc.binaryFiles("file:///app/images/*")

# def decode_image(data):
#     file_name, content = data
#     img_array = np.frombuffer(content, np.uint8)
#     img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
#     return file_name, img.shape if img is not None else (file_name, "Error")

# results = image_rdd.map(decode_image).collect()

# for file, shape in results:
#     print(f"{file}: {shape}")


from pyspark.sql import SparkSession
import cv2
import numpy as np

# Start Spark
spark = SparkSession.builder \
    .appName("ImageSanityCheck") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

sc = spark.sparkContext

# Load binary files
image_rdd = sc.binaryFiles("file:///app/images/*")

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