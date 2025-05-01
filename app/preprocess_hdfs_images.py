from pyspark.sql import SparkSession
import cv2
import numpy as np

def preprocess_image(data):
    path, content = data
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Could not decode image at {path}")
        return (path, None)
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    return (path, img_normalized.tolist())

def main():
    spark = SparkSession.builder.appName("HDFSImagePreprocessing").getOrCreate()
    sc = spark.sparkContext

    # Adjust the path below to match your actual HDFS image location!
    hdfs_path = "hdfs://namenode:9000/data/images/*"

    # Check if files exist at the path before proceeding
    try:
        image_rdd = sc.binaryFiles(hdfs_path)
        count = image_rdd.count()
        if count == 0:
            print(f"No files found at {hdfs_path}. Exiting.")
            return
        print(f"Found {count} images in HDFS. Starting preprocessing...")
    except Exception as e:
        print(f"Error accessing HDFS path {hdfs_path}: {e}")
        return

    preprocessed = image_rdd.map(preprocess_image)
    # Save or inspect preprocessed results
    for path, arr in preprocessed.take(5):
        print(f"{path}: shape {np.array(arr).shape if arr is not None else arr}")

    print("Preprocessing complete.")

if __name__ == "__main__":
    main()
