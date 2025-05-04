from pyspark.sql import SparkSession
import cv2
import numpy as np
import time
import socket
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from cnn import CNN


def wait_for_namenode(host='namenode', port=9000, timeout=2):
    """Wait until the namenode is ready and accessible"""
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                print("✅ Namenode is ready!")
                break
        except OSError:
            print("⌛ Waiting for namenode to be ready...")
            time.sleep(2)


def upload_images_to_hdfs():
    """Upload all images from /local_images to HDFS /data/images/ (no subfolders)"""
    print("Starting image upload to HDFS...")
    
    spark = SparkSession.builder.appName("HDFSUploader").getOrCreate()
    hadoop_conf = spark._jsc.hadoopConfiguration()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    hdfs_base_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images")
    
    # Create /data/images in HDFS if it doesn't exist
    if not fs.exists(hdfs_base_dir):
        fs.mkdirs(hdfs_base_dir)
        print("Created directory /data/images in HDFS")
    
    dataset_dir = "/local_images"
    print(f"Checking directory: {dataset_dir}")
    print(f"Directory exists: {os.path.exists(dataset_dir)}")
    if os.path.exists(dataset_dir):
        print(f"Directory contents: {os.listdir(dataset_dir)}")
        count = 0
        for filename in os.listdir(dataset_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            local_path = os.path.join(dataset_dir, filename)
            hdfs_path = f"/data/images/{filename}"
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_path}")
            hdfs_path_obj = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
            fs.copyFromLocalFile(False, True, local_path_obj, hdfs_path_obj)
            count += 1
            if count % 100 == 0:
                print(f"Uploaded {count} images so far.")

        print(f"Upload complete. Total: {count} images.")
        # Verify upload
        files = fs.listStatus(hdfs_base_dir)
        print(f"Files in HDFS: {len(files)}")
    else:
        print(f"Error: Dataset directory {dataset_dir} not found")

def preprocess_image(data):
    """Resize to 224x224, convert BGR to RGB, normalize to [0,1]"""
    path, content = data
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Failed to decode image: {path}")
        return (path, None)
    img_resized = cv2.resize(img, (224, 224))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb / 255.0
    return (path, img_normalized.tolist())

def main():
    wait_for_namenode()
    upload_images_to_hdfs()

    spark = SparkSession.builder.appName("ImagePreprocessing").getOrCreate()
    sc = spark.sparkContext

    # Load all images directly from /data/images/*
    hdfs_path = "hdfs://namenode:9000/data/images/*"
    image_rdd = sc.binaryFiles(hdfs_path)
    count = image_rdd.count()
    print(f"Number of images loaded from HDFS: {count}")

    preprocessed = image_rdd.map(preprocess_image)
    # Print shape of first 5 processed images to verify
    for path, arr in preprocessed.take(5):
        arr_np = np.array(arr) if arr is not None else None
        print(f"{os.path.basename(path)}: shape {arr_np.shape if arr_np is not None else arr_np}")

    print("Preprocessing complete.")

    # Loading both models
    class_names = ["cat", "dog"]

    cnn_model = CNN()
    cnn_model.eval()

    # ResNet50
    resnet = models.resnet50(pretrained=True)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Run predictions with both models
    print("\\n Running both CNN and ResNet50 classifiers \\n")
    for path, arr in preprocessed.take(10):
        if arr is None:
            continue
        try:
            arr_np = np.array(arr, dtype=np.float32)
            input_tensor = torch.tensor(arr_np).permute(2, 0, 1).unsqueeze(0)  # shape = [1, 3, 224, 224]

            # Simple CNN
            with torch.no_grad():
                cnn_out = cnn_model(input_tensor)
                cnn_pred = class_names[torch.argmax(cnn_out).item()]

            # ResNet50
            # input_norm = transform(torch.tensor(arr_np).permute(2, 0, 1)).unsqueeze(0)
            arr_uint8 = (arr_np * 255).astype(np.uint8)  # Convert back from float32 to uint8
            pil_image = Image.fromarray(arr_uint8)  # Shape: [224, 224, 3]
            input_norm = transform(pil_image).unsqueeze(0)
            with torch.no_grad():
                resnet_out = resnet(input_norm)
                resnet_pred = class_names[torch.argmax(resnet_out).item()]

            print(f"{os.path.basename(path)}: CNN={cnn_pred}, ResNet50={resnet_pred}")
        except Exception as e:
            print(f"Error processing {path}: {e}")


if __name__ == "__main__":
    main()
