#!/usr/bin/env python3
import argparse
import os
import socket
import time
from itertools import islice

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from pyspark.sql import SparkSession
from torchvision import models
from torchvision.models import ResNet50_Weights

from cnn import CNN


def wait_for_namenode(host='namenode', port=9000, timeout=2):
    """Wait until the namenode is ready and accessible"""
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                print("✅ Namenode is ready!")
                return
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


def batch_rdd(rdd, batch_size):
    # return rdd.mapPartitions(lambda it: (list(islice(it, batch_size)) for _ in iter(int, 1)))
    def chunker(iterator):
        batch = []
        for item in iterator:
            batch.append(item)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:  # Final partial batch
            yield batch

    return rdd.mapPartitions(chunker)

def run_inference_batch(batch, bc_cnn, bc_resnet, transform):
    cnn = CNN(); cnn.load_state_dict(bc_cnn.value); cnn.eval()
    res = models.resnet50(weights=None)
    res.fc = nn.Linear(res.fc.in_features, 2)
    res.load_state_dict(bc_resnet.value)
    res.eval()

    cnn_inputs = []
    resnet_inputs = []
    paths = []

    for path, arr in batch:
        img_np = np.array(arr, dtype=np.float32)
        tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        cnn_inputs.append(tensor)
        pil = Image.fromarray((img_np * 255).astype(np.uint8))
        resnet_inputs.append(transform(pil))
        paths.append(path)

    if not cnn_inputs or not resnet_inputs:
        print("⚠️ Skipping empty batch during inference.")
        return []

    cnn_batch = torch.stack(cnn_inputs)
    resnet_batch = torch.stack(resnet_inputs)

    with torch.no_grad():
        cnn_out = cnn(cnn_batch)
        res_out = res(resnet_batch)
        cnn_preds = torch.argmax(cnn_out, dim=1).tolist()
        res_preds = torch.argmax(res_out, dim=1).tolist()

    return list(zip(paths, zip(cnn_preds, res_preds)))


def main():
   # parse CLI flags for number of partitions and metrics file
    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", type=int, default=None,
                        help="Repartition the RDD into this many slices")
    parser.add_argument("--output-csv", type=str, default=None,
                        help="Append [images, partitions, time, throughput] to this CSV")
    args = parser.parse_args()

    wait_for_namenode()
    upload_images_to_hdfs()

    spark = SparkSession.builder.appName("DistributedInference").getOrCreate()
    sc = spark.sparkContext

    # Load all images directly from /data/images/*
    hdfs_path = "hdfs://namenode:9000/data/images/*"
    image_rdd = sc.binaryFiles(hdfs_path)
    count = image_rdd.count()
    print(f"Number of images loaded from HDFS: {count}")

    preprocessed = image_rdd.map(preprocess_image)

    print("Preprocessing complete.")

    # 3) Load & broadcast models
    cnn_model = CNN(); cnn_model.eval()
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.eval()
    bc_cnn = sc.broadcast(cnn_model.state_dict())
    bc_resnet = sc.broadcast(resnet.state_dict())

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std=[0.229,0.224,0.225])
    ])



    # 4) Build inference RDD
    batched = batch_rdd(preprocessed.filter(lambda x: x[1] is not None), 3)

    # Count and log number of batches
    batch_counts = batched.map(lambda _: 1).reduce(lambda a, b: a + b)
    print(f"📊 Total batches created: {batch_counts}")

    inference_rdd = (
            batched
            .map(lambda batch: run_inference_batch(batch, bc_cnn, bc_resnet, transform))
            .flatMap(lambda x: x)
        )

    # warm-up
    inference_rdd.count()

    # timed run
    start = time.time()
    results = inference_rdd.collect()
    elapsed = time.time() - start
    done = len(results)
    tput = done / elapsed if elapsed > 0 else float("inf")

    print(f"[Benchmark] images={done}, partitions={args.partitions or 'default'}, "
          f"time={elapsed:.2f}s, throughput={tput:.1f} img/s")

    # meytrics
    print("\n=== BENCHMARK METRICS ===")
    print("images,partitions,time_s,throughput_img_per_s")
    print(f"{done},{args.partitions or 'default'},{elapsed:.2f},{tput:.1f}")
    print("=========================\n")

    # sample predictions
    class_names = ["cat", "dog"]
    print("\nSample predictions:")
    for path, (c_pred, r_pred) in results[:10]:
        fn = os.path.basename(path)
        print(f"  {fn} → CNN={class_names[c_pred]}, ResNet50={class_names[r_pred]}")


if __name__ == "__main__":
    main()
