import argparse
import os
import socket
import tarfile
import time
from io import BytesIO
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


def waitForNamenode(host="namenode", port=9000, timeout=2):
    #Wait until the namenode is up and running
    while True:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            print("Namenode is up and running!")
            return
        except OSError:
            print("Waiting for namenode to be up")
            time.sleep(2)


def uploadToHDFS(
    local_tar="/local_images/catsdogs.tar",
    hdfs_tar="hdfs://namenode:9000/datasets/catsdogs/catsdogs.tar"
):
    #Copy catsdogs.tar into HDFS
    print("Uploading TAR to HDFS!")
    spark = SparkSession.builder.appName("TarUploader").getOrCreate()
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        spark._jsc.hadoopConfiguration()
    )
    local_path = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_tar}")
    hdfs_path  = spark._jvm.org.apache.hadoop.fs.Path(hdfs_tar)
    parent     = hdfs_path.getParent()
    if not fs.exists(parent):
        fs.mkdirs(parent)
    fs.copyFromLocalFile(False, True, local_path, hdfs_path)
    print(f"TAR is uploaded to {hdfs_tar}")


def extractFromTar(tar_bytes):
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m).read()


#Decoding JPEG bytes, resize to 224×224,convert BGR to RGB and normalize to [0,1]
def preprocess_image(item):
    path, content = item
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return path, None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return path, (img.astype(np.float32) / 255.0).tolist()

#Grouping an RDD into batches of size(batch_size)
def batch_rdd(rdd, batch_size):
    def chunker(it):
        buf = []
        for x in it:
            buf.append(x)
            if len(buf) == batch_size:
                yield buf
                buf = []
        if buf:
            yield buf
    return rdd.mapPartitions(chunker)

#Run batched CNN and ResNet inference on one partition's batch
def run_inference_batch(batch, bc_cnn, bc_resnet, transform):
    cnn = CNN(); cnn.load_state_dict(bc_cnn.value); cnn.eval()
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(bc_resnet.value); resnet.eval()

    cnn_inputs, res_inputs, paths = [], [], []
    for path, arr in batch:
        if arr is None:
            continue
        img_np = np.array(arr, dtype=np.float32)
        cnn_inputs.append(torch.from_numpy(img_np).permute(2, 0, 1))
        res_inputs.append(transform(Image.fromarray((img_np*255).astype(np.uint8))))
        paths.append(path)

    if not paths:
        return []
    cnn_batch = torch.stack(cnn_inputs)
    res_batch = torch.stack(res_inputs)
    with torch.no_grad():
        c_out = cnn(cnn_batch)
        r_out = resnet(res_batch)
        c_preds = torch.argmax(c_out, 1).tolist()
        r_preds = torch.argmax(r_out, 1).tolist()

    return list(zip(paths, zip(c_preds, r_preds)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images per inference batch")
    parser.add_argument("--partitions", type=int, default=None,
                        help="Repartition RDD into this many slices")
    parser.add_argument("--cnn-weights", default="cnn_mac_v2.pth",
                        help="Path to your trained CNN weights (in /app/)")
    parser.add_argument("--resnet-weights", default="resnet50_ft.pth",
                        help="Path to your fine-tuned ResNet50 weights (in /app/)")
    args = parser.parse_args()

    waitForNamenode()
    uploadToHDFS()

    spark = SparkSession.builder.appName("DistributedInference").getOrCreate()
    sc = spark.sparkContext

    #read the TAR from HDFS and expand it
    raw = (sc.binaryFiles("hdfs://namenode:9000/datasets/catsdogs/catsdogs.tar")
             .flatMap(lambda kv: extractFromTar(kv[1])))
    total_images = raw.count()
    print(f"📥 Loaded {total_images} images from HDFS TAR")

    #optionally repartition
    if args.partitions:
        raw = raw.repartition(args.partitions)
        print(f"🔀 Repartitioned into {args.partitions} slices")

    #preprocessing
    pre = raw.map(preprocess_image)

    #loading & broadcasting CNN and resnet models
    cnn = CNN(); cnn.load_state_dict(torch.load(args.cnn_weights)); cnn.eval()
    resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load(args.resnet_weights)); resnet.eval()
    bc_cnn = sc.broadcast(cnn.state_dict())
    bc_resnet = sc.broadcast(resnet.state_dict())
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    #batch and inference
    batched = batch_rdd(pre.filter(lambda x: x[1] is not None), args.batch_size)
    batch_count = batched.map(lambda _:1).reduce(lambda a,b: a+b)
    print(f"Total number of batches: {batch_count}")
    inf = (batched
           .map(lambda b: run_inference_batch(b, bc_cnn, bc_resnet, transform))
           .flatMap(lambda x: x))
    inf.count()
    t0 = time.time()
    results = inf.collect()
    total_time = time.time() - t0
    M = len(results)
    throughput = M / total_time if total_time > 0 else float("inf")

    #printing metrics
    print("\n=== BENCHMARK METRICS ===")
    print("images,partitions,batch_size,time_s,throughput_img_per_s")
    print(f"{M},{args.partitions or 'default'},{args.batch_size},"
          f"{total_time:.2f},{throughput:.1f}")
    print("=========================\n")

    #Sample predictions
    names = ["cat", "dog"]
    print("Sample predictions:")
    for path, (c, r) in results[:10]:
        fn = os.path.basename(path)
        print(f"  {fn} : CNN={names[c]}, ResNet50={names[r]}")


if __name__ == "__main__":
    main()
