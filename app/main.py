# main.py  – distributed Cats‑vs‑Dogs inference on Spark
import argparse, os, socket, tarfile, time
from io import BytesIO

import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image
from pyspark.sql import SparkSession
from torchvision import models, transforms
from cnn import CNN          # your custom small‑CNN

os.environ["SPARK_DRIVER_MEMORY"] = "4g"       # driver JVM heap
os.environ["SPARK_EXECUTOR_MEMORY"] = "3g"     # each executor heap

# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def wait_for_namenode(host="namenode", port=9000, timeout=2):
    """Block until HDFS namenode answers on <host>:<port>."""
    while True:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            print("Namenode is up and running!")
            return
        except OSError:
            print("Waiting for namenode …")
            time.sleep(2)

def upload_tar(local_tar, hdfs_tar, spark):
    """Copy <local_tar> (inside the image‑app container) to HDFS <hdfs_tar> if
    it is not already there."""
    fs   = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
               spark._jsc.hadoopConfiguration())
    lpth = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_tar}")
    hpth = spark._jvm.org.apache.hadoop.fs.Path(hdfs_tar)
    if fs.exists(hpth):
        print("TAR already present in HDFS")
        return
    parent = hpth.getParent()
    if not fs.exists(parent):
        fs.mkdirs(parent)
    fs.copyFromLocalFile(False, True, lpth, hpth)
    print(f"TAR is uploaded to {hdfs_tar}")

def extract_from_tar(tar_bytes):
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m).read()

def preprocess(item):
    """JPEG bytes ➜ float [0,1] RGB 224×224 list (keep it serialisable)."""
    path, content = item
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return path, None          # corrupt image
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return path, (img.astype(np.float32) / 255.0).tolist()

def batch_rdd(rdd, batch_sz):
    """Group records coming out of a partition into fixed‑size batches."""
    def _chunker(iterator):
        buf = []
        for x in iterator:
            buf.append(x)
            if len(buf) == batch_sz:
                yield buf; buf = []
        if buf:
            yield buf
    return rdd.mapPartitions(_chunker)

def run_batch(batch, bc_cnn, bc_resnet, tfm):
    """Infer one mini‑batch on the worker; returns (path,(cnn,resnet)) list."""
    cnn    = CNN();          cnn.load_state_dict(bc_cnn.value);    cnn.eval()
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(bc_resnet.value);                       resnet.eval()

    c_in, r_in, paths = [], [], []
    for path, arr in batch:
        if arr is None:                      # corrupt image was skipped
            continue
        img_np  = np.asarray(arr, dtype=np.float32)
        c_in.append(torch.from_numpy(img_np).permute(2, 0, 1))
        r_in.append(tfm(Image.fromarray((img_np * 255).astype(np.uint8))))
        paths.append(path)

    if not paths:
        return []
    with torch.no_grad():
        c_pred = torch.argmax(cnn(torch.stack(c_in)),    1).tolist()
        r_pred = torch.argmax(resnet(torch.stack(r_in)), 1).tolist()
    return list(zip(paths, zip(c_pred, r_pred)))

# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------
def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--local-tar",     default="/local_images/catsdogs.tar")
    argp.add_argument("--hdfs-tar",      default="hdfs://namenode:9000/datasets/catsdogs/catsdogs.tar")
    argp.add_argument("--partitions",    type=int, default=8)
    argp.add_argument("--batch-size",    type=int, default=8)
    argp.add_argument("--cnn-weights",   default="cnn_mac_v2.pth")
    argp.add_argument("--resnet-weights",default="resnet50_ft.pth")
    args = argp.parse_args()

    # 1) wait for HDFS ⇒ copy dataset once
    wait_for_namenode()
    spark = (SparkSession.builder
         .appName("TarUploader")
         .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")  # <─ NEW
         .getOrCreate())

    upload_tar(args.local_tar, args.hdfs_tar, spark)
    sc = spark.sparkContext

    # 2) Load the TAR into an RDD
    raw = (sc.binaryFiles(args.hdfs_tar)
             .flatMap(lambda kv: extract_from_tar(kv[1])))
    total = raw.count()
    print(f"Loaded {total} images from HDFS TAR")
    if args.partitions:
        raw = raw.repartition(args.partitions)
        print(f"Repartitioned into {args.partitions} slices")

    # 3) Broadcast models
    cnn = CNN(); cnn.load_state_dict(torch.load(args.cnn_weights, map_location="cpu")); cnn.eval()
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(torch.load(args.resnet_weights, map_location="cpu")); resnet.eval()
    bc_cnn    = sc.broadcast(cnn.state_dict())
    bc_resnet = sc.broadcast(resnet.state_dict())
    tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],
                                 [0.229,0.224,0.225])
          ])

    # 4) Pipeline ⇒ preprocess ⇒ batch ⇒ inference
    batched = batch_rdd(raw.map(preprocess)
                           .filter(lambda x: x[1] is not None),
                        args.batch_size)

    print(f"Total number of batches: {batched.map(lambda _:1).sum()}")

    inf = (batched
           .map(lambda b: run_batch(b, bc_cnn, bc_resnet, tfm))
           .flatMap(lambda x: x))              # (path,(cnn,resnet))

    # 5) Time a single full pass (action = count)
    t0 = time.time()
    # pred_count = inf.count()
    elapsed = time.time() - t0
    thr = total / elapsed if elapsed else float("inf")

    # 6) small sample back to driver
    sample = inf.take(10)

    # 7) Report
    print("\n=== BENCHMARK METRICS ===")
    print("images,partitions,batch_size,time_s,throughput_img_per_s")
    print(f"{total},{args.partitions},{args.batch_size},{elapsed:.2f},{thr:.1f}")
    print("=========================\n")

    names = ["cat", "dog"]
    print("Sample predictions:")
    for p,(c,r) in sample:
        print(f"{os.path.basename(p):<25}  CNN={names[c]},  ResNet50={names[r]}")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
