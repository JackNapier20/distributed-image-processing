import argparse, os, socket, tarfile, time
from io import BytesIO
import cv2, numpy as np, torch, torch.nn as nn
from PIL import Image
from pyspark.sql import SparkSession
from torchvision import models, transforms
from cnn import CNN

#spark driver memory
os.environ["SPARK_DRIVER_MEMORY"] = "2g"  

#spark execution memory
os.environ["SPARK_EXECUTOR_MEMORY"] = "2g"     

#Waiting until namenode is up and running
def waitForNamenode(host="namenode", port=9000, timeout=2):
    while True:
        try:
            socket.create_connection((host, port), timeout=timeout).close()
            print("Namenode is up and running!")
            return
        except OSError:
            print("Waiting for namenode to start!")
            time.sleep(2)

#copying the local tar file to HDFS
def uploadTarToHDFS(local_tar, hdfs_tar, spark):
    fs   = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
               spark._jsc.hadoopConfiguration())
    lpth = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_tar}")
    hpth = spark._jvm.org.apache.hadoop.fs.Path(hdfs_tar)
    if fs.exists(hpth):
        return
    parent = hpth.getParent()
    if not fs.exists(parent):
        fs.mkdirs(parent)
    fs.copyFromLocalFile(False, True, lpth, hpth)
    print(f"TAR file is uploaded to {hdfs_tar}")

def extractFromTar(tar_bytes):
    with tarfile.open(fileobj=BytesIO(tar_bytes)) as tf:
        for m in tf.getmembers():
            if m.isfile():
                yield m.name, tf.extractfile(m).read()

#prerocessing the images - resize, normalize and convert to RGB
def preprocess(item):
    path, content = item
    arr = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return path, None
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return path, (img.astype(np.float32)/255.0).tolist()

#Grouping into fixed size batches
def batch_rdd(rdd, batch_sz):
    def _chunker(iterator):
        buf = []
        for x in iterator:
            buf.append(x)
            if len(buf) == batch_sz:
                yield buf; buf = []
        if buf:
            yield buf
    return rdd.mapPartitions(_chunker)

#Function to run in batches
def run_batch(batch, bc_cnn, bc_resnet, tfm):
    cnn    = CNN();          cnn.load_state_dict(bc_cnn.value);    cnn.eval()
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(bc_resnet.value);                       resnet.eval()

    c_in, r_in, paths = [], [], []
    for path, arr in batch:
        if arr is None:
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

#Function for inference
def infer_partition(batches, bc_cnn, bc_resnet, tfm):
    #this runs exactly once per worker
    cnn    = CNN();    cnn.load_state_dict(bc_cnn.value);    cnn.eval()
    resnet = models.resnet50(weights=None)
    resnet.fc = nn.Linear(resnet.fc.in_features, 2)
    resnet.load_state_dict(bc_resnet.value);                resnet.eval()

    for batch in batches:
        c_in, r_in, paths = [], [], []
        for path, arr in batch:
            if arr is None: continue
            img_np = np.asarray(arr, np.float32)
            c_in.append(torch.from_numpy(img_np).permute(2,0,1))
            r_in.append(tfm(Image.fromarray((img_np*255).astype(np.uint8))))
            paths.append(path)

        if not paths:
            continue

        with torch.no_grad():
            c_pred = torch.argmax(cnn(torch.stack(c_in)),    1).tolist()
            r_pred = torch.argmax(resnet(torch.stack(r_in)), 1).tolist()

        for p, cr in zip(paths, zip(c_pred, r_pred)):
            yield (p, cr)


def main():
    argp = argparse.ArgumentParser()
    argp.add_argument("--local-tar",     default="/local_images/catsdogs.tar")
    argp.add_argument("--hdfs-tar",      default="hdfs://namenode:9000/datasets/catsdogs/catsdogs.tar")
    argp.add_argument("--partitions",    type=int, default=8)
    argp.add_argument("--batch-size",    type=int, default=8)
    argp.add_argument("--cnn-weights",   default="cnn_mac_v2.pth")
    argp.add_argument("--resnet-weights",default="resnet50_ft.pth")
    args = argp.parse_args()

    waitForNamenode()
    spark = (
        SparkSession.builder
        .appName("CatsVsDogsInference")
        .master("spark://spark-master:7077")
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000")
        .config("spark.driver.bindAddress", "0.0.0.0")
        .config("spark.driver.host",        "image-app")
        .config("spark.driver.port",        "7079")
        .config("spark.python.worker.reuse","true")
        # .config("spark.executor.cores", 2)
        # .config("spark.executor.memory", "4g")
        # .config("spark.default.parallelism", 12)
        # .config("spark.sql.shuffle.partitions", 12)
        .getOrCreate()
    )

    uploadTarToHDFS(args.local_tar, args.hdfs_tar, spark)
    sc = spark.sparkContext
    sc.addPyFile("/app/cnn.py")

    #Loading tar into rdd
    raw = (sc.binaryFiles(args.hdfs_tar)
             .flatMap(lambda kv: extractFromTar(kv[1])))
    total = raw.count()
    print(f"Loaded {total} images from hdfs TAR")
    if args.partitions:
        raw = raw.repartition(args.partitions)
        print(f"Repartitioned into {args.partitions} slices")

    # applying the models
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


    batched = batch_rdd(raw.map(preprocess)
                           .filter(lambda x: x[1] is not None),
                        args.batch_size)
    print(f"Total number of batches: {batched.map(lambda _:1).sum()}")

    #inference
    inf = batched.mapPartitions(lambda it: infer_partition(it, bc_cnn, bc_resnet, tfm))

    #Calculating time
    t0 = time.time()
    pred_count = inf.count()
    elapsed = time.time() - t0
    thr = total / elapsed if elapsed else float("inf")
    sample = inf.take(10)

    #metrics
    print("BENCHMARK METRICS : ")
    print("images,partitions | batch_size | time(sec)| throughput(image/sec)")
    print(f"{total} | {args.partitions} | {args.batch_size} | {elapsed:.2f},{thr:.1f}")
    names = ["cat", "dog"]
    print("Sample predictions:")
    for p,(c,r) in sample:
        print(f"{os.path.basename(p):<25}  CNN={names[c]},  ResNet50={names[r]}")

if __name__ == "__main__":
    main()
