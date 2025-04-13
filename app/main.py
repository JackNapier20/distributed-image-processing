from pyspark.sql import SparkSession
import cv2
import numpy as np
import time
import socket

def upload_images_to_hdfs():
    """Upload real cat and dog images using PySpark's Java HDFS client"""
    import os
    from pyspark.sql import SparkSession
    
    print("Starting image upload to HDFS...")
    
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("HDFSUploader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    # Get the Hadoop configuration
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Create a filesystem object
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    # Create Path objects for cats and dogs directories
    hdfs_base_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images")
    hdfs_cats_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images/cats")
    hdfs_dogs_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images/dogs")
    
    # Create directories in HDFS if they don't exist
    if not fs.exists(hdfs_base_dir):
        fs.mkdirs(hdfs_base_dir)
        print("Created directory /data/images in HDFS")
    
    if not fs.exists(hdfs_cats_dir):
        fs.mkdirs(hdfs_cats_dir)
        print("Created directory /data/images/cats in HDFS")
        
    if not fs.exists(hdfs_dogs_dir):
        fs.mkdirs(hdfs_dogs_dir)
        print("Created directory /data/images/dogs in HDFS")
    
    # Path to local images in the container
    # This should be the path where your Kaggle dataset is mounted
    dataset_dir = "/local_images"  # Adjust this path as needed
    
    if os.path.exists(dataset_dir):
        # Count uploaded files
        cat_count = 0
        dog_count = 0
        
        # Process all image files
        for filename in os.listdir(dataset_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            local_path = os.path.join(dataset_dir, filename)
            
            # Determine if it's a cat or dog image based on filename
            if filename.startswith('cat'):
                hdfs_path = f"/data/images/cats/{filename}"
                category = "cat"
                cat_count += 1
            elif filename.startswith('dog'):
                hdfs_path = f"/data/images/dogs/{filename}"
                category = "dog"
                dog_count += 1
            else:
                # Skip files that don't match the expected pattern
                continue
            
            # Create path objects
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_path}")
            hdfs_path_obj = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
            
            # Upload file to HDFS
            fs.copyFromLocalFile(False, True, local_path_obj, hdfs_path_obj)
            
            # Print progress every 100 files
            if (cat_count + dog_count) % 100 == 0:
                print(f"Uploaded {cat_count + dog_count} images so far ({cat_count} cats, {dog_count} dogs)")
        
        print(f"Upload complete. Total: {cat_count + dog_count} images ({cat_count} cats, {dog_count} dogs)")
        
        # Verify upload by counting files in HDFS
        cat_files = fs.listStatus(hdfs_cats_dir)
        dog_files = fs.listStatus(hdfs_dogs_dir)
        
        print(f"Files in HDFS: {len(cat_files)} cats, {len(dog_files)} dogs")
    else:
        print(f"Error: Dataset directory {dataset_dir} not found")

    """Upload real cat and dog images using PySpark's Java HDFS client"""

    
    print("Starting image upload to HDFS...")
    
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("HDFSUploader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    # Get the Hadoop configuration
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Create a filesystem object
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    # Create Path objects for cats and dogs directories
    hdfs_base_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images")
    hdfs_cats_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images/cats")
    hdfs_dogs_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images/dogs")
    
    # Create directories in HDFS if they don't exist
    if not fs.exists(hdfs_base_dir):
        fs.mkdirs(hdfs_base_dir)
        print("Created directory /data/images in HDFS")
    
    if not fs.exists(hdfs_cats_dir):
        fs.mkdirs(hdfs_cats_dir)
        print("Created directory /data/images/cats in HDFS")
        
    if not fs.exists(hdfs_dogs_dir):
        fs.mkdirs(hdfs_dogs_dir)
        print("Created directory /data/images/dogs in HDFS")
    
    # Path to local images in the container
    # This should be the path where your Kaggle dataset is mounted
    dataset_dir = "/local_images"  # Adjust this path as needed
    
    if os.path.exists(dataset_dir):
        # Count uploaded files
        cat_count = 0
        dog_count = 0
        
        # Process all image files
        for filename in os.listdir(dataset_dir):
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
                
            local_path = os.path.join(dataset_dir, filename)
            
            # Determine if it's a cat or dog image based on filename
            if filename.startswith('cat'):
                hdfs_path = f"/data/images/cats/{filename}"
                category = "cat"
                cat_count += 1
            elif filename.startswith('dog'):
                hdfs_path = f"/data/images/dogs/{filename}"
                category = "dog"
                dog_count += 1
            else:
                # Skip files that don't match the expected pattern
                continue
            
            # Create path objects
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_path}")
            hdfs_path_obj = spark._jvm.org.apache.hadoop.fs.Path(hdfs_path)
            
            # Upload file to HDFS
            fs.copyFromLocalFile(False, True, local_path_obj, hdfs_path_obj)
            
            # Print progress every 100 files
            if (cat_count + dog_count) % 100 == 0:
                print(f"Uploaded {cat_count + dog_count} images so far ({cat_count} cats, {dog_count} dogs)")
        
        print(f"Upload complete. Total: {cat_count + dog_count} images ({cat_count} cats, {dog_count} dogs)")
        
        # Verify upload by counting files in HDFS
        cat_files = fs.listStatus(hdfs_cats_dir)
        dog_files = fs.listStatus(hdfs_dogs_dir)
        
        print(f"Files in HDFS: {len(cat_files)} cats, {len(dog_files)} dogs")
    else:
        print(f"Error: Dataset directory {dataset_dir} not found")

    """Upload images using PySpark's Java HDFS client"""
    
    
    print("Starting image upload to HDFS...")
    
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("HDFSUploader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    # Get the Hadoop configuration
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Create a filesystem object
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    # Create Path objects
    hdfs_dir = spark._jvm.org.apache.hadoop.fs.Path("/data/images")
    
    # Create directory in HDFS if it doesn't exist
    if not fs.exists(hdfs_dir):
        fs.mkdirs(hdfs_dir)
        print("Created directory /data/images in HDFS")
    
    # Create a test image
    print("Creating test image...")
    import numpy as np
    import cv2
    
    # Create a simple test image (black and white checkerboard)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 56):
        for j in range(0, 224, 56):
            if (i//56 + j//56) % 2 == 0:
                test_img[i:i+56, j:j+56] = 255
    
    # Save locally
    test_img_path = "/app/test_cat.jpg"
    cv2.imwrite(test_img_path, test_img)
    
    # Upload to HDFS
    print(f"Uploading test image to HDFS...")
    local_path = spark._jvm.org.apache.hadoop.fs.Path(f"file://{test_img_path}")
    hdfs_path = spark._jvm.org.apache.hadoop.fs.Path("/data/images/test_cat.jpg")
    fs.copyFromLocalFile(False, True, local_path, hdfs_path)
    
    # Verify upload
    print("Verifying uploaded files:")
    status = fs.listStatus(hdfs_dir)
    for fileStatus in status:
        print(f"{fileStatus.getPath().getName()}: {fileStatus.getLen()} bytes")

    """Upload images from namenode container to HDFS"""
    import subprocess
    import os
    
    print("Starting image upload to HDFS...")
    
    # Create directory in HDFS
    subprocess.run(["hadoop", "fs", "-mkdir", "-p", "/data/images"])
    
    # Path to images in namenode container
    # Note: This won't work unless you have access to namenode's filesystem
    # You'll need to copy the images first or use a shared volume
    
    # Alternative: Use a test image if you can't access the namenode's files
    print("Creating test image...")
    import numpy as np
    import cv2
    
    # Create a simple test image (black and white checkerboard)
    test_img = np.zeros((224, 224, 3), dtype=np.uint8)
    for i in range(0, 224, 56):
        for j in range(0, 224, 56):
            if (i//56 + j//56) % 2 == 0:
                test_img[i:i+56, j:j+56] = 255
    
    # Save locally
    test_img_path = "/app/test_cat.jpg"
    cv2.imwrite(test_img_path, test_img)
    
    # Upload to HDFS
    print("Uploading test image to HDFS...")
    subprocess.run(["hadoop", "fs", "-put", test_img_path, "/data/images/test_cat.jpg"])
    
    # Verify upload
    print("Verifying uploaded files:")
    subprocess.run(["hadoop", "fs", "-ls", "/data/images"])

    import os
    local_images_dir = "/local_images"
    print(f"Checking directory: {local_images_dir}")
    print(f"Directory exists: {os.path.exists(local_images_dir)}")
    if os.path.exists(local_images_dir):
        print(f"Directory contents: {os.listdir(local_images_dir)}")

    from pyspark.sql import SparkSession
    
    print("Starting image upload to HDFS...")
    
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("HDFSUploader") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    # Get the Hadoop configuration
    hadoop_conf = spark._jsc.hadoopConfiguration()
    
    # Create a filesystem object
    fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    # Create Path objects
    hdfs_dir = spark._jvm.org.apache.hadoop.fs.Path("/local_images")
    
    # Create directory in HDFS if it doesn't exist
    if not fs.exists(hdfs_dir):
        fs.mkdirs(hdfs_dir)
    
    # Path to local images in the container
    local_images_dir = "/local_images"
    
    # Check if directory exists and has files
    if os.path.exists(local_images_dir):
        # Get list of image files
        image_files = [f for f in os.listdir(local_images_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"Found {len(image_files)} images to upload")
        
        # Upload each image to HDFS
        for filename in image_files:
            local_path = os.path.join(local_images_dir, filename)
            hdfs_path = spark._jvm.org.apache.hadoop.fs.Path(f"/local_images/{filename}")
            
            # Create local path object
            local_path_obj = spark._jvm.org.apache.hadoop.fs.Path(f"file://{local_path}")
            
            print(f"Uploading {filename} to HDFS...")
            fs.copyFromLocalFile(False, True, local_path_obj, hdfs_path)
        
        # Verify files were uploaded
        print("Verifying uploaded files:")
        status = fs.listStatus(hdfs_dir)
        for fileStatus in status:
            print(f"{fileStatus.getPath().getName()}: {fileStatus.getLen()} bytes")
    else:
        print(f"Error: Local images directory {local_images_dir} not found")



def wait_for_namenode(host='namenode', port=9000, timeout=2):
    while True:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                print("Namenode is ready!")
                break
        except OSError:
            print("Waiting for namenode to be ready...")
            time.sleep(2)

# Wait for namenode to be ready
wait_for_namenode()

# Upload images from local filesystem to HDFS
upload_images_to_hdfs()

# Start Spark
spark = SparkSession.builder \
    .appName("ImageSanityCheck") \
    .master("spark://spark-master:7077") \
    .getOrCreate()

sc = spark.sparkContext

# Now load the images from HDFS
image_rdd = sc.binaryFiles("hdfs://namenode:9000/data/images/*")
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
