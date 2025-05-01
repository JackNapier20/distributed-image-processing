from pyspark.sql import SparkSession
import cv2
import numpy as np
import time
import socket
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

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
    dataset_dir = "/local_images"
    
    # Debug information
    print(f"Checking directory: {dataset_dir}")
    print(f"Directory exists: {os.path.exists(dataset_dir)}")
    if os.path.exists(dataset_dir):
        print(f"Directory contents: {os.listdir(dataset_dir)}")
    
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

def preprocess_image(image_data):
    """
    Preprocess image data for CNN model input
    - Resize to 224x224 (standard for many CNNs)
    - Convert BGR to RGB
    - Normalize pixel values
    """
    # Convert binary data to OpenCV image
    img_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    # Resize to 224x224 (standard size for many CNNs)
    img_resized = cv2.resize(img, (224, 224))
    
    # Convert BGR to RGB (OpenCV uses BGR, PyTorch expects RGB)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize pixel values to [0,1]
    img_normalized = img_rgb / 255.0
    
    return img_normalized

def create_simple_cnn():
    """Create a simple CNN model for binary classification"""
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 28 * 28, 512)
            self.fc2 = nn.Linear(512, 2)  # 2 classes: cat and dog
            
        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 64 * 28 * 28)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    return SimpleCNN()

def create_resnet50():
    """Create a pre-trained ResNet50 model for binary classification"""
    model = models.resnet50(pretrained=True)
    
    # Modify the final layer for binary classification
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)  # 2 classes: cat and dog
    
    return model

def load_model(model_type="simple_cnn"):
    """Load the appropriate model based on type"""
    if model_type == "simple_cnn":
        model = create_simple_cnn()
    elif model_type == "resnet50":
        model = create_resnet50()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # In a real scenario, you would load pre-trained weights here
    # model.load_state_dict(torch.load('model_weights.pth'))
    
    model.eval()  # Set to evaluation mode
    return model

def classify_image(processed_image, model):
    """Classify a preprocessed image using the provided model"""
    # Extract filename from path if available
    filename = getattr(processed_image, 'filename', 'unknown')
    
    print(f"🔍 CLASSIFICATION: Starting classification for image {filename}")
    
    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float().unsqueeze(0)
    
    # Get prediction
    start_time = time.time()
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    elapsed = time.time() - start_time
    prediction = "Cat" if predicted.item() == 0 else "Dog"
    
    print(f"✅ CLASSIFICATION RESULT: Image {filename} classified as {prediction} in {elapsed:.4f} seconds")
    
    return predicted.item()  # 0 for cat, 1 for dog

    """Classify a preprocessed image using the provided model"""
    # Convert to PyTorch tensor
    image_tensor = torch.from_numpy(processed_image).permute(2, 0, 1).float().unsqueeze(0)
    
    # Get prediction
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
    
    return predicted.item()  # 0 for cat, 1 for dog

def process_images_distributed(sc, model_type="simple_cnn"):
    """Process images in a distributed manner using PySpark"""
    print(f"\n📊 CLASSIFICATION BATCH: Starting distributed classification with {model_type} model")
    start_time = time.time()
    
    # Load model
    model = load_model(model_type)
    print(f"🧠 MODEL LOADED: {model_type} model ready for inference")
    
    # Load images from HDFS
    print(f"📁 LOADING IMAGES: Fetching cat images from HDFS")
    cat_rdd = sc.binaryFiles("hdfs://namenode:9000/data/images/cats/*")
    cat_count = cat_rdd.count()
    print(f"🐱 CAT IMAGES: Loaded {cat_count} cat images")
    
    print(f"📁 LOADING IMAGES: Fetching dog images from HDFS")
    dog_rdd = sc.binaryFiles("hdfs://namenode:9000/data/images/dogs/*")
    dog_count = dog_rdd.count()
    print(f"🐶 DOG IMAGES: Loaded {dog_count} dog images")
    
    total_images = cat_count + dog_count
    print(f"📊 TOTAL IMAGES: Processing {total_images} images in parallel")
    
    # Combine and label
    cat_labeled = cat_rdd.map(lambda x: (x[0], x[1], 0))  # 0 for cats
    dog_labeled = dog_rdd.map(lambda x: (x[0], x[1], 1))  # 1 for dogs
    all_images = cat_labeled.union(dog_labeled)
    
    # Process in parallel
    print(f"🔄 PREPROCESSING: Starting image preprocessing")
    preprocessed = all_images.map(lambda x: (x[0], preprocess_image(x[1]), x[2]))
    
    # Then classify them
    print(f"🧮 CLASSIFYING: Starting image classification")
    results = preprocessed.map(lambda x: (x[0], classify_image(x[1], model), x[2]))
    
    # Force evaluation to get timing
    collected = results.collect()
    elapsed_time = time.time() - start_time
    
    print(f"✅ CLASSIFICATION COMPLETE: Processed {total_images} images in {elapsed_time:.2f} seconds")
    print(f"⚡ THROUGHPUT: {total_images/elapsed_time:.2f} images/second")
    
    return results

def evaluate_results(results):
    """Evaluate classification results"""
    print("\n" + "="*80)
    print("📈 CLASSIFICATION EVALUATION RESULTS")
    print("="*80)
    
    # Collect results
    collected_results = results.collect()
    
    # Calculate accuracy
    correct = sum(1 for path, pred, true in collected_results if pred == true)
    total = len(collected_results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"📊 Total images: {total}")
    print(f"✓ Correctly classified: {correct}")
    print(f"❌ Incorrectly classified: {total - correct}")
    print(f"🎯 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Calculate per-class metrics
    cat_correct = sum(1 for _, pred, true in collected_results if pred == true and true == 0)
    cat_total = sum(1 for _, _, true in collected_results if true == 0)
    dog_correct = sum(1 for _, pred, true in collected_results if pred == true and true == 1)
    dog_total = sum(1 for _, _, true in collected_results if true == 1)
    
    cat_accuracy = cat_correct / cat_total if cat_total > 0 else 0
    dog_accuracy = dog_correct / dog_total if dog_total > 0 else 0
    
    print(f"\n🐱 Cat classification accuracy: {cat_accuracy:.4f} ({cat_correct}/{cat_total})")
    print(f"🐶 Dog classification accuracy: {dog_accuracy:.4f} ({dog_correct}/{dog_total})")
    
    # Print some example predictions
    print("\n📋 Sample predictions:")
    print("-"*80)
    print(f"{'Filename':<30} {'Prediction':<15} {'Actual':<15} {'Result':<10}")
    print("-"*80)
    
    for i, (path, pred, true) in enumerate(collected_results[:10]):
        filename = path.split('/')[-1]
        result = "✓ Correct" if pred == true else "❌ Wrong"
        pred_label = "🐱 Cat" if pred == 0 else "🐶 Dog"
        true_label = "🐱 Cat" if true == 0 else "🐶 Dog"
        print(f"{filename:<30} {pred_label:<15} {true_label:<15} {result:<10}")
    
    print("="*80)

    """Evaluate classification results"""
    # Collect results
    collected_results = results.collect()
    
    # Calculate accuracy
    correct = sum(1 for path, pred, true in collected_results if pred == true)
    total = len(collected_results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"Classification Results:")
    print(f"Total images: {total}")
    print(f"Correctly classified: {correct}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print some example predictions
    print("\nSample predictions:")
    for i, (path, pred, true) in enumerate(collected_results[:10]):
        filename = path.split('/')[-1]
        result = "Correct" if pred == true else "Wrong"
        pred_label = "Cat" if pred == 0 else "Dog"
        true_label = "Cat" if true == 0 else "Dog"
        print(f"{filename}: Predicted {pred_label}, Actual {true_label} - {result}")

def benchmark(sc, num_nodes=1, model_type="simple_cnn"):
    """Benchmark the system with varying number of nodes"""
    print("\n" + "="*80)
    print(f"🚀 BENCHMARK: Running with {num_nodes} nodes using {model_type} model")
    print("="*80)
    
    # Set the number of partitions based on nodes
    sc.setLocalProperty("spark.default.parallelism", str(num_nodes))
    
    # Record start time
    start_time = time.time()
    
    # Process images
    results = process_images_distributed(sc, model_type)
    
    # Force evaluation and collect metrics
    collected_results = results.collect()
    count = len(collected_results)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    
    # Calculate throughput (images/second)
    throughput = count / elapsed_time if elapsed_time > 0 else 0
    
    # Calculate accuracy
    correct = sum(1 for _, pred, true in collected_results if pred == true)
    accuracy = correct / count if count > 0 else 0
    
    print("\n📊 BENCHMARK RESULTS:")
    print(f"🖥️  Number of nodes: {num_nodes}")
    print(f"🧠 Model type: {model_type}")
    print(f"🔢 Total images processed: {count}")
    print(f"⏱️  Total processing time: {elapsed_time:.2f} seconds")
    print(f"⚡ Throughput: {throughput:.2f} images/second")
    print(f"🎯 Classification accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("="*80)
    
    return {
        "num_nodes": num_nodes,
        "model_type": model_type,
        "count": count,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "accuracy": accuracy
    }

def main():
    # Wait for namenode to be ready
    wait_for_namenode()
    
    # Upload images to HDFS
    upload_images_to_hdfs()
    
    # ONLY AFTER upload is complete, create Spark session and load images
    spark = SparkSession.builder \
        .appName("DistributedImageProcessing") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    # Now load the images from HDFS
    image_rdd = sc.binaryFiles("hdfs://namenode:9000/data/images/*")

    print("Number of files loaded: ", image_rdd.count())
    
    # Continue with processing...

    """Main function to orchestrate the distributed image processing workflow"""
    # Wait for namenode to be ready
    wait_for_namenode()
    
    # Upload images to HDFS (only needs to be done once)
    upload_images_to_hdfs()
    
    # Start Spark
    spark = SparkSession.builder \
        .appName("DistributedImageProcessing") \
        .master("spark://spark-master:7077") \
        .getOrCreate()
    
    sc = spark.sparkContext
    
    # Simple verification that images are loaded correctly
    image_rdd = sc.binaryFiles("hdfs://namenode:9000/data/images/*")
    print("Number of files loaded: ", image_rdd.count())
    
    # Run a simple decoding test
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
    
    for path, shape in results[:5]:  # Show only first 5 for brevity
        print(f"{path.split('/')[-1]}: {shape}")
    
    # Run benchmarks with different configurations
    benchmark_results = []
    
    # Simple CNN with varying nodes
    for nodes in [1, 2, 4]:
        result = benchmark(sc, nodes, "simple_cnn")
        benchmark_results.append(result)
    
    # ResNet50 with varying nodes
    for nodes in [1, 2, 4]:
        result = benchmark(sc, nodes, "resnet50")
        benchmark_results.append(result)
    
    # Print summary of benchmark results
    print("\nBenchmark Summary:")
    print("=" * 80)
    print(f"{'Model':<10} {'Nodes':<6} {'Images':<8} {'Time (s)':<10} {'Throughput (img/s)':<20}")
    print("-" * 80)
    
    for result in benchmark_results:
        print(f"{result['model_type']:<10} {result['num_nodes']:<6} {result['count']:<8} "
              f"{result['elapsed_time']:<10.2f} {result['throughput']:<20.2f}")
    
    print("=" * 80)
    
    print("\nDistributed Image Processing completed successfully!")

if __name__ == "__main__":
    main()
