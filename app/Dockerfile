FROM openjdk:11-slim

ENV SPARK_VERSION=3.5.5 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/opt/spark

# Install Python 3.9 and dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip curl wget git && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    pip3 install --upgrade pip

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
RUN pip install --upgrade pip pyspark opencv-python

# Download and install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${PATH}"

RUN pip install --upgrade pip pyspark

WORKDIR /app
COPY . /app

# Optional: install any Python packages your app uses
RUN pip install -r requirements.txt || true

#Copy images
# FROM bde2020/hadoop-namenode:2.0.0-hadoop2.7.4-java8

# # Copy the custom entrypoint script into the image
# COPY copyimages.sh /usr/local/bin/entrypoint.sh

# # Ensure the script has executable permissions
# RUN chmod +x /usr/local/bin/entrypoint.sh

# # Use the custom entrypoint script as the entrypoint
# ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

CMD ["python", "main.py"]