FROM openjdk:11-slim

# Install Python 3.9 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 python3.9-distutils python3-pip curl wget git \
    && ln -s /usr/bin/python3.9 /usr/bin/python \
    && ln -s /usr/bin/python3.9 /usr/bin/python3 \
    && pip3 install --upgrade pip

# Set Spark & Python env variables
ENV PYSPARK_PYTHON=/usr/bin/python3
ENV PYSPARK_DRIVER_PYTHON=/usr/bin/python3
ENV SPARK_HOME=/opt/spark
ENV PATH=$SPARK_HOME/bin:$PATH

# Install Spark
ENV SPARK_VERSION=3.5.5
ENV HADOOP_VERSION=3

RUN wget https://downloads.apache.org/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz

WORKDIR $SPARK_HOME