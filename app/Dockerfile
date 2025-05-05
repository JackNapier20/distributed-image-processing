FROM openjdk:11-slim
ENV SPARK_VERSION=3.5.5 \
    HADOOP_VERSION=3 \
    SPARK_HOME=/opt/spark \
    HADOOP_HOME=/opt/hadoop
#Installing python 3.9 and dependencies
RUN apt-get update && \
    apt-get install -y python3.9 python3.9-distutils python3-pip curl wget git dnsutils netcat procps && \
    ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    pip3 install --upgrade pip
#Installing opencv dependencies
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
#Installing pytorch and other python packages
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --upgrade pip pyspark opencv-python numpy
#Download and install hadoop client
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz && \
    tar -xvzf hadoop-3.3.4.tar.gz && \
    mv hadoop-3.3.4 ${HADOOP_HOME} && \
    rm hadoop-3.3.4.tar.gz
#Download and install spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} ${SPARK_HOME} && \
    rm spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz
ENV PATH="${SPARK_HOME}/bin:${SPARK_HOME}/sbin:${HADOOP_HOME}/bin:${PATH}"
RUN pip install --upgrade pip pyspark
WORKDIR /app
COPY . /app
CMD ["python", "main.py"]