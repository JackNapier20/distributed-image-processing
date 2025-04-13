#!/bin/bash

NAMENODE_DIR="/tmp/hadoop-root/dfs/name"

# Format only if the directory is empty or not initialized
if [ ! -d "$NAMENODE_DIR/current" ]; then
  echo "$(date) First-time setup: formatting HDFS..."
  # Make sure we're using the correct configuration
  export HADOOP_CONF_DIR=/etc/hadoop
  /opt/hadoop-2.7.4/bin/hdfs namenode -format -force -nonInteractive
fi

# Now start the actual daemon with the correct configuration
echo "$(date) Starting NameNode service..."
export HADOOP_CONF_DIR=/etc/hadoop
/opt/hadoop-2.7.4/bin/hdfs namenode
