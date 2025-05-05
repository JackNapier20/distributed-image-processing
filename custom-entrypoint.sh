#!/bin/bash
NAMENODE_DIR="/tmp/hadoop-root/dfs/name"
#Formating only if the directory is empty or not initialized
if [ ! -d "$NAMENODE_DIR/current" ]; then
  echo "$(date) First-time setup for HDFS:"
  export HADOOP_CONF_DIR=/etc/hadoop
  /opt/hadoop-2.7.4/bin/hdfs namenode -format -force -nonInteractive
fi
#Starting daemon
echo "$(date) Starting NameNode service!"
export HADOOP_CONF_DIR=/etc/hadoop
/opt/hadoop-2.7.4/bin/hdfs namenode
