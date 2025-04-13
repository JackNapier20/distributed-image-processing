#!/bin/bash
set -e
(
  # Wait for HDFS to be fully available
  until hdfs dfsadmin -report | grep 'Datanodes available:' | grep -q '[1-9]'; do
    echo "Waiting for HDFS to be up..."
    sleep 2
  done
  echo "HDFS is up."

  # Create the /images directory and copy files
  hdfs dfs -mkdir -p /images || echo "/images already exists in HDFS."
  hdfs dfs -put /images/* /images/ 2>/dev/null || echo "Images already copied or no new images."
) &

# Then call the original entrypoint so that the namenode process runs in the foreground
exec /entrypoint.sh "$@"