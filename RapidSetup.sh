#!/usr/bin/env bash

#Set Up Paths for Rapids Jars

export SPARK_RAPIDS_DIR=/opt/sparkRapidsPlugin
export SPARK_CUDF_JAR=${SPARK_RAPIDS_DIR}/cudf-0.15-SNAPSHOT-cuda10-1.jar
export SPARK_RAPIDS_PLUGIN_JAR=${SPARK_RAPIDS_DIR}/rapids-4-spark_2.12-0.2.0-SNAPSHOT.jar

chmod 775 /opt/sparkRapidsPlugin/*.jar

#/opt/sparkRapidsPlugin/cudf-0.15-SNAPSHOT-cuda10-1.jar
#/opt/sparkRapidsPlugin/rapids-4-spark_2.12-0.2.0-SNAPSHOT.jar