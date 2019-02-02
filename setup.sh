hadoop fs -mkdir /tmp/
hadoop fs -put data/churn.all /tmp/

chmod 755 cdsw-build.sh
mkdir -p models/spark
