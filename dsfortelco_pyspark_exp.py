from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import trim
import pandas as pd
import cdsw

spark = SparkSession.builder \
      .appName("Telco Customer Churn") \
      .getOrCreate()
    
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])
raw_data = spark.read.schema(schemaData).csv('/tmp/churn.all')
churn_data=raw_data.withColumn("intl_plan",trim(raw_data.intl_plan))

reduced_numeric_cols = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]


from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
input_cols=['intl_plan_indexed'] + reduced_numeric_cols
assembler = VectorAssembler(
    inputCols = input_cols,
    outputCol = 'features')

param_numTrees=int(sys.argv[1])
param_maxDepth=int(sys.argv[2])
param_impurity=sys.argv[3]

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
classifier = RandomForestClassifier(labelCol = 'label', 
                                    featuresCol = 'features', 
                                    numTrees = param_numTrees, 
                                    maxDepth = param_maxDepth,  
                                    impurity = param_impurity)
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, classifier])
(train, test) = churn_data.randomSplit([0.7, 0.3])
model = pipeline.fit(train)

cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)


from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.functions import udf
predictions = model.transform(test)
evaluator = BinaryClassificationEvaluator()
auroc = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)

model.write().overwrite().save("models/spark")

!rm -r -f models/spark
!rm -r -f models/spark_rf.tar
!hdfs dfs -get models/spark 
!hdfs dfs -get models
!tar -cvf models/spark_rf.tar models/spark

cdsw.track_file("models/spark_rf.tar")

spark.stop()


