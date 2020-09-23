## Spark ML - Gradient Boost Tree on Rapids

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import trim
import pandas as pd
import cdsw
import cdsw
import time
import sys 

# Check NVidia Rapids Jars
!ls -l $SPARK_RAPIDS_DIR

RapidJarDir = os.path.join(os.environ["SPARK_RAPIDS_DIR"])
RapidJars = [os.path.join(RapidJarDir, x) for x in os.listdir(RapidJarDir)]
RapidgetGPURespource = os.environ["SPARK_RAPIDS_DIR"] + "/getGpusResources.sh" 

print(RapidJarDir)
print(RapidJars)
print(RapidgetGPURespource)

## Spark + Rapids on K8s     
# Initialize Rapids Spark Session 
spark = SparkSession.builder \
      .appName("GBT - Rapids") \
      .config("spark.plugins","com.nvidia.spark.SQLPlugin") \
      .config("spark.rapids.sql.format.csv.read.enabled", "false") \
      .config("spark.rapids.sql.enabled", "false") \
      .config("spark.executor.resource.gpu.discoveryScript", RapidgetGPURespource) \
      .config("spark.executor.resource.gpu.vendor","nvidia.com") \
      .config("spark.task.resource.gpu.amount",".25") \
      .config("spark.executor.cores","4") \
      .config("spark.executor.memoryOverhead","4G") \
      .config("spark.executor.resource.gpu.amount","1") \
      .config("spark.executor.memory","2G") \
      .config("spark.task.cpus","1") \
      .config("spark.rapids.memory.pinnedPool.size","2G") \
      .config("spark.locality.wait","0s") \
      .config("spark.sql.files.maxPartitionBytes","512m") \
      .config("spark.sql.shuffle.partitions","10") \
      .config("spark.dynamicAllocation.enabled","False") \
      .config("spark.jars", ",".join(RapidJars)) \
      .config("spark.files", RapidgetGPURespource)\
      .getOrCreate()

# Spark + Rapids Tuning Options      
#      .config("spark.plugins","com.nvidia.spark.SQLPlugin") \
#      .config("spark.rapids.sql.format.csv.read.enabled", "false") \
#      .config("spark.rapids.sql.enabled", "false") \
#      .config("spark.executor.resource.gpu.discoveryScript", RapidgetGPURespource) \
#      .config("spark.executor.resource.gpu.vendor","nvidia.com") \
#      .config("spark.task.resource.gpu.amount",".25") \
#      .config("spark.executor.cores","4") \
#      .config("spark.executor.memoryOverhead","4G") \
#      .config("spark.executor.resource.gpu.amount","1") \
#      .config("spark.executor.memory","2G") \
#      .config("spark.task.cpus","1") \
#      .config("spark.rapids.memory.pinnedPool.size","2G") \
#      .config("spark.locality.wait","0s") \
#      .config("spark.sql.files.maxPartitionBytes","512m") \
#      .config("spark.sql.shuffle.partitions","10") \

### Start Timer
startTime = time.process_time()

## Spark Version    
spark.version

#Define Dataframe Schema     
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])

#Build Dataframe from File
raw_data = spark.read.schema(schemaData).csv('data/churn.all')
churn_data=raw_data.withColumn("intl_plan",trim(raw_data.intl_plan))

reduced_numeric_cols = ["account_length", "number_vmail_messages",
                        "total_day_charge", 
                        "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

reduced_numeric_cols1 = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

#Review DataSet Balance 
churn_data.registerTempTable("ChurnData")
sqlResult = spark.sql("SELECT churned, COUNT(churned) as Churned FROM ChurnData group by churned")
sqlResult.show()

#Feature Engineering 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#String to Index
label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
input_cols=['intl_plan_indexed'] + reduced_numeric_cols1

#Feature Vector Assembler
assembler = VectorAssembler(inputCols = input_cols, outputCol = 'features')

#Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",withStd=True, withMean=False)

#Configure Gradient Boost Tree Classifier Model 
from pyspark.ml.classification import GBTClassifier
gbtclassifier = GBTClassifier(labelCol = 'label', featuresCol = 'scaledFeatures')

#Set GTB Pipeline Stages
from pyspark.ml import Pipeline
GBTpipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, scaler, gbtclassifier])

#Split Test and Train Set
(train, test) = churn_data.randomSplit([0.75, 0.25])

#Spark Model Hyper Turning
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Setting GBT Paramaters From Users
user_gbt_param_numInterSet = [2, 4, 8, 16,32]
user_gbt_param_maxDepthSet = [10, 20, 30]
user_gbt_param_numFolds = 3

#Settings for GBT - Paramaters Grid Search 
GBTparamGrid = ParamGridBuilder().addGrid(gbtclassifier.maxIter, user_gbt_param_numInterSet).addGrid(gbtclassifier.maxDepth, user_gbt_param_maxDepthSet).build()
evaluator = BinaryClassificationEvaluator()
multiEvaluator = MulticlassClassificationEvaluator()

#Setting Paramaters for Crossvalidation 
gbt_cv = CrossValidator( estimator=GBTpipeline, evaluator=evaluator, estimatorParamMaps=GBTparamGrid, numFolds=user_gbt_param_numFolds)
gbt_cvmodel = gbt_cv.fit(train)


#Evaluating GBT Model Performance 
from pyspark.sql.functions import udf

gbt_predictions = gbt_cvmodel.transform(test)
auroc = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(gbt_predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

f1score = multiEvaluator.evaluate(gbt_predictions, {multiEvaluator.metricName: "f1"})
weightedPrecision = multiEvaluator.evaluate(gbt_predictions, {multiEvaluator.metricName: "weightedPrecision"})
weightedRecall = multiEvaluator.evaluate(gbt_predictions, {multiEvaluator.metricName: "weightedRecall"})

"The F1 score: %s the Weighted Precision: %s the Weighted Recall is %s" % (f1score, weightedPrecision, weightedRecall)

#Select the Best Model GBT After Crossvalidation
gbtmodel = gbt_cvmodel.bestModel 
bestBGTModel = gbtmodel.stages[-1]

#Retrieving Paramaters from the Best RF Model 
param_BestModel_NumIter = bestBGTModel._java_obj.getMaxIter()
param_BestModel_Depth = bestBGTModel._java_obj.getMaxDepth()

#Feature Importance
impFeatures = gbtmodel.stages[-1].featureImportances
zipFeaturesToImportanceValue = zip(impFeatures, input_cols)
FeautureRankings = set(zipFeaturesToImportanceValue)
sortedFeaturRankings = sorted(FeautureRankings, reverse=True)

"Gradient Boost Tree - Feature Rankings Sorted By Importance Value %s" % (sortedFeaturRankings)
"When summed together, the values equal 1.0"

### Stop Timer
stopTime = time.process_time()
elapsedTime = stopTime-startTime
"Elapsed Process Time: %0.8f" % (elapsedTime)

#Return Paramaters to CDSW User Interface
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_metric("F1", f1score)
cdsw.track_metric("WeightedPrecision", weightedPrecision)
cdsw.track_metric("weightedRecall", weightedRecall)
cdsw.track_metric("numIter",param_BestModel_NumIter)
cdsw.track_metric("maxDepth",param_BestModel_Depth)
cdsw.track_metric("cvFolds",user_gbt_param_numFolds)


from pyspark.mllib.evaluation import BinaryClassificationMetrics
gbt_labelPredictionSet = gbt_predictions.select('prediction','label').rdd.map(lambda lp: (lp.prediction, lp.label))
gbtmetrics = BinaryClassificationMetrics(gbt_labelPredictionSet)

#Save GBT Model to Disk
gbtmodel.write().overwrite().save("models/spark/gbt")


!rm -r -f models/spark/gbt
!rm -r -f models/spark_gbt.tar
!hdfs dfs -get models/spark/gbt 
!hdfs dfs -get models/
!tar -cvf models/spark_gbt.tar models/spark/gbt

cdsw.track_file("models/spark_gbt.tar")

spark.stop()

## End of File