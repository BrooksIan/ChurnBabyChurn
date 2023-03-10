## Spark ML - Multilayer Perceptron

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import trim
import pandas as pd
import cdsw
import time
import sys 

#initalize Spark Session 
spark = SparkSession.builder \
      .appName("Churn- MLP") \
      .master("local[*]") \
      .getOrCreate()

## Spark Version    
spark.version

### Start Timer
startTime = time.process_time()      
      
#Define Dataframe Schema     
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])

#Build Dataframe from File
raw_data = spark.read.schema(schemaData).csv('file:///home/cdsw/data/churn.all')
churn_data=raw_data.withColumn("intl_plan",trim(raw_data.intl_plan))

reduced_numeric_cols = ["account_length", "number_vmail_messages",
                        "total_day_charge", "total_eve_charge",
                        "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

reduced_numeric_cols1 = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

#Review Dataset Balance 
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

#Configure Multilayer Perceptron Classifier Model 
from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier

mlpclassifier = MultilayerPerceptronClassifier(labelCol = 'label', featuresCol = 'scaledFeatures', layers = [12,12, 2])

#Set Multilayer Perceptron Pipeline Stages
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, scaler, mlpclassifier])

#Spilt Test and Train Sets
(train, test) = churn_data.randomSplit([0.75, 0.25])

#Spark Model Hyper Turning
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Setting Multilayer Perceptron Paramaters From Users
user_mlp_param_layers = [ [12,12, 2], [12, 6, 2] ]
user_mlp_param_maxIter = [10, 50, 100]
user_mlp_param_blockSize = [1, 128, 256]
user_mlp_param_numFolds = 3

#Settings for Multilayer Perceptron - Paramaters Grid Search 
mlp_paramGrid = ParamGridBuilder().addGrid(mlpclassifier.layers, user_mlp_param_layers).addGrid(mlpclassifier.maxIter, user_mlp_param_maxIter).addGrid(mlpclassifier.blockSize, user_mlp_param_blockSize).build()

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
multiEvaluator = MulticlassClassificationEvaluator()

#Setting Paramaters for Crossvalidation 
mlp_cv = CrossValidator( estimator=pipeline, evaluator=evaluator, estimatorParamMaps=mlp_paramGrid, numFolds=user_mlp_param_numFolds)
mlp_cvmodel = mlp_cv.fit(train)

#Evaluating Multilayer Perceptron Model Performance 
from pyspark.sql.functions import udf

mlp_predictions = mlp_cvmodel.transform(test)

auroc = evaluator.evaluate(mlp_predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(mlp_predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

f1score = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "f1"})
weightedPrecision = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "weightedPrecision"})
weightedRecall = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "weightedRecall"})

"The F1 score: %s the Weighted Precision: %s the Weighted Recall is %s" % (f1score, weightedPrecision, weightedRecall)

#Select The Best Multilayer Perceptron Model After Crossvalidation
mlpmodel = mlp_cvmodel.bestModel 
bestMLPModel = mlpmodel.stages[-1]

#Retrieving Paramaters from the Best MLP Model 
#param_BestModel_Layers = bestMLPModel._java_obj.layers
#param_BestModel_Iter = bestMLPModel._java_obj.maxIter

### Stop Timer
stopTime = time.process_time()
elapsedTime = stopTime-startTime
"Elapsed Process Time: %0.8f" % (elapsedTime)

#Return Paramaters to CDSW User Intemlpace
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_metric("F1", f1score)
cdsw.track_metric("WeightedPrecision", weightedPrecision)
cdsw.track_metric("weightedRecall", weightedRecall)
#cdsw.track_metric("Layers",param_BestModel_Layers)
#cdsw.track_metric("maxIter",param_BestModel_Iter)
cdsw.track_metric("cvFolds",user_mlp_param_numFolds)
cdsw.track_metric("ProcTime", elapsedTime)

from pyspark.mllib.evaluation import BinaryClassificationMetrics
labelPredictionSet = mlp_predictions.select('prediction','label').rdd.map(lambda lp: (lp.prediction, lp.label))
metrics = BinaryClassificationMetrics(labelPredictionSet)
#metrics.areaUnderPR


#Save MLP Model to Disk
mlpmodel.write().overwrite().save("models/spark/mlp")

cdsw.track_file("models/spark_mlp.tar")

spark.stop()

## End of File