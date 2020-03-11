from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import trim
import pandas as pd
import cdsw

#initalize Spark Session 
spark = SparkSession.builder \
      .master("local[*]") \
      .appName("Telco Customer Churn RF") \
      .config("spark.hadoop.fs.s3a.s3guard.ddb.region","us-east-1") \
      .config("spark.yarn.access.hadoopFileSystems","s3a://ibrooks008-cdp-bucket") \
      .config("spark.hadoop.yarn.resourcemanager.principal", "ibrooks") \
      .getOrCreate()


#Define Dataframe Schema     
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])

#Build Dataframe from File
raw_data = spark.read.schema(schemaData).csv('s3a://ibrooks008-cdp-bucket/TestBucket/churn.all')
churn_data=raw_data.withColumn("intl_plan",trim(raw_data.intl_plan))

reduced_numeric_cols = ["account_length", "number_vmail_messages",
                        "total_day_charge", "total_eve_charge",
                        "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

reduced_numeric_cols1 = ["account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]

#Review DataSet Balance 
churn_data.registerTempTable("ChurnData")
#sqlResult = spark.sql("SELECT churned, COUNT(churned) as Churned FROM ChurnData group by churned")
#sqlResult.show()

#Feature Engineering 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#String to Index
label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
input_cols=['intl_plan_indexed'] + reduced_numeric_cols


#Feature Vector Assembler
assembler = VectorAssembler(inputCols = input_cols, outputCol = 'features')

#Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",withStd=True, withMean=False)

#Configure Random Forest Classifier Model 
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

rfclassifier = RandomForestClassifier(labelCol = 'label', featuresCol = 'scaledFeatures')

#Set Random Forest Pipeline Stages
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, scaler, rfclassifier])

#Spilt Test and Train Sets
(train, test) = churn_data.randomSplit([0.75, 0.25])

#Spark Model Hyper Turning
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Setting Random Forest Paramaters From Users
user_rf_param_numTreeSet = [5, 10, 20, 40, 80]
user_rf_param_maxDepthSet = [10, 20, 30]
user_rf_param_impuritySet = ['gini', 'entropy']
user_rf_param_numFolds = 2

#Settings for Random Forest - Paramaters Grid Search 
rf_paramGrid = ParamGridBuilder().addGrid(rfclassifier.numTrees, user_rf_param_numTreeSet).addGrid(rfclassifier.maxDepth, user_rf_param_maxDepthSet).addGrid(rfclassifier.impurity, user_rf_param_impuritySet).build()
evaluator = BinaryClassificationEvaluator()
multiEvaluator = MulticlassClassificationEvaluator()

#Setting Paramaters for Crossvalidation 
rf_cv = CrossValidator( estimator=pipeline, evaluator=evaluator, estimatorParamMaps=rf_paramGrid, numFolds=user_rf_param_numFolds)
rf_cvmodel = rf_cv.fit(train)

#Evaluating Random Forest Model Performance 
from pyspark.sql.functions import udf

rf_predictions = rf_cvmodel.transform(test)
auroc = evaluator.evaluate(rf_predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(rf_predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

f1score = multiEvaluator.evaluate(rf_predictions, {multiEvaluator.metricName: "f1"})
weightedPrecision = multiEvaluator.evaluate(rf_predictions, {multiEvaluator.metricName: "weightedPrecision"})
weightedRecall = multiEvaluator.evaluate(rf_predictions, {multiEvaluator.metricName: "weightedRecall"})

"The F1 score: %s the Weighted Precision: %s the Weighted Recall is %s" % (f1score, weightedPrecision, weightedRecall)

#Select the Random Forest Best Model after Crossvalidation
rfmodel = rf_cvmodel.bestModel 
bestRFModel = rfmodel.stages[-1]

#Retrieving Paramaters from the Best RF Model 
param_BestModel_NumTrees = bestRFModel._java_obj.getNumTrees()
param_BestModel_Depth = bestRFModel._java_obj.getMaxDepth()
param_BestModel_impurity = bestRFModel._java_obj.getImpurity()

#Feature Importance
impFeatures = rfmodel.stages[-1].featureImportances
zipFeaturesToImportanceValue = zip(impFeatures, input_cols)
FeautureRankings = set(zipFeaturesToImportanceValue)
sortedFeaturRankings = sorted(FeautureRankings, reverse=True)

"Random Forest - Feature Rankings Sorted By Importance Value %s" % (sortedFeaturRankings)
"When summed together, the values equal 1.0"

#Return Paramaters to CDSW User Interface
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_metric("F1", f1score)
cdsw.track_metric("WeightedPrecision", weightedPrecision)
cdsw.track_metric("weightedRecall", weightedRecall)
cdsw.track_metric("numTrees",param_BestModel_NumTrees)
cdsw.track_metric("maxDepth",param_BestModel_Depth)
cdsw.track_metric("impurity",param_BestModel_impurity)
cdsw.track_metric("cvFolds",user_rf_param_numFolds)

from pyspark.mllib.evaluation import BinaryClassificationMetrics
labelPredictionSet = rf_predictions.select('prediction','label').rdd.map(lambda lp: (lp.prediction, lp.label))
metrics = BinaryClassificationMetrics(labelPredictionSet)

#Save RF Model to S3
rfmodel.write().overwrite().save("s3a://ibrooks008-cdp-bucket/ibrooks008-dl/model_output/spark/rf")

#Stop Spark
spark.stop()
