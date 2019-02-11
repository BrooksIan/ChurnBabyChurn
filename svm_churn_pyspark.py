from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import trim
import pandas as pd
import cdsw

#initalize Spark Session 
spark = SparkSession.builder \
      .appName("Telco Customer Churn SVM") \
      .config('spark.shuffle.service.enabled',"True") \
      .getOrCreate()

#Define Dataframe Schema     
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])

#Build Dataframe from File
raw_data = spark.read.schema(schemaData).csv('/tmp/churn.all')
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
sqlResult = spark.sql("SELECT churned, COUNT(churned) as Churned FROM ChurnData group by churned")
sqlResult.show()

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
from pyspark.ml.classification import LinearSVC

#svmclassifier = LinearSVC(labelCol = 'label', featuresCol = 'scaledFeatures')
svmclassifier = LinearSVC(labelCol = 'label', featuresCol = 'features')

#Set Random Forest Pipeline Stages
#pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, scaler, svmclassifier])
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, svmclassifier])


#Spilt Test and Train Sets
(train, test) = churn_data.randomSplit([0.75, 0.25])

#Spark Model Hyper Turning
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Setting Random Forest Paramaters From Users
user_svm_param_maxIter = [10, 20, 30]
user_svm_param_numFolds = 2

#Settings for Random Forest - Paramaters Grid Search 
svm_paramGrid = ParamGridBuilder().addGrid(svmclassifier.maxIter, user_svm_param_maxIter).build()
evaluator = BinaryClassificationEvaluator()
multiEvaluator = MulticlassClassificationEvaluator()

#Setting Paramaters for Crossvalidation 
svm_cv = CrossValidator( estimator=pipeline, evaluator=evaluator, estimatorParamMaps=svm_paramGrid, numFolds=user_svm_param_numFolds)
svm_cvmodel = svm_cv.fit(train)

#Evaluating Random Forest Model Performance 
from pyspark.sql.functions import udf

svm_predictions = svm_cvmodel.transform(test)
auroc = evaluator.evaluate(svm_predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(svm_predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

f1score = multiEvaluator.evaluate(svm_predictions, {multiEvaluator.metricName: "f1"})
weightedPrecision = multiEvaluator.evaluate(svm_predictions, {multiEvaluator.metricName: "weightedPrecision"})
weightedRecall = multiEvaluator.evaluate(svm_predictions, {multiEvaluator.metricName: "weightedRecall"})

"The F1 score: %s the Weighted Precision: %s the Weighted Recall is %s" % (f1score, weightedPrecision, weightedRecall)

#Select the Random Forest Best Model after Crossvalidation
svmModel = svm_cvmodel.bestModel 
bestSVMModel = svmModel.stages[-1]

#Retrieving Paramaters from the Best RF Model 
param_BestModel_Iter = bestSVMModel._java_obj.getMaxIter()

#Return Paramaters to CDSW User Interface
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_metric("F1", f1score)
cdsw.track_metric("WeightedPrecision", weightedPrecision)
cdsw.track_metric("weightedRecall", weightedRecall)
cdsw.track_metric("maxIter",param_BestModel_Iter)
cdsw.track_metric("cvFolds",user_svm_param_numFolds)


from pyspark.mllib.evaluation import BinaryClassificationMetrics
labelPredictionSet = svm_predictions.select('prediction','label').rdd.map(lambda lp: (lp.prediction, lp.label))
metrics = BinaryClassificationMetrics(labelPredictionSet)

#Save SVM Model to Disk
svmModel.write().overwrite().save("models/spark/svm")

!rm -r -f models/spark/svm
!rm -r -f models/spark_svm.tar
!hdfs dfs -get models/spark/svm
!hdfs dfs -get models/
!tar -cvf models/spark_svm.tar models/spark/svm

cdsw.track_file("models/spark_svm.tar")

spark.stop()
