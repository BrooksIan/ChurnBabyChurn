from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType
from pyspark.sql.functions import trim
import pandas as pd
import cdsw

spark = SparkSession.builder \
      .appName("Telco Customer Churn - Stacked MLP") \
      .config('spark.shuffle.service.enabled',"True") \
      .master("local[*]") \
      .getOrCreate()

#Define Dataframe Schema     
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])


#Load Pipeline Models
modelrf = PipelineModel.load("models/spark/rf") 
modelgbt = PipelineModel.load("models/spark/gbt") 
modelmlp = PipelineModel.load("models/spark/mlp")
modelsvm = PipelineModel.load("models/spark/svm") 


features = ["account_length", "number_vmail_messages", "total_day_calls",
            "total_day_charge", "total_eve_calls", "total_eve_charge",
            "total_night_calls", "total_night_charge", "total_intl_calls", 
            "total_intl_charge","number_customer_service_calls",
            "RF_Prediction", "GBT_Prediction", "SVM_Prediction"]

predfeatures = ["RF_Prediction", "GBT_Prediction", "SVM_Prediction"]

sharedFeatures = ["churned","phone_number","area_code", "intl_plan", "account_length", "number_vmail_messages", 
                  "total_day_calls","total_day_charge", "total_eve_calls", "total_eve_charge",
                  "total_night_calls", "total_night_charge", "total_intl_calls", 
                   "total_intl_charge","number_customer_service_calls",
                   "rawPrediction", "prediction"]

joinFeatures = ["churned","phone_number","area_code","number_vmail_messages", "intl_plan", 
                                               "account_length","total_day_calls", "total_day_charge",
                                              "total_eve_calls","total_eve_charge","total_night_calls",
                                              "total_night_charge","total_intl_calls","total_intl_charge",
                                              "number_customer_service_calls","total_night_charge",
                                              "total_intl_calls","total_intl_charge","number_customer_service_calls"]

#Create Dataframe from Churn Data
raw_data = spark.read.schema(schemaData).csv('file:///home/cdsw/data/churn.all')
churn_data=raw_data.withColumn("intl_plan",trim(raw_data.intl_plan))

#Spilt Test and Train Sets
(train, test) = churn_data.randomSplit([0.75, 0.25])

#Review Trainset Schema
train.printSchema()

# Random Forest
rf_train = modelrf.transform(train)
rf_TrainSet = rf_train.select(sharedFeatures).withColumnRenamed('rawPrediction','RF_RawPrediction').withColumnRenamed('prediction','RF_Prediction')
rf_test = modelrf.transform(test)
rf_TestSet = rf_test.select(sharedFeatures).withColumnRenamed('rawPrediction','RF_RawPrediction').withColumnRenamed('prediction','RF_Prediction')

# Gradient Boost Tree
gbt_train = modelgbt.transform(train)
gbt_TrainSet = gbt_train.select(sharedFeatures).withColumnRenamed('rawPrediction','GBT_RawPrediction').withColumnRenamed('prediction','GBT_Prediction')
gbt_test = modelgbt.transform(test)
gbt_TestSet = gbt_test.select(sharedFeatures).withColumnRenamed('rawPrediction','GBT_RawPrediction').withColumnRenamed('prediction','GBT_Prediction')

# Support Vector Machine
svm_train = modelsvm.transform(train)
svm_TrainSet = svm_train.select(sharedFeatures).withColumnRenamed('rawPrediction','SVM_RawPrediction').withColumnRenamed('prediction','SVM_Prediction')
svm_test = modelsvm.transform(test)
svm_TestSet = svm_test.select(sharedFeatures).withColumnRenamed('rawPrediction','SVM_RawPrediction').withColumnRenamed('prediction','SVM_Prediction')

#Build Train Set
Stack_TrainSet1 = rf_TrainSet.join(gbt_TrainSet, joinFeatures)
Stack_TrainSet = Stack_TrainSet1.join(svm_TrainSet, joinFeatures )
Stack_TrainSet.printSchema()

#Build Test Set
Stack_TestSet1 = rf_TestSet.join(gbt_TestSet, joinFeatures)
Stack_TestSet = Stack_TestSet1.join(svm_TestSet, joinFeatures )
Stack_TestSet.printSchema()


#Feature Engineering 
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler

#String to Index
label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
input_cols=['intl_plan_indexed'] + features

#Feature Vector Assembler
assembler = VectorAssembler(inputCols = predfeatures, outputCol = 'features')

#Standard Scaler
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",withStd=True, withMean=False)

#Define MLP Classifer 
from pyspark.ml.classification import MultilayerPerceptronClassifier
mlpclassifier = MultilayerPerceptronClassifier(labelCol = 'label', featuresCol = 'scaledFeatures', layers = [15,15, 2])

#Set Pipeline Stages
from pyspark.ml import Pipeline
pipeline = Pipeline(stages=[plan_indexer, label_indexer, assembler, scaler, mlpclassifier])

#Spark Model Hyper Turning
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

#Setting Multilayer Perceptron Paramaters From Users
user_mlp_param_layers = [ [3, 3, 2], [3, 6, 2]  ]
user_mlp_param_maxIter = [10, 50, 100]
user_mlp_param_blockSize = [1, 128, 256]
user_mlp_param_numFolds = 2

#Settings for Multilayer Perceptron - Paramaters Grid Search 
mlp_paramGrid = ParamGridBuilder().addGrid(mlpclassifier.layers, user_mlp_param_layers).addGrid(mlpclassifier.maxIter, user_mlp_param_maxIter).addGrid(mlpclassifier.blockSize, user_mlp_param_blockSize).build()

evaluator = BinaryClassificationEvaluator(rawPredictionCol='rawPrediction', labelCol='label')
multiEvaluator = MulticlassClassificationEvaluator()

#Setting Paramaters for Crossvalidation 
mlp_cv = CrossValidator( estimator=pipeline, evaluator=evaluator, estimatorParamMaps=mlp_paramGrid, numFolds=user_mlp_param_numFolds)
mlp_cvmodel = mlp_cv.fit(Stack_TrainSet)

#Select the Best Model after Crossvalidation
mlpmodel = mlp_cvmodel.bestModel 
bestMLPModel = mlpmodel.stages[-1]

#Evaluating Multilayer Perceptron Model Performance 
from pyspark.sql.functions import udf

mlp_predictions = mlp_cvmodel.transform(Stack_TestSet)

auroc = evaluator.evaluate(mlp_predictions, {evaluator.metricName: "areaUnderROC"})
aupr = evaluator.evaluate(mlp_predictions, {evaluator.metricName: "areaUnderPR"})
"The AUROC is %s and the AUPR is %s" % (auroc, aupr)

f1score = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "f1"})
weightedPrecision = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "weightedPrecision"})
weightedRecall = multiEvaluator.evaluate(mlp_predictions, {multiEvaluator.metricName: "weightedRecall"})

"The F1 score: %s the Weighted Precision: %s the Weighted Recall is %s" % (f1score, weightedPrecision, weightedRecall)

#Return Paramaters to CDSW User Interface
cdsw.track_metric("auroc", auroc)
cdsw.track_metric("aupr", aupr)
cdsw.track_metric("F1", f1score)
cdsw.track_metric("WeightedPrecision", weightedPrecision)
cdsw.track_metric("weightedRecall", weightedRecall)
cdsw.track_metric("cvFolds",user_mlp_param_numFolds)

#Save MLP Model to Disk
mlpmodel.write().overwrite().save("models/spark/stackedmlp")

!rm -r -f models/spark/stackedmlp
!rm -r -f models/spark_stackedmlp.tar
!hdfs dfs -get models/spark/stackedmlp 
!hdfs dfs -get models/
!tar -cvf models/spark_stackedmlp.tar models/spark/stackedmlp

cdsw.track_file("models/spark_stackedmlp.tar")

spark.stop()
