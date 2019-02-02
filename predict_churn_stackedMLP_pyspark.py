from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, Row

spark = SparkSession.builder \
      .appName("Telco Customer Churn") \
      .master("local[*]") \
      .getOrCreate()

modelrf = PipelineModel.load("models/spark/rf") 
modelgbt = PipelineModel.load("models/spark/gbt") 
modelsvm = PipelineModel.load("models/spark/svm")
modelStacked = PipelineModel.load("models/spark/stackedmlp")


Pipelinefeatures = ["intl_plan", "account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]
def predict(args):
  account=args["feature"].split(",")
  feature = spark.createDataFrame([account[:1] + list(map(float,account[1:12]))], Pipelinefeatures)
  
  resultrf = modelrf.transform(feature).collect()[0].prediction
  resultgbt = modelgbt.transform(feature).collect()[0].prediction
  resultsvm = modelsvm.transform(feature).collect()[0].prediction
  
  # Create the Employees
  modelResult = Row("RF_Prediction", "GBT_Prediction", "SVM_Prediction")
  rowResult = modelResult(resultrf, resultgbt, resultsvm )
  modelRow = [rowResult]
  
  stackFeatures = spark.createDataFrame(modelRow)
  result = modelStacked.transform(stackFeatures).collect()[0].prediction
  
  return {"result " : result}

#features = ["intl_plan_indexed","account_length", "number_vmail_messages", "total_day_calls",
#                     "total_day_charge", "total_eve_calls", "total_eve_charge",
#                     "total_night_calls", "total_night_charge", "total_intl_calls", 
#                    "total_intl_charge","number_customer_service_calls"
predict({
  "feature": "no, 128, 25, 256, 110, 197.4, 50, 244.7, 91, 10, 5, 1"
}) 
