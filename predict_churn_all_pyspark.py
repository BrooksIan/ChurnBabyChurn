from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pyspark.sql.types import StructType, StructField, IntegerType, StringType

spark = SparkSession.builder \
      .appName("Telco Customer Churn") \
      .master("local[*]") \
      .getOrCreate()

modelrf = PipelineModel.load("models/spark/rf") 
modelgbt = PipelineModel.load("models/spark/gbt") 
modelmlp = PipelineModel.load("models/spark/mlp")
modelSVM = PipelineModel.load("models/spark/svm") 


features = ["intl_plan", "account_length", "number_vmail_messages", "total_day_calls",
                        "total_day_charge", "total_eve_calls", "total_eve_charge",
                        "total_night_calls", "total_night_charge", "total_intl_calls", 
                        "total_intl_charge","number_customer_service_calls"]
def predict(args):
  account=args["feature"].split(",")
  feature = spark.createDataFrame([account[:1] + list(map(float,account[1:12]))], features)
  
  resultrf = modelrf.transform(feature).collect()[0].prediction
  resultgbt = modelgbt.transform(feature).collect()[0].prediction
  resultmlp = modelmlp.transform(feature).collect()[0].prediction
  resultsvm = modelmlp.transform(feature).collect()[0].prediction
  
  return {"result RF"  : resultrf, 
          "result GBT" : resultgbt, 
          "result MLP" : resultmlp, 
          "result SVM" : resultsvm}

#features = ["intl_plan_indexed","account_length", "number_vmail_messages", "total_day_calls",
#                     "total_day_charge", "total_eve_calls", "total_eve_charge",
#                     "total_night_calls", "total_night_charge", "total_intl_calls", 
#                    "total_intl_charge","number_customer_service_calls"
predict({
  "feature": "no, 128, 25, 256, 110, 197.4, 50, 244.7, 91, 10, 5, 1"
}) 
