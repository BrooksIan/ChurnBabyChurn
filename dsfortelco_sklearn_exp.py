from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import pickle
import cdsw

spark = SparkSession.builder \
      .appName("Telco Customer Churn") \
      .getOrCreate()
    
schemaData = StructType([StructField("state", StringType(), True),StructField("account_length", DoubleType(), True),StructField("area_code", StringType(), True),StructField("phone_number", StringType(), True),StructField("intl_plan", StringType(), True),StructField("voice_mail_plan", StringType(), True),StructField("number_vmail_messages", DoubleType(), True),     StructField("total_day_minutes", DoubleType(), True),     StructField("total_day_calls", DoubleType(), True),     StructField("total_day_charge", DoubleType(), True),     StructField("total_eve_minutes", DoubleType(), True),     StructField("total_eve_calls", DoubleType(), True),     StructField("total_eve_charge", DoubleType(), True),     StructField("total_night_minutes", DoubleType(), True),     StructField("total_night_calls", DoubleType(), True),     StructField("total_night_charge", DoubleType(), True),     StructField("total_intl_minutes", DoubleType(), True),     StructField("total_intl_calls", DoubleType(), True),     StructField("total_intl_charge", DoubleType(), True),     StructField("number_customer_service_calls", DoubleType(), True),     StructField("churned", StringType(), True)])
churn_data = spark.read.schema(schemaData).csv('tmp/churn.all')

reduced_churn_data= churn_data.select("account_length", "number_vmail_messages", "total_day_calls",
                     "total_day_charge", "total_eve_calls", "total_eve_charge",
                     "total_night_calls", "total_night_charge", "total_intl_calls", 
                    "total_intl_charge","number_customer_service_calls")

label_indexer = StringIndexer(inputCol = 'churned', outputCol = 'label')
plan_indexer = StringIndexer(inputCol = 'intl_plan', outputCol = 'intl_plan_indexed')
pipeline = Pipeline(stages=[plan_indexer, label_indexer])
indexed_data = pipeline.fit(churn_data).transform(churn_data)

(train_data, test_data) = indexed_data.randomSplit([0.7, 0.3])

pdTrain = train_data.toPandas()
pdTest = test_data.toPandas()
features = ["intl_plan_indexed","account_length", "number_vmail_messages", "total_day_calls",
                     "total_day_charge", "total_eve_calls", "total_eve_charge",
                     "total_night_calls", "total_night_charge", "total_intl_calls", 
                    "total_intl_charge","number_customer_service_calls"]

param_numTrees = int(sys.argv[1])
param_maxDepth = int(sys.argv[2])
param_impurity = sys.argv[3]

randF=RandomForestClassifier(n_jobs=10,
                             n_estimators=param_numTrees, 
                             max_depth=param_maxDepth, 
                             criterion = param_impurity,
                             random_state=0)

cdsw.track_metric("numTrees",param_numTrees)
cdsw.track_metric("maxDepth",param_maxDepth)
cdsw.track_metric("impurity",param_impurity)

randF.fit(pdTrain[features], pdTrain['label'])

predictions=randF.predict(pdTest[features])

#temp = randF.predict_proba(pdTest[features])

pd.crosstab(pdTest['label'], predictions, rownames=['Actual'], colnames=['Prediction'])

list(zip(pdTrain[features], randF.feature_importances_))


y_true = pdTest['label']
y_scores = predictions
auroc = roc_auc_score(y_true, y_scores)
ap = average_precision_score (y_true, y_scores)
print(auroc, ap)

cdsw.track_metric("auroc", auroc)
cdsw.track_metric("ap", ap)

pickle.dump(randF, open("models/sklearn_rf.pkl","wb"))

cdsw.track_file("models/sklearn_rf.pkl")