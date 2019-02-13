#Install Packagages
install.packages("sparklyr")

#Load Libs
library(sparklyr)
library(dplyr)
library(devtools)


## Setting Up Spark Context
sc <- spark_connect(master = "local")
sc

reloaded_model <- ml_load(sc, path ="file:///home/cdsw/models/spark/rf")


emp.data <- data.frame(
  intl_plan_indexed = 0,
  account_length = 128, 
  number_vmail_messages = 25, 
  total_day_calls = 256, 
  total_day_charge = 110, 
  total_eve_calls = 197.4, 
  total_eve_charge = 50, 
  total_night_calls = 244.7, 
  total_night_charge = 91, 
  total_intl_calls = 10, 
  total_intl_charge = 5, 
  number_customer_service_calls = 1)
  
sdf_test <- copy_to(sc, emp.data)


#fitted_pipeline
pred <- ml_transform(reloaded_model, sdf_test)
#sdf_schema(pred)

summarise(pred, prediction, rawPrediction)

# Close Spark Context
spark_disconnect(sc)
