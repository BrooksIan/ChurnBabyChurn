import pickle
import numpy as np

model = pickle.load(open("models/sklearn_rf.pkl","rb"))

def predict(args):
  print(args["feature"])
  account=np.array(args["feature"].split(",")).reshape(1,-1)
  return {"result" : model.predict(account)[0]}
  
#features = ["intl_plan_indexed","account_length", "number_vmail_messages", "total_day_calls",
#                     "total_day_charge", "total_eve_calls", "total_eve_charge",
#                     "total_night_calls", "total_night_charge", "total_intl_calls", 
#                    "total_intl_charge","number_customer_service_calls"
predict({
  "feature": "1, 128, 25, 256, 110, 197.4, 50, 244.7, 91, 10, 5, 1"
}) 
