## Data Science in Apache Py-Spark
### Customer Churn Project
#### Ensemable Models

**Level**: Moderate

**Language**: Python

**Requirements**: 
- HDP + CDSW 
- Spark 2.3

**Author**: Ian Brooks

**Follow**: [LinkedIn - Ian Brooks PhD](https://www.linkedin.com/in/ianrbrooksphd/)

**Orginal Fork From**: [CDSW Demo](https://github.infra.cloudera.com/SE-SPEC-DPML/dsfortelcoCDSW) 

## Churn Baby Churn 

![churn](https://blog.aircall.io/wp-content/uploads/2017/03/customer-churn.png "churn")

This Github repo is designed to be optmized for Cloudera Data Science Workbench (CDSW), but it is not required.  The PySpark code can be used with Apache Spark, and the code examples wil run with the included dataset.

In this project, there are 5 different supervised classifer models designed for telco customer churn.  The first four classsifer models user are: Random Forest, Gradient Boost Tree, Suport Vector Machines, and Multilayer Perception.  The most sucessful model is a Stacked Ensemble Model.    

## CDSW Run Instructions

1.  In CSDW, download the project using the git url for [here](https://github.com/BrooksIan/ChurnBabyChurn.git) 
2.  Open a new session, and execute the setup.sh file
3.  In Expirments, run the following scripts
    * dsforteko_pyspark.py  - vanilla random forest churn model
    * gbt_churn_pyspark.py  - gradient boost tree churn model with normamlized variables, hyperturning, and crossvalidation
    * mlp_churn_pyspark.py  - multilayer perceptron churn model with normamlized variables, hyperturning, and crossvalidatio
    * RF_churn_pyspark.py  -  random forest churn model with normamlized variables, hyperturning, and crossvalidation
    * SVM_churn_pyspark.py -  support vection machine churn model with normamlized variables, hyperturning, and crossvalidation
4. Once all experiments have completed, the stacked ensemble classifer model be built. run the following script
   * stacked_churn_pyspark.py - stacked ensemble model trained on the prediction of random forest, gradient boost tree, and support vector machine 
5. Once the stacked experiment has been completed, the stacked model can be deployed using the following script.
   * predict_churn_stackedMLP_pyspark.py
