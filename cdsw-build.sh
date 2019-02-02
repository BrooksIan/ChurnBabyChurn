#!/bin/bash

if [ -d "models/spark/" ] 
then
  rm -r -f models/spark/*
else
  mkdir -p models/spark
fi

if [ -f "spark_rf.tar" ]
then 
  tar -xf spark_rf.tar 
fi

if [ -f "sklearn_rf.pkl" ]
then 
  mv sklearn_rf.pkl models
fi