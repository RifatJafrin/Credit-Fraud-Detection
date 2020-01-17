#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 23:57:55 2019

@author: Rifat

Problem Statement:
    Identify fraudulent credit card transactions. It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase

Data Source:
    https://www.kaggle.com/mlg-ulb/creditcardfraud
"""
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd

dataset = pd.read_csv('creditcard.csv')

## DATA OBSERVATION #######################################################
#analyze the data
print(dataset.shape)

dataset.head()

dataset.info()


class_type = {0:'Not Fraud', 1:'Fraud'}
print(dataset.Class.value_counts().rename(index = class_type))

## FEATURE ENGINEERING #######################################################

#the independent variable
independentfeatures = dataset.iloc[:, 1:30].columns

# The dependent variable which we would like to predict, is the 'Class' variable
target = 'Class'

#Create a variable X containing all the dependent 
X = dataset[independentfeatures]
y = dataset[target]

 ## Run models and evaluate ##########################################################

# split the dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, test_size=0.40, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

def modelfit(alg, Xtrain, ytrain, Xtest, ytest, algname):
    alg.fit(Xtrain, ytrain)
    ypred = alg.predict(Xtest)
    cm = confusion_matrix(ytest, ypred)
    print ('\n\n'+algname+' Confusion Metrix:\n'+ str(cm))
    print (algname+' Classification Report:\n' +str(classification_report(ytest, ypred)))
    
# Logistic Regression
from sklearn.linear_model import LogisticRegression
lclassifier = LogisticRegression(random_state = 0)
algname = 'Logistic Regression'
modelfit(lclassifier, X_train, y_train, X_test, y_test, algname)

# K-Nearest Neighbors (K-NN)
from sklearn.neighbors import KNeighborsClassifier
knnclassifier = KNeighborsClassifier(metric = 'minkowski', p = 2)
algname = 'K-NN'
modelfit(knnclassifier, X_train, y_train, X_test, y_test, algname)

# Support Vector Machine (SVM)
from sklearn.svm import SVC
svmclassifier = SVC(kernel = 'linear', random_state = 0)
algname = 'SVM'
modelfit(svmclassifier, X_train, y_train, X_test, y_test, algname)

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dtclassifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
algname = 'Decision Tree'
modelfit(dtclassifier, X_train, y_train, X_test, y_test, algname)

#RandomForestClassification
from sklearn.ensemble import RandomForestClassifier
rfclassifier = RandomForestClassifier(n_estimators = 100, n_jobs =4, criterion = 'entropy', random_state = 20)
algname = 'Random Forest Classification'
modelfit(rfclassifier, X_train, y_train, X_test, y_test, algname)
    

## Findings ###############################################
"""
We first defined some models  and then loop through a train and test set .First, we train the model by the train set and 
then valide the results with the test set.In the following we will share the test results of each model and see which model works best for this dataset.

1. Logistic Regression Confusion Metrix:
[[113710     14]
 [    81    118]]

Logistic Regression Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113724
           1       0.89      0.59      0.71       199

   micro avg       1.00      1.00      1.00    113923
   macro avg       0.95      0.80      0.86    113923
weighted avg       1.00      1.00      1.00    113923

The transactions without fraud (class = 0) are predicted with 100% precision and recall.
The fraudent transection (class = 1) are predicted with 89% precision and f1-score is .71 .
So,  It can predict fraud with 89% precision

2. K-NN Confusion Metrix:
[[113713     11]
 [    51    148]]

K-NN Classification Report:-
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113724
           1       0.93      0.74      0.83       199

   micro avg       1.00      1.00      1.00    113923
   macro avg       0.97      0.87      0.91    113923
weighted avg       1.00      1.00      1.00    113923

The transactions without fraud (class = 0) are predicted with 100% precision and recall.
The fraudent transection (class = 1) are predicted with 93% precision and f1-score is .83.
So, It can predict fraud with 93% precision

3. SVM Confusion Metrix:
[[113694     30]
 [    41    158]]

SVM Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113724
           1       0.84      0.79      0.82       199

   micro avg       1.00      1.00      1.00    113923
   macro avg       0.92      0.90      0.91    113923
weighted avg       1.00      1.00      1.00    113923

The transactions without fraud (class = 0) are predicted with 100% precision and recall.
The fraudent transection (class = 1) are predicted with 84% precision and f1-score is .82.
So, It can predict fraud with 84% precision

4. Decision Tree Confusion Metrix:
[[113687     37]
 [    55    144]]

Decision Tree Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113724
           1       0.80      0.72      0.76       199

   micro avg       1.00      1.00      1.00    113923
   macro avg       0.90      0.86      0.88    113923
weighted avg       1.00      1.00      1.00    113923

The transactions without fraud (class = 0) are predicted with 100% precision and recall.
The fraudent transection (class = 1) are predicted with 80% precision and f1-score is .76.
So, It can predict fraud with 80% precision

5. Random Forest Classification Confusion Metrix:
[[113716      8]
 [    43    156]]

Random Forest Classification Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00    113724
           1       0.95      0.78      0.86       199

   micro avg       1.00      1.00      1.00    113923
   macro avg       0.98      0.89      0.93    113923
weighted avg       1.00      1.00      1.00    113923

The transactions without fraud (class = 0) are predicted with 100% precision and recall.
The fraudent transection (class = 1) are predicted with 95% precision and f1-score is .86.
So, It can predict fraud with 95% precision.

After analyzing all these model we see that The random forest model have precision score .95 for fraudent transection.
This means only 5% of fraudulent transactions are undetected by the system.
"""