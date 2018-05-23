#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:59:14 2018

@author: antoinekrainc
"""

# importer les librairies
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# importer le dataset 
dataset = pd.read_csv("train.csv")

# Detecting Nan

def isnan(dataframe, column):
    for i in range(0, column):
            if dataframe.iloc[:,i].isnull().any() == True:
                print("Column ", i, "has Nan")
                

isnan(dataset, 81)

# Replacing nan for strings
                
for i in range(0, 81):
    if type(dataset.iloc[0, i]) == str or np.isnan(dataset.iloc[0,i]):
        if dataset.iloc[:,i].isnull().any() == True:
            dataset.iloc[:, i] = dataset.iloc[:, i].replace(np.nan, "None", regex = True)
        

# Spliting Dataset into independant & Dependant Variables
X = dataset.iloc[:, 1:-1].values
Y = dataset.iloc[:, 80:81].values


# Replacing Missing Values for numbers
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)
imputer.fit(X[:, [2, 25, 58]])
X[:,[2, 25, 58]] = imputer.transform(X[:, [2, 25, 58]])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
k = []
for i in range(0,79):
    if type(X[0,i]) == str:        
        X[:,i] = labelencoder.fit_transform(X[:,i])
        k +=[i]
        
onehotencoder = OneHotEncoder(categorical_features = [k])    
X = onehotencoder.fit_transform(X).toarray()

# Séparer entre training set et test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

# linear Regression Model

from sklearn.linear_model import LinearRegression
regressor_lr = LinearRegression()
regressor_lr.fit(X_train, y_train)

y_pred_lr = regressor_lr.predict(X_test)


# Testing accuracy of each model
accuracy_lr = []
        
for i in range (0, 292):
    if y_test[i] - y_pred_lr[i] < 0:
        accuracy_lr.append(y_pred_lr[i] - y_test[i])
    else:
        accuracy_lr.append(y_test[i] - y_pred_lr[i])


accuracy_lr = np.asarray(accuracy_lr)
accuracy_lr.mean()

# OLS 
import statsmodels.formula.api as sm
X = np.append(X, np.ones((1460,1)).astype(int), axis=1)
regressor_OLS = sm.OLS(endog = Y, exog = X).fit()
regressor_OLS.summary()

corr = np.corrcoef(X, rowvar =0)

# Applying model to test.csv
test = pd.read_csv("test.csv")

# Preprocessing test file

# Replacing nan for strings
                
for i in range(0, 80):
    if type(test.iloc[0, i]) == str or np.isnan(test.iloc[0,i]):
        if test.iloc[:,i].isnull().any() == True:
            test.iloc[:, i] = test.iloc[:, i].replace(np.nan, "None", regex = True)
                   
isnan(test, 80)

# Replacing Missing Values for numbers
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values="NaN", strategy = "median", axis = 0)
imputer.fit(test.iloc[:, [3, 26, 34, 36, 37, 38]])
test.iloc[:,[3, 26, 34, 36, 37, 38]] = imputer.transform(test.iloc[:, [3, 26, 34, 36, 37, 38]])

imputer.fit(test.iloc[:, [47, 48, 59, 61, 62]])
test.iloc[:, [47, 48, 59, 61, 62]] = imputer.transform(test.iloc[:,[47, 48, 59, 61, 62]])


# Encoding Categorical Variables
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for i in range(0,80):
    if type(test.iloc[0,i]) == str:        
        test.iloc[:,i] = labelencoder.fit_transform(test.iloc[:,i])
        onehotencoder = OneHotEncoder(categorical_features = [i])
    
test = onehotencoder.fit_transform(test).toarray()

test = test[:,1:]

y_pred_to_df = pd.DataFrame(y_pred_rd_testset)
y_pred_to_df.to_csv("submissions_final.csv")
