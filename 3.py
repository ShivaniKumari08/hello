#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  9 20:32:15 2020

@author: sagarika
"""

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.preprocessing import LabelEncoder,StandardScaler,Normalizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('Dataset_spine.csv')
data.head()
features= ['Col1','Col2','Col3','Col4','Col5','Col6','Col7','Col8','Col9','Col10','Col11','Col12']
label=['Class_att']

X = data[features]
y = data[label]
X=X.drop(['Col7','Col8','Col9','Col10','Col11','Col12'],axis=1)

X.isnull().sum()
s=Normalizer()
X.iloc[:,:]=s.fit_transform(X.iloc[:,:])
X.head()

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
  
c=pd.get_dummies(data[label],drop_first=True)
c.head()

y=pd.concat([y,c],axis=1)
y.drop(label,axis=1,inplace=True)
y.head()

clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=500, alpha=0.0001,
    solver='sgd', verbose=10,  random_state=21,tol=0.000000001)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test) 

print("Test Accuracy"+str(accuracy_score(y_test, y_pred).round(3)*100)+"%")

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, center=True,annot=True,square=True,cbar=False)
plt.title('Confusion Matrix')

plt.show() 

print(classification_report(y_pred,y_test))    