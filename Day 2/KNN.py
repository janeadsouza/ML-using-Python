# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 14:36:42 2018

@author: jd
"""

import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris=load_iris()

print(iris['feature_names'])

print(iris['target'])

x=iris.data
y=iris.target

print(x)

print(y)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1) #to randomly assign training sets

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#Training the model using KNN

neigh=KNeighborsClassifier(n_neighbors=3)   #object creation
fit=neigh.fit(x_train,y_train)      #fit training model to created object
predict=neigh.predict(x_test)   #predict y_test results
#print(predict)
acc=sklearn.metrics.accuracy_score(y_test,predict)  #compare actual and obtained results and decide accuracy
print("Accuracy of KNN with K=3 is",acc)