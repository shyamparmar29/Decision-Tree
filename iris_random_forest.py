# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 21:19:08 2019

@author: Shyam Parmar
"""

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from sklearn.model_selection import train_test_split

np.random.seed(0)

iris = load_iris()
df = pd.DataFrame(iris.data, columns = iris.feature_names) # Creating a dataframe with four feature variables
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names) # Adding a new column for the species name

X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]  # Features
y = df['species']  # Labels

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Create a Classifier
clf = RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))