# -*- coding: utf-8 -*-
"""
Created on Sat Mar  3 13:21:01 2018

@author: admin
"""

import numpy as np #libraries for arrays
import pandas as pd #for data handling
from sklearn import cross_validation, tree, preprocessing #for data sampling, modelling & preprocessing

path = "D:\Python\Titanic_train.csv"
g_path = "D:\Python"
titanic_df = pd.read_csv(path) #read data
titanic_df.shape #to find the range of dataset
titanic_df.head() #print few data

""" Data Exploration & Processing """
titanic_df['Survived'].mean()

titanic_df.groupby('Pclass').mean()

class_sex_grouping = titanic_df.groupby(['Pclass', 'Sex']).mean()
class_sex_grouping

class_sex_grouping['Survived'].plot.bar()


group_by_age = pd.cut(titanic_df["Age"], np.arange(0,90,10))
age_grouping = titanic_df.groupby(group_by_age).mean()
age_grouping['Survived'].plot.bar()


titanic_df.count()

titanic_df = titanic_df.drop(['Cabin'], axis=1)

titanic_df = titanic_df.dropna()

titanic_df.count()


""" Data PreProcessing function """
def preprocess_titanic_df(df):
    processed_df = df.copy()
    le = preprocessing.LabelEncoder()
    processed_df.Sex = le.fit_transform(processed_df.Sex)
    processed_df.Embarked = le.fit_transform(processed_df.Embarked)
    processed_df = processed_df.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
    return processed_df

processed_df = preprocess_titanic_df(titanic_df)

X = processed_df.drop(['Survived'], axis = 1).values #Features dataset
y = processed_df['Survived'].values #Target Variables


#Train _Test Split

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2,random_state=50)


# Model implementation
clf_dt = tree.DecisionTreeClassifier(max_depth=10) # Define Model
clf_dt.fit(X_train, y_train) # fir the model with ur data
predictions = clf_dt.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)



with open(g_path + "\decisionTree.txt", "w") as f:
    f = tree.export_graphviz(clf_dt, out_file=f)
    
