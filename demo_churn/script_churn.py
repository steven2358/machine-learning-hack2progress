#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

print '\nCustomer churn prediction demo.'

# load data
data = pd.read_csv('churn.csv')

# show column names and sample data
col_names = data.columns.tolist()
print "\nColumn names:"
print col_names

to_show = col_names[:3] + col_names[-3:]
print "\nSample data:"
print data[to_show].head(6)

# define target data y
churn_result = data['Churn?']
y = np.where(churn_result == 'True.',1,0)

# convert yes/no to boolean
yes_no_cols = ["Int'l Plan","VMail Plan"]
data[yes_no_cols] = data[yes_no_cols] == 'yes'

# remove non-numeric columns and targets
to_drop = ['State','Phone','Churn?']
data = data.drop(to_drop,axis=1)

# define input data X
X = data.as_matrix().astype(np.float)

# print some stats
print "\nData contains %d observations and %d features" % X.shape
print "Unique target labels:", np.unique(y)

# split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# initialize classifier
clf = RandomForestClassifier()

# train classifier
clf.fit(X_train,y_train)

# test classifier
y_pred = clf.predict(X_test)

# print result
print "\nAccuracy of Random Forest: %.3f\n" % np.mean(y_test == y_pred)
