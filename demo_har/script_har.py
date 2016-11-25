#!/usr/bin/env python

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import data_handler as dh

print '\nHuman Activity Recognition demo.'

# read data
print '\nReading data...'
X_train, X_test, y_train, y_test = dh.read_data()    

# print some stats
print '\nTraining data: %d instances, %d features' % X_train.shape
print 'Test data:     %d instances, %d features' % X_test.shape
print "Unique target labels:", np.unique(y_train[0])

# read activities
print "\nActivities:"
print dh.read_activities()

# encode target variables for multiclass learning
y_train,_ = pd.factorize(y_train[0])
y_test,_ = pd.factorize(y_test[0])

# initialize classifier
clf = RandomForestClassifier()

# train classifier
clf.fit(X_train,y_train)

# test classifier
y_pred = clf.predict(X_test)

# print results    
t = pd.crosstab(y_test, y_pred, rownames=['actual'], colnames=['predictions'])
print "\nConfusion matrix:"
print t
print "\nAccuracy of Random Forest: %.3f\n" % np.mean(y_test == y_pred)
