#!/usr/bin/python
from scipy.io import arff
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced,geometric_mean_score

from sklearn import preprocessing

def read_from_arff(filename):
	with open(filename,'r') as f:
		data, meta = arff.loadarff(f)
		data_X = np.asarray([list(i)[:-1] for i in data])
		data_Y = np.asarray([i[-1] for i in data])
	return data_X, data_Y, meta
	
def model(clf, train_X, train_Y, test_X, test_Y, classes, name):
	clf.fit(train_X, train_Y)
	predict_test = clf.predict(test_X)
	conf_m = confusion_matrix(test_Y, predict_test, labels = classes)
	row_sum = conf_m.sum(axis = 1)
	classes_acc = np.diag(conf_m) * 1.0 / row_sum
	print "_________________ Using Model %s __________________\n"%(name)
	print conf_m
	print classification_report_imbalanced(test_Y, predict_test, labels = classes)
	print "Average Accuracy : %.4f"%(np.mean(classes_acc))
	print "Precision : %.4f"%(np.mean(test_Y == predict_test))
	
classes = ['1','2','3','4','5','6','7']
labels = ["N","J","S","F","A","C","D"]
test_X, test_Y, meta =read_from_arff('Features/test.arff')
train_X, train_Y, meta = read_from_arff('Features/train.arff')

scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X) 
test_X = scaler.transform(test_X) 


clf = LinearSVC()
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
clf = OneVsRestClassifier(LinearSVC())
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
clf = OneVsRestClassifier(RandomForestClassifier())







