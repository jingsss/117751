#!/usr/bin/python
from scipy.io import arff
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA

#from imblearn.metrics import classification_report_imbalanced,geometric_mean_score

from sklearn import preprocessing
import random
def read_from_arff(filename):
	with open(filename,'r') as f:
		data, meta = arff.loadarff(f)
		data_X = np.asarray([list(i)[:-1] for i in data])
		data_Y = np.asarray([i[-1] for i in data])
		n = len(data_Y)
		idx = []
		for i in range(n):
			if data_Y[i] != '3':
				idx.append(i)
		data_X = data_X[idx]
		data_Y = data_Y[idx]
	return data_X, data_Y, meta
	
def model(clf, train_X, train_Y, test_X, test_Y, classes, name):
	clf.fit(train_X, train_Y)
	predict_test = clf.predict(test_X)
	conf_m = confusion_matrix(test_Y, predict_test, labels = classes)
	row_sum = conf_m.sum(axis = 1)
	classes_acc = np.diag(conf_m) * 1.0 / row_sum
	classes_acc = [i for i in classes_acc if i >= 0]
	print "_________________ Using Model %s __________________\n"%(name)
	print conf_m
	#print classification_report_imbalanced(test_Y, predict_test, labels = classes)
	print "Average Accuracy : %.4f"%(np.mean(classes_acc))
	print "Precision : %.4f"%(np.mean(test_Y == predict_test))
	

	
	
classes = ['1','2','4','5','6','7']
labels = ["N","J","F","A","C","D"]
train_X, train_Y, meta = read_from_arff('Features_db/emodb.arff')
test_X, test_Y, meta =read_from_arff('Features_db/emovo.arff')

X, Y, meta =read_from_arff('Features_db/emodb.arff')

n = len(Y)
idx = range(n)
print len(idx)
random.seed(1)
random.shuffle(idx)
train_X = X[idx[n / 5:]]
train_Y = Y[idx[n / 5:]]
test_X = X[idx[:n / 5]]
test_Y = Y[idx[:n / 5]]

scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X) 
test_X = scaler.transform(test_X) 


clf = OneVsRestClassifier(LinearSVC())
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
clf = OneVsRestClassifier(RandomForestClassifier())
model(clf, train_X, train_Y, test_X, test_Y, classes, "RandomForestClassifier(OvR)")
clf = OneVsRestClassifier(RidgeClassifier())
model(clf, train_X, train_Y, test_X, test_Y, classes, "RidgeClassifier(OvR)")
clf = MLPClassifier(solver='lbfgs', alpha=16,hidden_layer_sizes=(12, 6), random_state=1, max_iter = 500, early_stopping=True)
model(clf, train_X, train_Y, test_X, test_Y, classes, "MLPClassifier")





