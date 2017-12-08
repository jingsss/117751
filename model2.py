#!/usr/bin/python
from scipy.io import arff
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from os import listdir
from os.path import isfile, join

TESS_FOLDER = "tess_features/"

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
	print "Average Accuracy : %.4f"%(np.mean(classes_acc))
	print "Precision : %.4f"%(np.mean(test_Y == predict_test))

def append_train_data(list_train_set):
	test_X = test_Y = None
	assigned = False
	for train_set in list_train_set:
		#print train_set
		test_X1, test_Y1, meta =read_from_arff(train_set)
		if not assigned:
			test_X = test_X1
			test_Y = test_Y1
			assigned = True
		else:
			test_X = np.concatenate((test_X, test_X1), axis=0)
			test_Y = np.concatenate((test_Y, test_Y1), axis=0)
			
	return test_X, test_Y

def get_all_arff_tess(person=""):
	tess_files = [TESS_FOLDER+f for f in listdir(TESS_FOLDER) if isfile(join(TESS_FOLDER, f)) and f.endswith("arff") and person in f]
	return tess_files

	
classes = ['1','2','3','4','5','6','7']
labels = ["N","J","S","F","A","C","D"]
list_train_set = get_all_arff_tess("young")
train_X, train_Y = append_train_data(list_train_set)
list_train_set = get_all_arff_tess("old")
test_X, test_Y = append_train_data(list_train_set)

# scaler = preprocessing.StandardScaler().fit(train_X)
# train_X = scaler.transform(train_X) 
# test_X = scaler.transform(test_X) 



#clf = LinearSVC()
#odel(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
clf = OneVsRestClassifier(LinearSVC())
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
# clf = OneVsRestClassifier(RandomForestClassifier())