#!/usr/bin/python
from scipy.io import arff
import numpy as np
from os import listdir
from os.path import isfile,join
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
#from imblearn.metrics import classification_report_imbalanced,geometric_mean_score

from sklearn import preprocessing
import random
def read_from_arff(filename):
	with open(filename,'r') as f:
		data, meta = arff.loadarff(f)
		data_X = np.asarray([list(i)[1:-1] for i in data])
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
	conf_m = conf_m * 1.0 /row_sum[:,np.newaxis]
	conf_m = np.nan_to_num(conf_m)
	df_cm = pd.DataFrame(conf_m, index = ["N","J","F","A","C","D"],
			  columns = ["N","J","F","A","C","D"])
	plt.figure(figsize = (10,7))
	sn.heatmap(df_cm, annot=True)
	plt.show()
#	print "_________________ Using Model %s __________________\n"%(name)
	print conf_m
	print "Average Accuracy : %.4f"%(np.mean(classes_acc))
	print "Precision : %.4f"%(np.mean(test_Y == predict_test))
	return np.mean(classes_acc), np.mean(test_Y == predict_test)
	
def append_data(list_train_set):
	test_X = test_Y = None
	assigned = False
	for train_set in list_train_set:
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
	
TESS_FOLDER = "Features_tess_db/"
classes = ['1','2','4','5','6','7']
labels = ["N","J","F","A","C","D"]
temp = get_all_arff_tess()

X1, Y1, meta = read_from_arff('Features_db/emodb_m.arff')
X2, Y2, meta =read_from_arff('Features_db/emodb_f.arff')
train_X, train_Y= append_data(get_all_arff_tess("older"))
test_X,test_Y = append_data(get_all_arff_tess("younger"))

legend = ["Italian", "German", "English"]
#X = np.vstack([X1, X2, X3])
#Y = np.hstack([Y1, Y2, Y3])

#n = len(Y)
#idx = range(n)
#print len(idx)
#random.seed(1)
#random.shuffle(idx)
#train_X = X[idx[n / 5:]]
#train_Y = Y[idx[n / 5:]]
#test_X = X[idx[:n / 5]]
#test_Y = Y[idx[:n / 5]]
#avg_a = []
#a = []
#for i in range(3):
#	for j in range(3):
#		if i!= j:
#			print legend[i], legend[j]
#			train_X = X[i]
#			train_Y = Y[i]
#			test_X = X[j]
#			test_Y = Y[j]
#			#	clf = MLPClassifier(solver='lbfgs', alpha=16,hidden_layer_sizes=(12, 6), random_state=1, max_iter = 500, early_stopping=True)
#			#	a1, b1 = model(clf, train_X, train_Y, test_X, test_Y, classes, "MLPClassifier")
#			scaler = preprocessing.StandardScaler().fit(train_X)
#			train_X = scaler.transform(train_X) 
#			test_X = scaler.transform(test_X)
#			
#			clf = OneVsRestClassifier(LinearSVC(random_state = 0))
#			a1,b1 = model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
#	X_temp = [X[j] for j in range(3) if j != i ]
#	Y_temp = [Y[j] for j in range(3) if j != i ]
#	X_temp = np.vstack(X_temp)
#	Y_temp = np.hstack(Y_temp)
##	X_temp = X[i]
##	Y_temp = Y[i]
#	n = len(Y_temp)
#	idx = range(n)
#	print len(idx)
#	random.seed(1)
#	random.shuffle(idx)
#	train_X = X_temp[idx[n / 5:]]
#	train_Y = Y_temp[idx[n / 5:]]
#	test_X = X_temp[idx[:n / 5]]
#	test_Y = Y_temp[idx[:n / 5]]
#

##	clf = MLPClassifier(solver='lbfgs', alpha=16,hidden_layer_sizes=(12, 6), random_state=1, max_iter = 500, early_stopping=True)
##	a1, b1 = model(clf, train_X, train_Y, test_X, test_Y, classes, "MLPClassifier")
#	clf = OneVsRestClassifier(LinearSVC(random_state = 0))
#	a1,b1 = model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
#	avg_a.append(a1)
#	a.append(b1)
#		
#print np.mean(avg_a)
#print np.mean(a)
#scaler = preprocessing.StandardScaler().fit(train_X)
#train_X = scaler.transform(train_X) 
#test_X = scaler.transform(test_X) 
#
clf = OneVsRestClassifier(LinearSVC(random_state = 0))
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
#clf = OneVsRestClassifier(RandomForestClassifier())
#model(clf, train_X, train_Y, test_X, test_Y, classes, "RandomForestClassifier(OvR)")
#clf = OneVsRestClassifier(RidgeClassifier())
#model(clf, train_X, train_Y, test_X, test_Y, classes, "RidgeClassifier(OvR)")
clf = MLPClassifier(solver='lbfgs', alpha=16,hidden_layer_sizes=(12, 6), random_state=1, max_iter = 500, early_stopping=True)
model(clf, train_X, train_Y, test_X, test_Y, classes, "MLPClassifier")






