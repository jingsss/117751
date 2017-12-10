#!/usr/bin/python

from scipy.io import arff
import numpy as np
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced,geometric_mean_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
import random
from sklearn import preprocessing
from os import listdir
from sklearn.neural_network import MLPClassifier

def read_from_arff(filename):
	with open(filename,'r') as f:
		data, meta = arff.loadarff(f)
		data_X = np.asarray([list(i)[:-1] for i in data])
		data_Y = np.asarray([int(i[-1]) for i in data])
	return data_X, data_Y, meta

def get_dataset(filename):
	X = []
	Y = []
	for i in filename:
		temp_X, temp_Y, meta = read_from_arff(i)
		X.append(temp_X)
		Y.append(temp_Y)
	return np.vstack(X), np.hstack(Y)
	
def model(clf, train_X, train_Y, test_X, test_Y, classes, name):
	clf.fit(train_X, train_Y)
	predict_test = clf.predict(test_X)
	conf_m = confusion_matrix(test_Y, predict_test, labels = classes)
	row_sum = conf_m.sum(axis = 1)
	classes_acc = np.diag(conf_m) * 1.0 / row_sum
	classes_acc = [i if i >= 0 else 0 for i in classes_acc]
	print "_________________ Using Model %s __________________\n"%(name)
	print conf_m
#	print classification_report_imbalanced(test_Y, predict_test, labels = classes)
	print "Average Accuracy : %.4f"%(np.mean(classes_acc))
	print "Precision : %.4f"%(np.mean(test_Y == predict_test))


class level_tree():
	value = None
	left = None
	right = None
	clf = None
	def __init__(self, value):
		self.value = value
	def assign_left_right(self, left_value, right_value):
		self.left = level_tree(left_value)
		self.right = level_tree(right_value)
	def print_cur_level(self):
		print self.left.value
		print self.right.value
		
def construct_tree():
	classes = [1,2,3,4,5,6,7]
	labels = ["N","J","S","F","A","C","D"]
	l = dict()
	for i in classes:
		l[labels[i - 1]] = i	
	x = level_tree(None)
	x.assign_left_right([2,3,4,5,6,7],[1])
	x.left.assign_left_right([l["A"], l["J"],l["S"]], [l["C"],l["F"],l["D"]])
	x.left.left.assign_left_right([l["A"]], [l["J"],l["S"]])
	x.left.left.left.assign_left_right([l["A"]], [l["D"]])
	x.left.left.right.assign_left_right([l["J"]], [l["S"]])
	x.left.right.assign_left_right([l["C"]], [l["F"],l["D"]])
	x.left.right.right.assign_left_right([l["F"]],[l["D"]])
	x.right.assign_left_right([l["S"]], [l["N"]])
	return x

class tree_classifier():
	X = None
	Y = None
	level = None
	def __init__(self,train_X, train_Y):
		self.level = construct_tree()
		self.X = train_X
		self.Y = train_Y
		self.generate_clf_for_level()
	def generate_classifier(self):
		l = self.level.left.value
		r = self.level.right.value
		index = []
		train_Y = []
		for i in range(len(self.Y)):
			if self.Y[i] in l:
				index.append(i)
				train_Y.append(0)
			elif self.Y[i] in r:
				index.append(i)
				train_Y.append(1)
				
		train_X = self.X[index]
		train_Y = np.asarray(train_Y)
		clf = LinearSVC()
		clf.fit(train_X, train_Y)
		return clf
		
	def generate_clf_for_level(self):
		q = [self.level]
		while len(q) > 0:
			cur = q.pop()				
			if cur.left != None:
#				cur.print_cur_level()
				cur.clf = self.generate_classifier()
				q.append(cur.left)
				q.append(cur.right)
				
	def by_level(self):
		q = [self.level]
		while len(q) > 0:
			cur = q.pop()				
			if cur.left != None:
				cur.print_cur_level()
				print cur.clf
				q.append(cur.left)
				q.append(cur.right)
	
	def predict_by_level(self, test_X):
		q = [self.level]
		predict_Y = None
		while len(q) > 0:
			cur = q.pop()
			if cur.left != None and cur.right != None:
				predict_test = cur.clf.predict(test_X)				
			if predict_test == 1:
				if cur.right != None:
					q.append(cur.right)
					if cur.right.right == None:
						return cur.right.value[0]
			else:
				if cur.left != None:
					q.append(cur.left)
					if cur.left.right == None:
						return cur.left.value[0]
				
			
def process_by_x(test, x):
	test_temp = []
	for i in test:
		if i in x:
			test_temp.append(1)
		else:
			test_temp.append(0)
	return np.asarray(test_temp)
		
	


directory = "Features/"
filenames = ["shrek_2","carstoons", "chicken_run", "ice_age", "team_america", "ice_age_a_mammoth_christmas"]
languages = ["eng", "tur", "deu", "slk", "rus", "ukr", "spa", "lit", "lav", "est"]
speed = ["slower", "Original", "faster"]

X, Y,m = read_from_arff("Features/ice_age_a_mammoth_christmas_eng_Original.arff")
X1, Y1,m = read_from_arff("Features/ice_age_a_mammoth_christmas_eng_faster.arff")
X2, Y2,m = read_from_arff("Features/ice_age_a_mammoth_christmas_eng_slower.arff")



#test_X, test_Y = get_dataset(test)
n = len(Y)
idx = range(n)
random.shuffle(idx)
test_X = X[idx[:n/5]]
test_Y = Y[idx[:n/5]]
train_X = np.vstack([X[idx[n/5:]], X1[idx[n/5:]],X2[idx[n/5:]]])
train_Y = np.hstack([Y[idx[n/5:]], Y1[idx[n/5:]],Y2[idx[n/5:]]])
print train_X.shape
print test_X.shape

##Use a part of movie to predict the other 
#train = [directory + i + "_eng_" + j + ".arff" for i in filenames[:5] for j in speed]
#test = [directory + i + "_eng_" + "Original" + ".arff" for i in filenames[5:]]
#train_X, train_Y = get_dataset(train)
#test_X, test_Y = get_dataset(test)
#print train_X.shape
#print test_X.shape
#
#
##Use 90% to train the other 
#language = ["eng", "tur", "deu"]
#language = ["eng", "tur", "rus", "ukr", "spa", "lit", "lav", "est"]
#train = [directory + "ice_age_a_mammoth_christmas" + "_" + i + "_"+ j + ".arff" for i in language for j in ["Original"]]
#train_faster = [directory + "ice_age_a_mammoth_christmas" + "_" + i+ "_" +j + ".arff" for i in language for j in ["faster"]]
#train_slower = [directory + "ice_age_a_mammoth_christmas" + "_" + i + "_"+ j + ".arff" for i in language for j in ["slower"]]
#
#X, Y = get_dataset(train)
#X1, Y1 = get_dataset(train_faster)
#X2, Y2 = get_dataset(train_slower)
#
##test_X, test_Y = get_dataset(test)
#n = len(Y)
#idx = range(n)
#random.shuffle(idx)
#test_X = X[idx[:n/5]]
#test_Y = Y[idx[:n/5]]
#train_X = np.vstack([X[idx[n/5:]], X1[idx[n/5:]],X2[idx[n/5:]]])
#train_Y = np.hstack([Y[idx[n/5:]], Y1[idx[n/5:]],Y2[idx[n/5:]]])
#print train_X.shape
#print test_X.shape


## Usinfg 7 to train 1
#language = ["eng", "tur", "rus", "ukr", "spa", "lit", "lav", "est"]
#train = [directory + "ice_age_a_mammoth_christmas" + "_" + i + "_" + j + ".arff" for i in language[1:] for j in speed]
#test =  [directory + "ice_age_a_mammoth_christmas" + "_" + i + "_" +"Original" + ".arff" for i in language[:1]]
#train_X,train_Y = get_dataset(train)
#test_X, test_Y = get_dataset(test)
#print train_X.shape
#print test_X.shape


# using all to train 

#test_pre = "ice_age_a_mammoth_christmas"
#train = []
#test = []
#for i in listdir(directory):
#	if i.endswith(".arff") and not i.startswith("."):
#		if not i.startswith(test_pre) and "eng" not in i:
#			train.append(directory + i)
#		if i.startswith(test_pre + "_eng_Original"):
#			test.append(directory + i)
#
#train_X,train_Y = get_dataset(train)
#test_X, test_Y = get_dataset(test)
#print train_X.shape
#print test_X.shape		
			


scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X) 
test_X = scaler.transform(test_X) 

classes = [1,2,3,4,5,6,7]
labels = ["N","J","S","F","A","C","D"]

#test_Y = process_by_x(test_Y, [1])
#train_Y = process_by_x(train_Y, [1])
#print np.mean(test_Y == 1)
#classes = [0, 1]
#
#tree = tree_classifier(train_X, train_Y)
#predict_y = []
#for i in test_X:
#	v = tree.predict_by_level(np.array([i]))
#	predict_y.append(v)
#predict_y = np.asarray(predict_y)
#print np.mean(predict_y == test_Y)
#conf_m = confusion_matrix(test_Y, predict_y, labels = classes)
#print conf_m
#row_sum = conf_m.sum(axis = 1)
#classes_acc = np.diag(conf_m) * 1.0 / row_sum
#print np.mean(classes_acc)
clf = LinearSVC()
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
clf = OneVsRestClassifier(LinearSVC())
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
clf = OneVsRestClassifier(RandomForestClassifier())
model(clf, train_X, train_Y, test_X, test_Y, classes, "RandomForestClassifier")
clf = OneVsRestClassifier(RidgeClassifier())
model(clf, train_X, train_Y, test_X, test_Y, classes, "RidgeClassifier")
clf = MLPClassifier(solver='lbfgs', alpha=16, hidden_layer_sizes=(12, 6), random_state=1, max_iter = 2000)
model(clf, train_X, train_Y, test_X, test_Y, classes, "MLPClassifier")

