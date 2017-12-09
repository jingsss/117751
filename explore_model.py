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

from sklearn import preprocessing

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
	x.right.assign_left_right([l["C"]], [l["N"]])
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
				
			
				
	
		
	


directory = "Features/"
filenames = ["shrek_2","carstoons", "chicken_run", "ice_age_a_mammoth_christmas", "ice_age", "team_america"]
languages = ["eng", "tur", "deu", "slk", "rus", "ukr", "spa", "lit", "lav", "est"]
speed = ["slower", "Original", "faster"]


train = [directory + i + "_eng_" + j + ".arff" for i in filenames[:4] for j in speed]
test = [directory + i + "_eng_" + j + ".arff" for i in filenames[4:] for j in speed]

train_X, train_Y = get_dataset(train)
test_X, test_Y = get_dataset(test)
print train_X.shape
print test_X.shape

scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X) 
test_X = scaler.transform(test_X) 

classes = [1,2,3,4,5,6,7]
labels = ["N","J","S","F","A","C","D"]
l = dict()
for i in classes:
	l[labels[i - 1]] = i

tree = tree_classifier(train_X, train_Y)
predict_y = []
for i in test_X:
	v = tree.predict_by_level(np.array([i]))
	predict_y.append(v)
predict_y = np.asarray(predict_y)
print np.mean(predict_y == test_Y)
conf_m = confusion_matrix(test_Y, predict_y, labels = classes)
print conf_m
row_sum = conf_m.sum(axis = 1)
classes_acc = np.diag(conf_m) * 1.0 / row_sum
print np.mean(classes_acc)
clf = LinearSVC()
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
clf = OneVsRestClassifier(LinearSVC())
model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC(OvR)")
#clf = OneVsRestClassifier(RandomForestClassifier())
#model(clf, train_X, train_Y, test_X, test_Y, classes, "RandomForestClassifier")
#clf = OneVsRestClassifier(RidgeClassifier())
#model(clf, train_X, train_Y, test_X, test_Y, classes, "RidgeClassifier")
