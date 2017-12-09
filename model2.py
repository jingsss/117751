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
from sklearn import mixture
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

TESS_FOLDER = "tess_features/"
FEATURE_FOLDER = "Features/"
FEATURE_2S_FOLDER = "Features_2s/"

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
	train_size = 170
	for train_set in list_train_set:
		test_X1, test_Y1, meta =read_from_arff(train_set)
		if not assigned:
			test_X = test_X1[:train_size]
			test_Y = test_Y1[:train_size]
			assigned = True
		else:
			test_X = np.concatenate((test_X, test_X1[:train_size]), axis=0)
			test_Y = np.concatenate((test_Y, test_Y1[:train_size]), axis=0)
			
	return test_X, test_Y

def append_test_data(list_train_set):
	test_X = test_Y = None
	assigned = False
	test_size = 170
	for train_set in list_train_set:
		test_X1, test_Y1, meta =read_from_arff(train_set)
		if not assigned:
			test_X = test_X1[test_size:]
			test_Y = test_Y1[test_size:]
			assigned = True
		else:
			test_X = np.concatenate((test_X, test_X1[test_size:]), axis=0)
			test_Y = np.concatenate((test_Y, test_Y1[test_size:]), axis=0)
			
	return test_X, test_Y


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
def get_aff_arff(lang="", s2=False, together=False):
	
	feature_files1 = [FEATURE_FOLDER+f for f in listdir(FEATURE_FOLDER) if isfile(join(FEATURE_FOLDER, f)) and f.endswith("arff")and lang in f]
	feature_files2 = [FEATURE_2S_FOLDER+f for f in listdir(FEATURE_2S_FOLDER) if isfile(join(FEATURE_2S_FOLDER, f)) and f.endswith("arff")and lang in f]
	if not s2:
		return feature_files2
	else:
		
		if not together:
			return feature_files2
		if together:
			return feature_files1 + feature_files2


def plot_coefficients(classifier, feature_names, top_features=20):
	print len(classifier.coef_)
	coef = classifier.coef_.ravel()
	print len(coef)
	top_positive_coefficients = np.argsort(coef)[-top_features:]
	top_negative_coefficients = np.argsort(coef)[:top_features]
	top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
	# create plot
	plt.figure(figsize=(15, 5))
	colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
	plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
	feature_names = np.array(feature_names)
	plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
	plt.show()
	print top_positive_coefficients
	
classes = ['1','2','3','4','5','6','7']
labels = ["N","J","S","F","A","C","D"]
list_train_set = get_all_arff_tess("")
feature_files = get_aff_arff("eng",s2=False)
feature_files += get_all_arff_tess(person="")
print feature_files
meta = None
#train_X, train_Y = append_train_data(list_train_set)
#list_train_set = get_all_arff_tess("")
#test_X, test_Y = append_test_data(feature_files)
#train_X, train_Y,meta = read_from_arff("Features/team_america_eng_faster.arff")
all_feature_names = []

if meta:
	for i in meta:
		all_feature_names.append(i)
	print len(all_feature_names)
#all_feature_names = all_feature_names[1:-1]
train_X, train_Y = append_data(feature_files)
#train_X, train_Y = train_X[::2], train_Y[::2]
#feature_files = get_aff_arff("eng",s2=False)
#feature_files += get_all_arff_tess(person="")
#test_X, test_Y ,_ = read_from_arff("Features/team_america_eng_faster.arff")
test_X, test_Y = append_data(feature_files)
print test_X.shape
#test_X, test_Y = test_X[1::2], test_Y[1::2]
#gmm = mixture.GaussianMixture(n_components=range(7),
#                                      covariance_type=cv_type)
print train_X.shape, train_Y.shape

print test_X.shape, test_Y.shape

scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X) 
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
#train_X = sel.fit_transform(train_X)
#cv = CountVectorizer()
#cv.fit(train_X) 
#X_train = cv.transform(train_X)
#sklearn.feature_selection.f_classif 
# sklearn.feature_selection.mutual_info_classif
#test_X = scaler.transform(test_X) 
#train_X = SelectKBest(mutual_info_classif, k=80).fit_transform(train_X, train_Y)
#test_X = SelectKBest(mutual_info_classif, k=80).fit_transform(test_X, test_Y)
#test_X = sel.fit_transform(test_X) 
#rng = np.random.RandomState(42)
#clf = NearestCentroid()

clf = ExtraTreesClassifier()
#clf = IsolationForest(max_samples=100, random_state=rng)
#clf = LinearSVC()
#model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
#clf = OneVsRestClassifier(LinearSVC())
#importances = clf.feature_importances_
model(clf, train_X, train_Y, test_X, test_Y, classes, clf.__str__)
#plot_coefficients(clf, all_feature_names)
# clf = OneVsRestClassifier(RandomForestClassifier())