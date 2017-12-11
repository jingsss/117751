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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals.six.moves import xrange
from sklearn.mixture import GMM
from sklearn import datasets
from sklearn.cross_validation import StratifiedKFold
import matplotlib as mpl

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

def make_ellipses(gmm, ax):
    for n, color in enumerate('rgb'):
        v, w = np.linalg.eigh(gmm._get_covars()[n][:2, :2])
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        v *= 9
        ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
                                  180 + angle, color=color)
        ell.set_clip_box(ax.bbox)
        ell.set_alpha(0.5)
        ax.add_artist(ell)


def guassian(X_train, y_train, X_test, y_test):
    y_train = np.array([int(i) for i in y_train])
    y_test = np.array([int(i) for i in y_test])
    n_classes = len(np.unique(y_train))
    #print n_classes

    # Try GMMs using different types of covariances.
    classifiers = dict((covar_type, GMM(n_components=n_classes,
                        covariance_type=covar_type, init_params='wc', n_iter=20))
                       for covar_type in ['spherical', 'diag', 'tied', 'full'])
    n_classifiers = len(classifiers)

    plt.figure(figsize=(3 * n_classifiers / 2, 6))
    plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
                    left=.01, right=.99)

    best_model_acc = 0
    best_prediction = None
    for index, (name, classifier) in enumerate(classifiers.items()):
        # Since we have class labels for the training data, we can
        # initialize the GMM parameters in a supervised manner.
        classifier.means_ = np.array([X_train[y_train == i+1].mean(axis=0)
                                      for i in xrange(n_classes)])
        print classifier.means_.shape
        print X_train.shape

        # Train the other parameters using the EM algorithm.
        classifier.fit(X_train)

        h = plt.subplot(2, n_classifiers / 2, index + 1)
        make_ellipses(classifier, h)


        y_train_pred = classifier.predict(X_train)
        train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
        plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
                 transform=h.transAxes)

        y_test_pred = classifier.predict(X_test)
        print "y test pred:",y_test_pred
        print "y label:", y_test
        # y_test = np.array([str(i) for i in y_test])
        test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
        print "test accuracy:",test_accuracy
        labels = np.unique(y_train)
        if test_accuracy > best_model_acc:
            best_model_acc = test_accuracy
            best_prediction = y_test_pred
    return best_prediction, best_model_acc

        

classes = ['1','2','3','4','5','6','7']
labels = ["N","J","S","F","A","C","D"]
list_train_set = get_all_arff_tess("")
feature_files = get_aff_arff("eng",s2=False)
#feature_files += get_all_arff_tess(person="")
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
train_X, train_Y = train_X[::2], train_Y[::2]
#feature_files = get_aff_arff("eng",s2=False)
#feature_files += get_all_arff_tess(person="")
#test_X, test_Y ,_ = read_from_arff("Features/team_america_eng_faster.arff")
test_X, test_Y = append_data(feature_files)
print test_X.shape
test_X, test_Y = test_X[-20:-5:], test_Y[-20:-5:]
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
clf = KNeighborsClassifier(n_neighbors=3)
#clf = ExtraTreesClassifier()
#clf = IsolationForest(max_samples=100, random_state=rng)
#clf = LinearSVC()
#model(clf, train_X, train_Y, test_X, test_Y, classes, "LinearSVC")
#clf = OneVsRestClassifier(LinearSVC())
#importances = clf.feature_importances_
#model(clf, train_X, train_Y, test_X, test_Y, classes, clf.__str__)
y_test_predict, best_model_acc = guassian(train_X, train_Y, test_X, test_Y)
print "==============="
print best_model_acc
print test_Y
y_test_predict = [str(i) for i in y_test_predict]
print y_test_predict
conf_m = confusion_matrix(test_Y, y_test_predict, labels = classes)
row_sum = conf_m.sum(axis = 1)
for i in range(len(row_sum)):
    if row_sum[i] == 0:
        row_sum[i] = 1
classes_acc = np.diag(conf_m) * 1.0 / row_sum
print conf_m



# iris = datasets.load_iris()
# skf = StratifiedKFold(iris.target, n_folds=4)
# # Only take the first fold.
# train_index, test_index = next(iter(skf))
# X_train = iris.data[train_index]
# y_train = iris.target[train_index]
# X_test = iris.data[test_index]
# y_test = iris.target[test_index]
# print X_train.shape, "X_train"
# print y_train.shape, "X_train"
#guassian(X_train, y_train, X_test, y_test)
#plot_coefficients(clf, all_feature_names)
# clf = OneVsRestClassifier(RandomForestClassifier())