import pylab, numpy
from numpy.random import RandomState

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import cross_validation
from sklearn.naive_bayes import GaussianNB
import pybrain



results = pylab.loadtxt('credit.txt')
target = results[:,-1]
data = numpy.delete(results,-1,1)

n_samples, n_features = data.shape
print "samples", n_samples, "features",n_features,"targets", len(target)

#Manually create separate training and test sets, randomise
#indices = numpy.arange(n_samples)
#RandomState(42).shuffle(indices)
#X = data[indices]
#y = target[indices]
# 2/3 training, 1/3 test
#split = (n_samples * 2) / 3
#X_train, X_test = X[:split], X[split:]
#y_train, y_test = y[:split], y[split:]
#clf = svm.LinearSVC().fit(X_train, y_train)
#print "classification", numpy.mean(clf.predict(X_test) == y_test)

#'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'
classifiers = {'linsvm':SVC(kernel='linear', C=1),
               'rbfsvm':SVC(kernel='rbf', C=1),
               'sigsvm':SVC(kernel='sigmoid',C=1),
               'polysvm':SVC(kernel='poly', C=1),
               'nbayes':GaussianNB(),
               'cart':DecisionTreeClassifier(),
               'knn':KNeighborsClassifier(10)}


for key in classifiers:
    clf = classifiers[key]
    scores = cross_validation.cross_val_score(clf, data, target, cv=10)
    print key, "accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std())

#skf = cross_validation.StratifiedKFold(target, 2)
#print skf
#for train, test in skf:
#print train, test

