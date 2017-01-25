from sklearn.datasets import load_iris
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from numpy.random import RandomState
import numpy as np

"""An unsupervised learning algorithm only uses a single set of
observations X with shape (n_samples, n_features) and does not use any
kind of labels. A supervised learning algorithm makes the distinction
between the raw observed data X with shape (n_samples, n_features) and
some label given to the model while training by some teacher. In
scikit-learn this array is often noted y and has generally the shape
(n_samples,)."""

#get the datas
iris = load_iris()
print iris.target_names
print iris.feature_names
print iris.data.shape
n_samples, n_features = iris.data.shape
X, y = iris.data, iris.target

#train the svm
clf = LinearSVC()
clf = clf.fit(X, y)
clf.coef_
clf.intercept_

#use svm model to classify new instances
X_new = [[ 5.0,  3.6,  1.3,  0.25]]
print "clf classification", clf.predict(X_new)

#linear regression model
clf2 = LogisticRegression().fit(X, y)
print "clf2 probs", clf2.predict_proba(X_new)
print "clf2 classification", clf2.predict(X_new)

#create separate training and test sets, randomise
indices = np.arange(n_samples)
RandomState(42).shuffle(indices)
X = iris.data[indices]
y = iris.target[indices]

# 2/3 training, 1/3 test
split = (n_samples * 2) / 3
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
print X_train.shape, X_test.shape
print y_train.shape, y_test.shape

clf = LinearSVC().fit(X_train, y_train)
print "classification", np.mean(clf.predict(X_test) == y_test)
