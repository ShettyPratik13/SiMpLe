from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import mltools as ml
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings('ignore')


np.random.seed(0)

# Data loading
X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
X_test = np.genfromtxt("data/X_test.txt", delimiter=None)

# Train and Validation splits
#Xtr, Xval, Ytr, Yval = ml.splitData(X, Y, 0.75)

# Taking a subsample of the data so that trains faster.  You should train on whole data for homework and Kaggle.
#Xt, Yt = Xtr[:4000], Ytr[:4000]

# flatten y into a 1-D array
#Ytf = np.ravel(Yt)
Yf = np.ravel(Y)

# instantiate a logistic regression model, and fit with X and y
model = LogisticRegression()
#model = model.fit(Xt, Ytf)
model = model.fit(X, Yf)

# check the accuracy on the training set
#print(model.score(Xt, Ytf))
print(model.score(X, Yf))

# predict class labels for the test set
predicted = model.predict(X_test)
predicted = predicted.astype(int)
print(predicted)

probs = model.predict_proba(X_test)

Y_sub = np.vstack([np.arange(predicted.shape[0]), probs[:, 1]]).T


np.savetxt('Y_predicted.txt', Y_sub, '%d, %.5f', header='ID,Prob1', delimiter=',')


