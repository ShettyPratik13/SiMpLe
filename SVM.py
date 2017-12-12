import mltools as ml
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn import svm
from sklearn import metrics

np.random.seed(0)
X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)
X,Y = ml.shuffleData(X,Y)
Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xt, Yt = Xtr[:50], Ytr[:50] #sampled training data
clf = svm.SVC(kernel='linear', C = 1.0, probability = True)
clf.fit(Xt, Yt)
Yvahat = clf.decision_function(Xva[:100])
print(metrics.roc_auc_score(Yva[:100], Yvahat, average='macro', sample_weight=None))


# correct_predictions = np.count_nonzero(Yvahat-Yva)

# Xte = np.genfromtxt('data/X_test.txt', delimiter=None)
# #learner = svm.SVC(kernel='linear', C = 1.0)
# Yte = clf.predict(Xte)
# np.savetxt('Y_submit.txt', Yte, '%d, %.2f', header='ID,Prob1', comments='', delimiter=',')