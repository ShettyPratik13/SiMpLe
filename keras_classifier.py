#import dataframe as dataframe
import numpy as np
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# fix random seed for reproducibility
seed = 1
np.random.seed(seed)


# load dataset
X_dataframe = pandas.read_csv("data/X_train*.csv", header=None)
X_dataset = X_dataframe.values
X = X_dataset.astype(float)
Y_dataframe = pandas.read_csv("data/Y_train*.csv", header=None)
Y_dataset = Y_dataframe.values
Y = Y_dataset.astype(int)
Y = Y.ravel()

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(30, input_dim=14, kernel_initializer='normal', activation='tanh'))
	model.add(Dense(30, kernel_initializer='normal', activation='tanh'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, nb_epoch=100, batch_size=5, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

X_test_dataframe = pandas.read_csv("data/X_test*.csv", header=None)
X_test = X_test_dataframe.values
# predict class labels for the test set
predicted = estimator.predict(X_test)
predicted = predicted.astype(int)


Y_sub = np.vstack([np.arange(predicted.shape[0]), predicted[:, 0]]).T

np.savetxt('Y_predicted.txt', Y_sub, '%d, %.5f', header='ID,Prob1', delimiter=',')


