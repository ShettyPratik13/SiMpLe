# Sample Multilayer Perceptron Neural Network in Keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
# load and prepare the dataset
X = np.genfromtxt("data/X_train.txt", delimiter=None)
Y = np.genfromtxt("data/Y_train.txt", delimiter=None)
X_test = np.genfromtxt("data/X_test.txt", delimiter=None)

# Taking a subsample of the data so that trains faster.  You should train on whole data for homework and Kaggle.
#X, Y = X[:5000], Y[:5000]

# 1. define the network
model = Sequential()
model.add(Dense(30, input_dim=14, activation='tanh'))
model.add(Dense(30, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))
# 2. compile the network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# 3. fit the network
history = model.fit(X, Y, epochs=100, batch_size=10)
# 4. evaluate the network
loss, accuracy = model.evaluate(X, Y)
print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))
# 5. make predictions
predicted_pro = model.predict(X)
predicted = predicted_pro.astype(int)
accuracy = np.mean(predicted == Y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))

probs = model.predict(X_test)
Y_sub = np.vstack([np.arange(probs.shape[0]), probs[:, 0]]).T
np.savetxt('Y_predicted.txt', Y_sub, '%d, %.5f', header='ID,Prob1', delimiter=',')



