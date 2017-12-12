'''
Must install modules scikit-learn and xgboost
The X_train.csv should have a header line, and 14 feature column along with last column for output class

'''


from numpy import loadtxt
from xgboost import XGBClassifier
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def xgboost():
	from numpy import loadtxt
	from xgboost import XGBClassifier
	import xgboost
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import accuracy_score
	
	data = loadtxt('X_train.csv', skiprows = 1,delimiter=",")

	test_data = loadtxt('X_test.csv', skiprows = 1,delimiter=",")

	train_input = data[:,:-1]

	train_output = data[:,-1]
	
	xgtrain = xgboost.DMatrix(train_input, label=train_output)
	testdmat = xgboost.DMatrix(test_data)
	params = {'objective' : 'multi:softprob', 'num_class':2,'eta': 0.1,'max_depth':14}
	final_gb = xgboost.train(params, xgtrain, num_boost_round = 300)
	y_pred = final_gb.predict(testdmat)[:,1]
	pred_test = pd.DataFrame(y_pred)
	pred_test.to_csv("Y_test.txt")	
 
xgboost()
