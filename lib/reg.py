import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing as pre

from lib.utils import qr
from lib.utils import X_build
from lib.local_reg_class import LocalRidge

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

def lasso_regression(data,to_predict,lam,degree,print_coef=False):

	""" Runs lasso regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy Array]
				Array holding iso, name, code, t vector, and X matrix 

			to_predict::[Numpy Array]
				Matrix to predict with model

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer list]
				List of polynomial degree to use with lasso regression
			
			print_coef::[Boolean]
				Whether or not to print the model's coefficients

	"""

	# pull out each column from X matrix
	# run build each column into matrix
	# smash matricies together
	# this is how we use multivariate regression

	X = X_build(data,degree)
	
	# run lasso regression with the given weight penalty 
	# then fit model to data and return predictions

	model = Lasso(alpha=lam, max_iter=100000000, tol=1e-2)
	model.fit(X, data[3])
	
	# optional argument

	if(print_coef):
		print("\nmodel coefficients")
		print(model.coef_,"\n")
	
	return data[4][:,0], np.array(model.predict(X_build(to_predict,degree))) # years, predictionss

def ridge_regression(data,to_predict,lam,degree,print_coef=False):
	
	""" Runs ridge regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix
			
			to_predict::[Numpy Array]
				Matrix to predict with model

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer list]
				List of polynomial degree to use with ridge regression

			print_coef::[Boolean]
				Whether or not to print the model's coefficients (default is to not)
		
	"""

	# standardize by taking the QR decomposition of X before passing it through the input
	# need to do the same for the matrix that we will be predicting, to_predict, as well
	# uses numpy.linalg - qr

	QR = qr(data[4])
	Q = QR[0]
	#data[4] = Q

	QR = qr(to_predict[4])
	Q = QR[0]
	#to_predict[4] = Q

	# otherwise we can also use preprocessing instead of qr decomposition

	scaler = pre.StandardScaler()
	data[4] = scaler.fit_transform(data[4])
	to_predict[4] = scaler.fit_transform(to_predict[4])

	# like in lasso regression, first we
	# pull out each column from X matrix
	# run build each column into matrix
	# smash matricies together
	# this is how we use multivariate regression

	X = X_build(data,degree)

	## run ridge regression

	model = Ridge(alpha=lam, max_iter=10000000000, solver='svd', fit_intercept=False)
	model.fit(X, data[3])
	
	# optional argument

	if(print_coef):
		print("model coefficients")
		print(model.coef_,"\n")

	return data[4][:,0], np.array(model.predict(X_build(to_predict,degree))) # years, predictionss 

def neural_network(X,t,X_test,weight_penalty,activation_function):
    
    """ Returns predictions for mlp regression neural network.

    		Args:

			X::[Numpy Array]
				Train matrix

			t::[Numpy Array]
				Target vector

			X_test::[Numpy Array]
				Matrix to return predictions for

			weight_penalty::[Float]
				Penalty for large coefficients
			
			activation_function::[Function]
				Non-linear function used for neural networks

    """

    ### X represents the matrix for the train data and t represents the target data for the train set
    ### In order to run we must scale the parameters using StandardScaler which standardizes using Zscore
    
    np.random.seed(0)
    scaler = pre.StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_test = scaler.fit_transform(X_test)
    model = MLPRegressor(solver='lbfgs', hidden_layer_sizes=50, max_iter=100000, learning_rate='constant',activation=activation_function,alpha=weight_penalty)
    model.fit(X_train_scaled,t)
    
    #print("coefs",model.coefs_)

    return model.predict(X_test)

def local_ridge_regression(data,to_predict,lam,degree,print_coef=False):
	
	""" Runs ridge regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix
			
			to_predict::[Numpy Array]
				Matrix to predict with model

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer list]
				List of polynomial degree to use with ridge regression

			print_coef::[Boolean]
				Whether or not to print the model's coefficients (default is to not)
		
	"""
	

	# standardize by taking the QR decomposition of X before passing it through the input
	# need to do the same for the matrix that we will be predicting, to_predict, as well
	# uses numpy.linalg - qr

	QR = qr(data[4])
	Q = QR[0]
	#data[4] = Q # seems to throw off fit

	QR = qr(to_predict[4])
	Q = QR[0]
	#to_predict[4] = Q

	# otherwise we can also use preprocessing instead of qr decomposition

	scaler = pre.StandardScaler()
	data[4] = scaler.fit_transform(data[4])
	to_predict[4] = scaler.fit_transform(to_predict[4])

	# like in lasso regression, first we
	# pull out each column from X matrix
	# run build each column into matrix
	# smash matricies together
	# this is how we use multivariate regression

	X = X_build(data,degree)
	


	## run ridge regression

	model = LocalRidge(lam=lam)
	model.fit(X, data[3])
	
	return data[4][:,0], np.array(model.predict(X_build(to_predict,degree))) # years, predictionss 

