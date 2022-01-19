import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn import preprocessing as pre

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

			degree::[Integer]
				Polynomial degree to use with lasso regression
			
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

			degree::[Integer]
				Polynomial degree to use with ridge regression

			print_coef::[Boolean]
				Whether or not to print the model's coefficients (default is to not)
		
	"""
	
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

def qr(X):

    """ Computes Q and R for the input matrix X.

		Args:

			X::[Numpy Array]
                		Matrix for which to compute the QR.
    
    """

    return np.linalg.qr(X)


def targetvector(data,t,Xnew,lambda_value,D):
    
    """ Builds the weights for ridge regression. 

		Args:
			
			t::[Numpy Array]
				Target vector

			Xnew::[Numpy Array]
				Train matrix

			lambda_value::[Float]
				Weight penalty for high polynomial orders

			D::[Integer]
				Polynomial order for ridge function
	
    """


    X = X_build(data, D)
    
    QR = qr(X)

    

    I = np.identity(X.shape[1])
    
    #print("I:",I)

    lam = I * lambda_value
    lam[0][0] = 0
    
    
    here = np.matmul(np.linalg.inv(I + lam), QR[0].T)

    theta = np.matmul(here, t)
    
    w = np.matmul(np.linalg.inv(QR[1]), theta)
    
    return np.array(w)


def apply_ridge(data,X,t,to_predict,lam,D):
	
	""" Returns predicted values for model based on X and t.

		t::[Numpy Array]
			Target vector

		X::[Numpy Array]
			Train Matrix

		to_predict::[Numpy Array]
			Matrix to predict values for e.g. test matrix
		
		lam::[Float]
			Lambda value (weight penalty for high polynomial orders)

		D::[Integer]
			Polynomial order for ridge function

	"""

	weights = targetvector(data,t,X,lam,D)
	values = np.dot(X_build(data,D),weights) # changed to_predict -> X
	
	return values

def local_ridge_regression(data,to_predict,lam,degree,print_coef=False):
	
	""" Runs ridge regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix
			
			to_predict::[Numpy Array]
				Matrix to predict with model

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer]
				Polynomial degree to use with ridge regression

			print_coef::[Boolean]
				Whether or not to print the model's coefficients (default is to not)
		
	"""
	
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

