import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

def X_build(col,D):
    
    """ Takes a single predictor column and a positive integer D, and creates a predictor matrix whose columns consist of powers of the entries in the original column from 0 to D.

		Args:
		
			col::[Numpy Array]
                		Single predictor column
        
	    		D::[Integer]
                		Positive integer
    
    """

    predictor_matrix = np.ones((len(col),D+1))

    for i in range(0,len(col)):
        for j in range(0,D+1):
            val = col[i]**(j)
            predictor_matrix[i][j] = val

    return predictor_matrix

def lasso_regression(data,lam,degree):

	""" Runs lasso regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy Array]
				Array holding iso, name, code, t vector, and X matrix 

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer]
				Polynomial degree to use with lasso regression

	"""

	# pull out each column from X matrix
	# run build each column into matrix
	# smash matricies together
	# this is how we use multivariate regression

	X = X_build(data[4][:,0],degree)

	for col in range(data[4].shape[1]-1):
		X = np.hstack((X,X_build(data[4][:,col+1],degree)
))

	# run lasso regression with the given weight penalty 
	# then fit model to data and return predictions

	model = Lasso(alpha=lam, max_iter=100000000, tol=1e-2)
	model.fit(X, data[3])

	return data[4][:,0], np.array(model.predict(X)) # years, predictionss 

def ridge_regression(data,lam,degree):
	
	""" Runs ridge regression with the given parameters and returns predictions corresponding to years.

		Args:

			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer]
				Polynomial degree to use with ridge regression
		
	"""
	
	# like in lasso regression, first we
	# pull out each column from X matrix
	# run build each column into matrix
	# smash matricies together
	# this is how we use multivariate regression

	X = X_build(data[4][:,0],degree)

	for col in range(data[4].shape[1]-1):
		X = np.hstack((X,X_build(data[4][:,col+1],degree)))

	## run ridge regression

	model = Ridge(alpha=1, max_iter=10000000000)
	model.fit(X, data[3])

	return data[4][:,0], np.array(model.predict(X)) # years, predictionss 

