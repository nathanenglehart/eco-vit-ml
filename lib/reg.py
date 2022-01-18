import numpy as np
import pandas as pd

from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from lib.utils import X_build

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

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

	X = X_build(data,degree)
	
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

	X = X_build(data,degree)

	## run ridge regression

	model = Ridge(alpha=lam, max_iter=10000000000)
	model.fit(X, data[3])

	return data[4][:,0], np.array(model.predict(X)) # years, predictionss 

