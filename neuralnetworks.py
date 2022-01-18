import numpy as np
import pandas as pd
import warnings

from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
def neuralnetwork(X,t):
    ### X represents the matrix for the train data and t represents the target data for the train set
    model = MLPRegresson.fit(X,t)
    print("coefs",model.coefs_)

def grid_search(cv,lams,X,seed,k,verbose):
    activation_functions = np.array(["identity", "logistic", "tanh", "relu"])
    
    #

	""" Calculates and returns the optimal D and lambda parameters after running grid search on given model.
		Args:

			cv::[Function]
				Cross validation function
			lams::[List]
				List of lambda values to test
			degrees::[List]
				List of polynomial degrees to test

			countries::[Numpy Array]
				Mutlidimensional numpy array to plug into regression algorithm

			country_idx::[Integer]
				Index of country to run regression on
			seed::[Integer]
				Seed for random number generation used in cross validation algorithm
			k::[Integer]
				Number of folds to split data into when running cross validation
			verbose::[Boolean]
				Indicates whether to display verbose output
	"""

	min_mse = lasso_kfcv(lasso_regression,countries,country_idx,k,seed,lams[0],1,verbose)
	pair = 1, lams[0]

	# find every combination of lambda and D from our given lists
	# if the average mse of a calculation is lower than our default, set it as our global min value
	# return optimal pair which minimizes mse

	for lam in lams:

		for degree in degrees:

			average_mse = cv(lasso_regression,countries,country_idx,k,seed,lam,degree,verbose)

			if(average_mse < min_mse):
				pair = degree, lam

	return pair
