import numpy as np
import pandas as pd
import sys

def mean_squared_error(t,t_hat):
    
    """ Returns the mean squared error between each entry in target vector and predicted target vector.

    		Args:
			t::[Numpy array]
				Target vector

			t_bat::[Numpy Array]
				Predicted target vector
    """

    return np.square(t-t_hat).mean()

def split_by_year(t,X,k,seed):
	
	""" Returns numpy array holding train and validation X matricies and t vectors
		
		Args:
			
			t::[Numpy Array]
				Target vector to split into k individual vectors

			X::[Numpy Array]
				Train matrix to split into k individual matricies

			k::[Integer]
				Number of folds to split data into (<= years)

			seed::[Integer]
				Seed for random state generation
	
	"""

	full = np.column_stack((t,X))
	np.random.shuffle(full)
	n = (len(full) - (k * (len(full) // k)))
	a = np.split(full[:-n,:],k)
	
	t_vecs = list()
	X_matricies = list()

	for i in a:

		t_entry = list()
		X_entry = list()

		for j in i:
			t_entry.append(j[0])
			X_entry.append(j[1:])
			
		t_vecs.append(t_entry)
		X_matricies.append(X_entry)
	
	return t_vecs, X_matricies
	
def lasso_kfcv(func,countries,country_idx,k,seed,weight_penalty,degree,verbose):

	""" Returns array of error statistics from run of k fold cross validation for Lasso Regression. 
		
		Args:
			
			dataset::[Pandas Dataframe]
				Dataset used to compute error statistics on using given parameter and classifier

			k::[Integer]
				Number of folds to split the dataset into

			seed::[Integer]
				Seed used to randomly split the given dataset into k folds
			
			weight_penalty::[Float]
				Penalty for high degree polynomials in model

			degree::[Integer]
				Polynomial degree for model
			
			verbose::[Boolean]
				Option to run program with verbose output
		
	"""

	mse_error = 0

	t_vecs, X_matricies = split_by_year(countries[country_idx][3],countries[country_idx][4],k,seed) 

	for i in range(k):
		
		# set validation and train sets

		validation = t_vecs[i-1], X_matricies[i-1] # folds[i-1]
		train = list(), list() 

		for j in range(k+1):
			if(j-1 != i-1):
				train[0].append(t_vecs[j-1])
				train[1].append(X_matricies[j-1])

		year, preds = func(countries,weight_penalty,degree,country_idx)

		cda = countries[country_idx][3]

		mse_error += mean_squared_error(cda,preds)

	return mse_error/k
