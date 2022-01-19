import numpy as np
import pandas as pd

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

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
	rng = np.random.default_rng(seed)
	rng.shuffle(full)
	#np.random.shuffle(full)
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
	
	#print("t:",t_vecs)
	#print("X:",X_matricies)
	
	return t_vecs, X_matricies
	
def lasso_kfcv(lasso_function,data,k,seed,weight_penalty,degree,verbose):

	""" Returns array of error statistics from run of k fold cross validation for Lasso Regression. 
		
		Args:
			
			lasso_function::[Function]
				Lasso regression function
		
			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix 
			
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

	t_vecs, X_matricies = split_by_year(data[3],data[4],k,seed)

	for i in range(1,k+1):
		
		# set validation and train sets

		validation = ['','','',[],[]] # folds[i-1]
		train = ['','','',[],[]] 

		validation[3].append(np.array(t_vecs[i-1]))
		validation[4].append(np.array(X_matricies[i-1]))
		

		for j in range(1,k+1):
			if(j-1 != i-1):
				
				train[3].append(t_vecs[j-1])
				train[4].append(X_matricies[j-1])
	
		train[4] = np.concatenate(np.array(train[4]))
		train[3] = np.concatenate(np.array(train[3]))
		
		validation[4] = np.concatenate(np.array(validation[4]))
		validation[3] = np.concatenate(np.array(validation[3]))

		year, preds = lasso_function(np.array(train,dtype=object),validation,weight_penalty,degree)

		mse_error += mean_squared_error(validation[3],preds) # t, t_hat
		
	return mse_error/k
	
def ridge_kfcv(ridge_function,data,k,seed,weight_penalty,degree,verbose):

	""" Returns array of error statistics from run of k fold cross validation for Lasso Regression. 
		
		Args:
			
			ridge_function::[Function]
				Ridge regression function
		
			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix 
			
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

	t_vecs, X_matricies = split_by_year(data[3],data[4],k,seed)

	#print("\nrun\n")

	for i in range(1,k+1):
		
		# set validation and train sets

		validation = ['','','',[], []] # folds[i-1]
		train = ['','','',[],[]] 

		validation[3].append(np.array(t_vecs[i-1]))
		validation[4].append(np.array(X_matricies[i-1]))


		for j in range(1,k+1):
			if(j-1 != i-1):
				train[3].append(t_vecs[j-1])
				train[4].append(X_matricies[j-1])
	
		train[4] = np.concatenate(np.array(train[4]))
		train[3] = np.concatenate(np.array(train[3]))

		validation[4] = np.concatenate(np.array(validation[4]))
		validation[3] = np.concatenate(np.array(validation[3]))
		
		#print("train:",len(train[4]))
		#print(train[4])
		#print("validation:",len(validation[4]))
		#print(validation[4])

		year, preds = ridge_function(np.array(train,dtype=object),validation,weight_penalty,degree)

		mse_error += mean_squared_error(validation[3],preds) # t, t_hat

	return mse_error/k
