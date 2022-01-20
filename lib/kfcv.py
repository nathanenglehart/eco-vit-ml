import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing as pre

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

	#scaler = pre.StandardScaler()
    	#X = scaler.fit_transform(X)
    	#comment out code to not standardize
	
	full = np.column_stack((t,X))
	rng = np.random.default_rng(seed)
	rng.shuffle(full)
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

		years, preds = lasso_function(np.array(train,dtype=object),validation,weight_penalty,degree)


		mse_error += mean_squared_error(validation[3],preds) # t, t_hat

		# for debugging

		if(False):
			print("D =",degree)
			print("lam =",weight_penalty)

			plt.scatter(validation[4][:,0], validation[3], color = 'g')
			plt.scatter(validation[4][:,0], preds, label="preds")
			plt.xlabel('Years')
			plt.ylabel('MLD')
			plt.show()



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

		years, preds = ridge_function(np.array(train,dtype=object),validation,weight_penalty,degree)

		mse_error += mean_squared_error(validation[3],preds) # t, t_hat

		# for debugging

		if(False):
			print("D =",degree)
			print("lam =",weight_penalty)

			plt.scatter(validation[4][:,0], validation[3], color = 'g')
			plt.scatter(validation[4][:,0], preds, label="preds")
			plt.xlabel('Years')
			plt.ylabel('MLD')
			plt.show()

	return mse_error/k

def neural_network_kfcv(neural_network_function,data,k,seed,weight_penalty,activation_function,verbose):

	""" Returns array of error statistics from run of k fold cross validation for Lasso Regression.

		Args:

			neural_network_function::[Function]
				Neural network function

			data::[Numpy array]
				Array holding iso, name, code, t vector, and X matrix

			k::[Integer]
				Number of folds to split the dataset into

			seed::[Integer]
				Seed used to randomly split the given dataset into k folds

			weight_penalty::[Float]
				Penalty for high degree polynomials in model

			activation_function::[String]
				Parameter for neutral network function

			verbose::[Boolean]
				Option to run program with verbose output

	"""

	mse_error = 0

	t_vecs, X_matricies = split_by_year(data[3],data[4],k,seed)

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

		preds = neural_network_function(train[4],train[3],validation[4],weight_penalty,activation_function)

		mse_error += mean_squared_error(validation[3],preds) # t, t_hat

		# for debugging

		if(False):
			print("D =",degree)
			print("lam =",weight_penalty)

			plt.scatter(validation[4][:,0], validation[3], color = 'g')
			plt.scatter(validation[4][:,0], preds, label="preds")
			plt.xlabel('Years')
			plt.ylabel('MLD')
			plt.show()

	return mse_error/k


