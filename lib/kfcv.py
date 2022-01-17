import numpy as np
import pandas as pd
import sys

def mean_squared_errors(t,t_hat):
    
    """ Returns the mean squared error between each entry in target vector and predicted target vector.

    		Args:
			t::[Numpy array]
				Target vector

			t_bat::[Numpy Array]
				Predicted target vector
    """

    return np.square(t-t_hat).mean()

def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))

def delete_given_size(a, size):
	ret = list()
	for i in range(len(a)):
		if(len(a[i]) == size):
			ret.append(a[i])
	return np.array(ret)	

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

	col_size = len(t) // k
	print(col_size)

	x = delete_given_size(split_given_size(t,col_size),col_size)
	
	

	print(x)

	
	
def lasso_kfcv(dataset,k,seed,weight_penalty,degree,verbose):

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

	folds = split(dataset,k,seed)

	for i in range(k):
		

		#lasso_regression(countries,theta,degree,country_idx)

		# set validation and train sets

		validation = folds[i-1]
		train = pd.DataFrame()

		for j in range(len(folds)+1):
            		if(j-1 != i-1):
                		train = pd.concat([folds[j-1],train])
		
		countries = build_dataset(train,verbose)

		# use probabilities to make predictions

		classifications = list()
		num_labels = 0

		for i in probabilities:
			num_labels += 1
			classifications.append(np.argmax(i))
		
		classifications = np.array(classifications)
		
		ground_truth_classifications = list()

		first_row = validation.iloc[:,0].values

		for i in first_row:
			ground_truth_classifications.append(int(i))

		print("ground truth:",ground_truth_classifications,"pred:",classifications)

		error = misclassification_rate(classifications,ground_truth_classifications,num_labels)
		total_error += error

	return total_error/k
