import itertools

import numpy as np
import pandas as pd

#get_all_degree_combinations(degrees, k):

from lib.utils import get_all_degree_combinations
from lib.utils import plot
from lib.utils import all_world_countries
from lib.utils import plot_reg
from lib.utils import build_world
from lib.utils import build_dataset
from lib.utils import load_dataset

from lib.kfcv import lasso_kfcv
from lib.reg import lasso_regression

from lib.kfcv import ridge_kfcv
from lib.reg import ridge_regression

from lib.reg import local_ridge_regression

from lib.reg import neural_network
from lib.kfcv import neural_network_kfcv

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

def grid_search(cv,reg,lams,degrees,data,seed,k,verbose,nn_output=False):

	""" Calculates and returns the optimal D and lambda parameters after running grid search on given model.

		Args:

			cv::[Function]
				Cross validation function

			reg::[Function]
				Regression function i.e. lasso or ridge

			lams::[List]
				List of lambda values to test

			degrees::[List]
				List of polynomial degrees to test

			data::[Numpy Array]
				Array holding iso, name, code, t vector, and X matrix

			seed::[Integer]
				Seed for random number generation used in cross validation algorithm

			k::[Integer]
				Number of folds to split data into when running cross validation

			verbose::[Boolean]
				Indicates whether to display verbose output

	"""

	if(verbose):
		print("grid search")

	min_mse = cv(reg,data,k,seed,lams[0],degrees[0],verbose)
	pair = degrees[0], lams[0]

	# find every combination of lambda and D from our given lists
	# if the average mse of a calculation is lower than our default, set it as our global min value
	# return optimal pair which minimizes mse

	for lam in lams:

		for degree in degrees:

			average_mse = cv(reg,data,k,seed,lam,degree,verbose)

			if(verbose):
				if(nn_output):
					print("activation function =",degree,"lam =",lam,"average mse:",average_mse)
				else:
					print("D =",degree,"lam =",lam,"average mse:",average_mse)

			if(average_mse < min_mse):
				pair = degree, lam
				min_mse = average_mse

	if(verbose):
		print("")

	return pair

def driver(verbose,mode,country_names,seed,k):

	""" Drives each machine learning method we use based on parameter inputs.

		Args:

			verbose::[Boolean]
				Determines whether to run algorithms with verbose output

			mode::[Integer]
				Determines which method to use i.e. 1 for lasso regression, 2 for neural networks

			country_names::[Numpy array]
				Array of country names (strings) for which to load

			country_indicies::[Numpy array]
				Array of indicies of countries for which to run driver on

			seed::[Integer]
				Seed for random number generation used in cross validation

			k::[Integer]
				Number of folds to split the data into

	"""

	# create single dataset

	dataset = load_dataset()
	countries = build_dataset(dataset,country_names,verbose)

	# indexing works as follows
	# countries[0] gives the first country from our list of selected countries in alphabetical order
	# e.g. if our list contains Australia, Brazil, and Columbia, countries[0] would give Australia
	# countries[0][1] gives the iso for the first country and countries[0][2] gives the country name
	# for the above example [0][1] would give AUS and [0][2] would give Austrialia
	# countries[0][3] gives the target vector, t, in the form of an np array (this array should be a column)
	# countries[0][4] gives the train martix, X, in the form of an np array of np arrays (each array is a column)
	# including a demo below

	#if verbose:
	#	print("demo:",countries[0][1]) # country iso
	#	print("len(t):",len(countries[0][3])) # target vector i.e. cda
	#	print("len(X):",np.size(countries[0][4],1)) # number of features
	#	print("len(x1):",len(countries[0][4][:,0])) # year
	#	print("len(x2):",len(countries[0][4][:,1])) # grl (grassland)
	#	print("len(x3):",len(countries[0][4][:,2])) # tcl (tree cover)
	#	print("len(x4):",len(countries[0][4][:,3])) # wtl (wetland)

	# for our purposes, we actually want to consolidate all of this data into
	# a singular t vector and X matrix for the world

	t, X = build_world(countries)
	data = np.array(['0','world','WOR',t,X],dtype=object)

	lams = np.logspace(-6,2,num=10)
	individual_degrees = [1,2,3,4]
	#degrees = list(itertools.permutations(individual_degrees))
	degrees = get_all_degree_combinations(individual_degrees, 4)

	if(verbose):
		print("sanitized dataset\n")

	if(mode == 1):

		if(verbose):
			print("mode 1: lasso regression\n")

		# run grid search to find the optimal combination of weight penalty (lambda/lam) and polynomial
		# order (D/degree) for lasso regression; this works by running cv on every combination
		# and finding the combination which minimizes average mean squared error betweeen the target and
		# predicted vectors; once this is found, run lasso regression with optimal lambda and polynomial
		# order to generate predictions for each year and display findings with matplotlib

		D, lam = grid_search(lasso_kfcv,lasso_regression,lams,degrees,data,seed,k,verbose)

		print("optimal D:",D)
		print("optimal lambda:",lam)

		plot_reg(data,data,lasso_regression,lam,D,verbose)
		print("mse across folds:",lasso_kfcv(lasso_regression,data,k,seed,lam,D,verbose))

	if(mode == 2):

		if(verbose):
			print("mode 2: neural networks\n")

		functions = ['relu','logistic','tanh']

		function, lam = grid_search(neural_network_kfcv,neural_network,lams,functions,data,seed,k,verbose,nn_output=True)

		preds = neural_network(data[4],data[3],data[4],lam,function)

		plot(data,X[:,0],preds)

		print("mse across folds:",neural_network_kfcv(neural_network,data,k,seed,lam,function,verbose))

	if(mode == 3):

		if(verbose):
			print("mode 3: ridge regression\n")

		# like above, here we do the exact same process, excpet for ridge regression; in other words
		# run grid search to find the optimal combination of weight penalty (lambda/lam) and polynomial
		# order (D/degree) for ridge regression (optimal parameters are stored in the variables D and lam)
		# this works by running cv on every combination and finding the combination which minimizes average
		# mean squared error betweeen the target and predicted vectors; once this is found, run ridge
		# regression with optimal lambda and polynomial order to generate predictions for each year and
		# display findings with matplotlib; this part is run in the plot_reg function

		D, lam = grid_search(ridge_kfcv,ridge_regression,lams,degrees,data,seed,k,verbose)

		print("optimal D:",D)
		print("optimal lambda:",lam)

		plot_reg(data,data,ridge_regression,lam,D,verbose)
		print("mse across folds:",ridge_kfcv(ridge_regression,data,k,seed,lam,D,verbose))

	if(mode == 4):
		
		if(verbose):
			print("mode 4: local ridge regression\n")

		# same as ridge regression except that we use a locally coded function instead of sklearn

		D, lam = grid_search(ridge_kfcv,local_ridge_regression,lams,degrees,data,seed,k,verbose)

		print("optimal D:",D)
		print("optimal lambda:",lam)
		
		plot_reg(data,data,local_ridge_regression,lam,D,verbose)
		print("mse across folds:",ridge_kfcv(local_ridge_regression,data,k,seed,lam,D,verbose))

if __name__ == "__main__":

	# verbose option indicates whether to run the program with verbose
	# mode indicates which method to run the program with
	# i.e. 1 -> lasso regression, 2 -> neural networks, 3 -> ridge regression, 4 -> local ridge regression
	# countries corresponds to each individual country to consider with our algorithms
	# seed is used in random number generation and should be set to make results reproducable
	# k is the number of folds to use when cross validating our algorithm to find
	# optimal parameters

	verbose = True
	mode = 1
	countries = all_world_countries()
	seed = 40
	k = 5

	driver(verbose,mode,countries,seed,k)
