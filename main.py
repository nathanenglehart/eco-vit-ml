import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from lib.utils import build_dataset
from lib.utils import load_dataset
from lib.kfcv import lasso_kfcv

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

def lasso_regression(countries,lam,degree,country_idx):
	
	""" Runs lasso regression with the given parameters and returns predictions corresponding to years.

		Args:

			countries::[Numpy Array]	
				Numpy array containing iso, code, name, t, and X info for each selected country

			lam::[Float]
				Weight penalty for high polynomial degrees

			degree::[Integer]
				Polynomial degree to use with lasso regression

			country_idx::[Integer]
				Index of country for which we want to compute a regression for
			
	"""

	model=make_pipeline(PolynomialFeatures(degree),Lasso(alpha=lam,max_iter=10000))
	model.fit(countries[country_idx][4],countries[country_idx][3]) # X, t

	year = 1995

	preds = list()
	years = list()
		
	for i in range(26):
		years.append(year)
		year_idx = year - 1995
		year += 1
		preds.append(model.predict([countries[country_idx][4][year_idx]]))

	preds = np.array(preds)
	years = np.array(years)

	return years, preds

def grid_search(cv,lams,degrees,countries,country_idx,verbose):

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

			verbose::[Boolean]
				Indicates whether to display verbose output

	"""

	k = 10
	seed = 40	

	min_mse = lasso_kfcv(lasso_regression,countries,country_idx,k,seed,lams[0],1,verbose)
	pair = 1, lams[0]

	# Determine optimal D parameter

	for lam in lams:

		for degree in degrees:
			
			average_mse = cv(lasso_regression,countries,country_idx,k,seed,lam,degree,verbose)
			
			if(average_mse < min_mse):
				pair = degree, lam
	
	return pair


def driver(verbose,mode):

	""" Drives each machine learning method we use based on parameter inputs.

		Args:

			verbose::[Boolean]
				Determines whether to run algorithms with verbose output

			mode::[Integer]
				Determines which method to use i.e. 1 for lasso regression, 2 for neural networks

	"""	

	# Create single dataset

	dataset = load_dataset()
	countries = build_dataset(dataset,verbose)
	
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

	if(mode == 1):

		if(verbose):
			print("mode 1: lasso regression")

		lams = [0.001,0.01,0.1,1.0,10.0]
		degrees = [1,2,3,4,5,6,7]
		country_idx = 0 # Australia

		optimal_polynomial_order, optimal_weight_value = grid_search(lasso_kfcv,lams,degrees,countries,country_idx,verbose)
		D = optimal_polynomial_order
		lam = optimal_weight_value

		print("optimal D:",D)
		print("optimal lambda:",lam)

		years, preds = lasso_regression(countries,lam,D,country_idx)
		plt.scatter(years, countries[0][3], color = 'g')
		plt.plot(years, preds, label="preds")
		plt.xlabel('Years')
		plt.ylabel('cda')
		plt.show()

	if(mode == 2):
		
		if(verbose):
			print("mode 2: neural networks")

if __name__ == "__main__":
	
	# program settings

	verbose = True
	mode = 1

	# run program

	driver(verbose,mode)


	
