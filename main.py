import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.neural_network import MLPClassifier

from lib.utils import build_world
from lib.utils import build_dataset
from lib.utils import load_dataset

from lib.kfcv import lasso_kfcv
from lib.reg import lasso_regression

from lib.kfcv import ridge_kfcv
from lib.reg import ridge_regression

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)


def grid_search(cv,reg,lams,degrees,data,seed,k,verbose):

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

	min_mse = cv(reg,data,k,seed,lams[0],degrees[0],verbose)
	pair = degrees[0], lams[0]

	# find every combination of lambda and D from our given lists
	# if the average mse of a calculation is lower than our default, set it as our global min value
	# return optimal pair which minimizes mse

	for lam in lams:

		for degree in degrees:

			average_mse = cv(reg,data,k,seed,lam,degree,verbose)

			if(average_mse < min_mse):
				pair = degree, lam

	return pair

def driver(verbose,mode,country_names,country_indicies,seed,k):

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

	if(mode == 1):

		if(verbose):
			print("mode 1: lasso regression\n")

		lams = [0.001,0.01,0.1,1.0,10.0]
		degrees = [1,2,3,4,5,6,7]

		optimal_polynomial_order, optimal_weight_value = grid_search(lasso_kfcv,lasso_regression,lams,degrees,data,seed,k,verbose)
		D = optimal_polynomial_order
		lam = optimal_weight_value
			
		print("optimal D:",D)
		print("optimal lambda:",lam,"\n")

		years, preds = lasso_regression(data,lam,D)
		plt.scatter(years, data[3], color = 'g')
		plt.plot(years, preds, label="preds")
		plt.xlabel('Years')
		plt.ylabel('MLD')
		plt.show()

	if(mode == 2):

		if(verbose):
			print("mode 2: neural networks\n")

	if(mode == 3):
		
		if(verbose):
			print("mode 3: ridge regression\n")

		lams = [0.001,0.01,0.1,1.0,10.0]
		degrees = [1,2,3,4,5,6,7]

		optimal_polynomial_order, optimal_weight_value = grid_search(ridge_kfcv,ridge_regression,lams,degrees,data,seed,k,verbose)
		D = optimal_polynomial_order
		lam = optimal_weight_value
			
		print("optimal D:",D)
		print("optimal lambda:",lam,"\n")

		years, preds = ridge_regression(data,lam,D)
		plt.scatter(years, data[3], color = 'g')
		plt.plot(years, preds, label="preds")
		plt.xlabel('Years')
		plt.ylabel('MLD')
		plt.show()



if __name__ == "__main__":

	# verbose option indicates whether to run the program with verbose
	# mode indicates which method to run the program with
	# i.e. 1 -> lasso regression, 2 -> neural networks
	# countries corresponds to each individual country to consider with our algorithms
	# seed is used in random number generation and should be set to make results reproducable
	# k is the number of folds to use when cross validating our algorithm to find
	# optimal parameters

	verbose = True
	mode = 3
	countries = np.array(['Afghanistan','Albania','Algeria','Angola','Antigua and Barbuda','Argentina','Armenia','Aruba','Australia','Austria','Azerbaijan','Bahamas','Bahrain','Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bhutan','Bolivia','Bosnia and Herzegovina','Botswana','Brazil','Brunei Darussalam','Bulgaria','Burkina Faso','Burundi','Cabo Verde','Cambodia','Cameroon','Canada','Central African Republic','Chad','Chile','China','Colombia','Comoros','Costa Rica',"Cote d'Ivoire",'Croatia','Cuba','Cyprus','Czech Republic','Dem. Rep. Congo','Denmark','Djibouti','Dominica','Dominican Republic','Ecuador','Egypt','El Salvador','Equatorial Guinea','Eritrea','Estonia','Eswatini','Ethiopia','Fiji','Finland','France','Gabon','Gambia','Georgia','Germany','Ghana','Greece','Grenada','Guatemala','Guinea','Guinea-Bissau','Guyana','Haiti','Honduras','Hong Kong','Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati','Kuwait','Kyrgyzstan','Laos','Latvia','Lebanon','Lesotho','Liberia','Libya','Lithuania','Luxembourg','Macao','Madagascar','Malawi','Malaysia','Maldives','Mali','Malta','Marshall Islands','Mauritania','Mauritius','Mexico','Micronesia','Moldova','Mongolia','Montenegro','Morocco','Mozambique','Myanmar','Namibia','Nauru','Nepal','Netherlands','New Zealand','Nicaragua','Niger','Nigeria','North Macedonia','Norway','Oman','Pakistan','Palau','Panama','Papua New Guinea','Paraguay','Peru','Philippines','Poland','Portugal','Qatar','Republic of Congo','Romania','Russia','Rwanda','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Samoa','San Marino','Sao Tome and Principe','Saudi Arabia','Senegal','Serbia','Seychelles','Sierra Leone','Singapore','Slovakia','Slovenia','Solomon Islands','South Africa','South Korea','South Sudan','Spain','Sri Lanka','Sudan','Suriname','Sweden','Switzerland','Taiwan','Tajikistan','Tanzania','Thailand','Timor-Leste','Togo','Tonga','Trinidad and Tobago','Tunisia','Turkey','Turkmenistan','Tuvalu','Uganda','Ukraine','United Arab Emirates','United Kingdom','United States of America','Uruguay','Uzbekistan','Vanuatu','Venezuela','Viet Nam','Yemen','Zambia','Zimbabwe'])
	country_indicies = np.array([0])
	seed = 40
	k = 10

	driver(verbose,mode,countries,country_indicies,seed,k)
