import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

def qr(X):

	""" Computes Q and R for X matrix.
    		
		Args:
            		
			X::[Numpy Array]
                		Matrix for which to compute the QR. Note: run before X_build()
    	"""
	
	return np.linalg.qr(X)

def load_dataset():

	""" Loads datasets relavent to the project.

		Args:
			
			NA

	"""

	grl_dataset = pd.read_csv("data/GRL_ind_na.csv")
	grl_dataset.columns = ['code','iso','country','GRL.ind.1995','GRL.ind.1996','GRL.ind.1997','GRL.ind.1998','GRL.ind.1999','GRL.ind.2000','GRL.ind.2001','GRL.ind.2002','GRL.ind.2003','GRL.ind.2004','GRL.ind.2005','GRL.ind.2006','GRL.ind.2007','GRL.ind.2008','GRL.ind.2009','GRL.ind.2010','GRL.ind.2011','GRL.ind.2012','GRL.ind.2013','GRL.ind.2014','GRL.ind.2015','GRL.ind.2016','GRL.ind.2017','GRL.ind.2018','GRL.ind.2019','GRL.ind.2020']

	tcl_dataset = pd.read_csv("data/TCL_ind_na.csv")
	tcl_dataset.columns = ['code','iso','country','TCL.ind.1995','TCL.ind.1996','TCL.ind.1997','TCL.ind.1998','TCL.ind.1999','TCL.ind.2000','TCL.ind.2001','TCL.ind.2002','TCL.ind.2003','TCL.ind.2004','TCL.ind.2005','TCL.ind.2006','TCL.ind.2007','TCL.ind.2008','TCL.ind.2009','TCL.ind.2010','TCL.ind.2011','TCL.ind.2012','TCL.ind.2013','TCL.ind.2014','TCL.ind.2015','TCL.ind.2016','TCL.ind.2017','TCL.ind.2018','TCL.ind.2019','TCL.ind.2020']

	wtl_dataset = pd.read_csv("data/WTL_ind_na.csv")
	wtl_dataset.columns = ['code','iso','country','WTL.ind.1995','WTL.ind.1996','WTL.ind.1997','WTL.ind.1998','WTL.ind.1999','WTL.ind.2000','WTL.ind.2001','WTL.ind.2002','WTL.ind.2003','WTL.ind.2004','WTL.ind.2005','WTL.ind.2006','WTL.ind.2007','WTL.ind.2008','WTL.ind.2009','WTL.ind.2010','WTL.ind.2011','WTL.ind.2012','WTL.ind.2013','WTL.ind.2014','WTL.ind.2015','WTL.ind.2016','WTL.ind.2017','WTL.ind.2018','WTL.ind.2019','WTL.ind.2020']

	#mld_dataset = pd.read_csv("data/percentchangeco2data.csv")
	mld_dataset = pd.read_csv("data/mld_cleaned.csv")
	mld_dataset.columns = ['code','iso','country','1995','1996','1997','1998','1999','2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015','2016','2017','2018','2019','2020']

	del grl_dataset['code']
	del tcl_dataset['code']
	del wtl_dataset['code']

	del grl_dataset['iso']
	del tcl_dataset['iso']
	del wtl_dataset['iso']

	del grl_dataset['country']
	del tcl_dataset['country']
	del wtl_dataset['country']

	mld_dataset = pd.concat([mld_dataset,wtl_dataset],axis=1)
	mld_dataset = pd.concat([mld_dataset,tcl_dataset],axis=1)
	mld_dataset = pd.concat([mld_dataset,grl_dataset],axis=1)

	return mld_dataset

def build_dataset(dataset,countries,verbose):

	""" Returns multidimensional numpy array for quick searching given input dataset.

		Args:

			dataset::[Pandas Dataframe]
				Dataset with which to create numpy array

			countries::[Numpy array]
				Array of the names of countries (string format) to load from datasets

			verbose::[Boolean]
				Determines whether to run algorithms with verbose output

	"""



	country_rows = list()

	for i in range(len(dataset)):

		row = np.array(dataset.iloc[i])

		if row[2] in countries:
			country_rows.append(row)

	country_rows = np.array(country_rows)

	# 3 values at first then lcb for 25, cda for 25, grl for 25, tcl for 25, wtl for 25

	countries = list()
	country = list()

	for entry in country_rows:

		t = list()
		X = [np.array([1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])]
		col = list()
		col_count = 0

		feature = 2

		if verbose:
			print(entry[1])

		for i in range(len(dataset.columns)):

			if(i < 29 and i >= 3):
				t.append(entry[i])
				if(i == 28 and verbose):
					print("added t")
					print("added x1")

			if(i >= 29):
				col.append(entry[i])
				col_count += 1

			if(col_count == 26):
				if(verbose):
					msg = "added x" + str(feature)
					print(msg)
					feature += 1

				X.append(np.array(col))
				col.clear()
				col_count = 0

		if(verbose):
			print("")

		t = np.array(t)
		X = np.array(X,dtype=object).T

		check_t = True
		check_X = True

		if((t.dtype.char == 'S') == False and np.isnan(np.sum(t)) == False):
			check_t = True
		else:
			check_t = False

		for i in range(len(X)):

			x = X[i]

			if((x.dtype.char == 'S') == True or np.isnan(np.sum(x)) == True):
				check_X = False
				break

		if(check_t == True and check_X == True):

			country.append(entry[0]) # code
			country.append(entry[1]) # iso
			country.append(entry[2]) # country
			country.append(t) # target vector
			country.append(X) # train matrix

			countries.append(np.array(country,dtype=object))

		country.clear()

	return np.array(countries)


def build_world(countries):

	""" Returns averaged X matrix and t vector for all countries.

		Args:

			countries::[Numpy array]
				Array countaining X matrix and t vector for all loaded countries

	"""

	t = countries[0][3]
	X = countries[0][4]

	for i in range(len(countries)-1):
		t = t + countries[i+1][3]
		X = X + countries[i+1][4]

	return t / (len(countries)), X / (len(countries))


def X_build_col(col,D):

    """ Takes a single predictor column and a positive integer D, and creates a predictor matrix whose columns consist of powers of the entries in the original column from 0 to D.

		Args:

			col::[Numpy Array]
                		Single predictor column

	    		D::[Integeri list]
                		List of positive integer to be used as polynomial orders

    """

    # create a D + 1 matrix from the column matrix


    predictor_matrix = np.ones((len(col),D+1))

    for i in range(0,len(col)):
        for j in range(0,D):
            val = col[i]**(j)
            predictor_matrix[i][j] = val

    return predictor_matrix

def X_build(data,degree):

	""" Takes all columns in matrix, makes them into predictor matricies, and stacks them together horizontally.

		Args:

			data::[Numpy Array]
				Array that holds iso, name, code, t vector, and X matrix
			
			degree::[Integer list]
				List that holds all polynomial orders to use in building our matrix

	"""

	# for each column, expand into D + 1 matrix that is from 0 to D, raise each column to the power
	# of the current power, add it to the column matrix; once reaching D, add it to the matrix

	place = 0

	D = degree[place]

	X = X_build_col(data[4][:,0],D)

	place += 1

	for col in range(0,data[4].shape[1]-1):
		D = degree[place]
		X = np.column_stack((X,X_build_col(data[4][:,col+1],D)))
		place += 1

	return X

def plot_reg(data,to_predict,reg,lam,D,print_coef):

	""" Runs regression on data with given lambda and D parameters and shows graph output with matplotlib.

		Args:

			data::[Numpy Array]
				Array that holds iso, name, code, t vector, and X matrix

			to_predict::[Numpy Array]
				Matrix to predict with model

			reg::[Function]
				Regression function i.e. lasso or ridge

			lam::[Float]
				Optimal lambda parameter (penalty for weights) for regression

			D::[Integer]
				Optimal D parameter (polynomial order) for regression

			print_coef::[Boolean]
				Whether or not to print the model coefficients

	"""

	
	years, preds = reg(data,to_predict,lam,D,print_coef)
	plt.scatter(years, data[3], color = 'g')
	plt.plot(years, preds, label="preds")
	plt.title(reg)
	plt.xlabel('Years')
	plt.ylabel('Atmospheric Carbon Dioxide Levels (micromol/mol)')
	plt.show()

def plot(data,years,preds):

	""" Graphs two numpy arrays (year vs pred) against each other.

		Args:

			data::[Numpy Array]
				Array that holds iso, name, code, t vector, and X matrix

			years::[Numpy Array]
				Array that holds years

			preds::[Numpy Array]
				Array that holds predicted values (from function)

	"""

	plt.scatter(years,data[3], color ='g')
	plt.plot(years,preds,label='preds')
	plt.xlabel('Years')
	plt.ylabel('Atmospheric Carbon Dioxide Levels (micromol/mol)')
	plt.show()


def all_world_countries():

	""" Returns array containing string names for every country in the world.

		Args:

			NA

	"""

	return np.array(['Afghanistan','Albania','Algeria','Angola','Antigua and Barbuda','Argentina','Armenia','Aruba','Australia','Austria','Azerbaijan','Bahamas','Bahrain','Bangladesh','Barbados','Belarus','Belgium','Belize','Benin','Bhutan','Bolivia','Bosnia and Herzegovina','Botswana','Brazil','Brunei Darussalam','Bulgaria','Burkina Faso','Burundi','Cabo Verde','Cambodia','Cameroon','Canada','Central African Republic','Chad','Chile','China','Colombia','Comoros','Costa Rica',"Cote d'Ivoire",'Croatia','Cuba','Cyprus','Czech Republic','Dem. Rep. Congo','Denmark','Djibouti','Dominica','Dominican Republic','Ecuador','Egypt','El Salvador','Equatorial Guinea','Eritrea','Estonia','Eswatini','Ethiopia','Fiji','Finland','France','Gabon','Gambia','Georgia','Germany','Ghana','Greece','Grenada','Guatemala','Guinea','Guinea-Bissau','Guyana','Haiti','Honduras','Hong Kong','Hungary','Iceland','India','Indonesia','Iran','Iraq','Ireland','Israel','Italy','Jamaica','Japan','Jordan','Kazakhstan','Kenya','Kiribati','Kuwait','Kyrgyzstan','Laos','Latvia','Lebanon','Lesotho','Liberia','Libya','Lithuania','Luxembourg','Macao','Madagascar','Malawi','Malaysia','Maldives','Mali','Malta','Marshall Islands','Mauritania','Mauritius','Mexico','Micronesia','Moldova','Mongolia','Montenegro','Morocco','Mozambique','Myanmar','Namibia','Nauru','Nepal','Netherlands','New Zealand','Nicaragua','Niger','Nigeria','North Macedonia','Norway','Oman','Pakistan','Palau','Panama','Papua New Guinea','Paraguay','Peru','Philippines','Poland','Portugal','Qatar','Republic of Congo','Romania','Russia','Rwanda','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Samoa','San Marino','Sao Tome and Principe','Saudi Arabia','Senegal','Serbia','Seychelles','Sierra Leone','Singapore','Slovakia','Slovenia','Solomon Islands','South Africa','South Korea','South Sudan','Spain','Sri Lanka','Sudan','Suriname','Sweden','Switzerland','Taiwan','Tajikistan','Tanzania','Thailand','Timor-Leste','Togo','Tonga','Trinidad and Tobago','Tunisia','Turkey','Turkmenistan','Tuvalu','Uganda','Ukraine','United Arab Emirates','United Kingdom','United States of America','Uruguay','Uzbekistan','Vanuatu','Venezuela','Viet Nam','Yemen','Zambia','Zimbabwe'])

def append_ones(X):

      """ Appends row of ones to matrix.
		
		Args:

			X::[Numpy Array]

      """

      ones = np.ones((X.shape[0],1))
      return np.column_stack((ones,X))


combinations = list()

def get_all_degree_combinations(degrees, k):

	""" Generates all possible combinations of polynomial orders to test.

		Args:
			
			degrees::[Integer list]
				List of polynomial degrees to get combinations for

	"""

	for i in range(len(degrees)):
		degrees[i] = str(degrees[i])

	n = len(degrees)

	get_all_degree_combinations_rec_helper(degrees, "", n, k)

	return combinations

def get_all_degree_combinations_rec_helper(degrees, current, n, k):

	""" Recursive helper for generating all possible combinations of polynomial orders to test.

		Args:

			degrees::[Character list]
				List of polynomial degrees to get combinations for

			current::[String]
				Current string prefix

			n::[Integer]
				Actual length of combination

			k::[Integer]
				Current length in recursive helper (which starts at n)

	"""

	if (k == 0) :

		ret = list()

		for i in range(0,n):
			ret.append(int(current[i]))

		combinations.append(ret)

		return
		

	for i in range(n):

		new = current + degrees[i]
		
		get_all_degree_combinations_rec_helper(degrees, new, n, k - 1)


