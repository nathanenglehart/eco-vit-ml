import numpy as np
import pandas as pd

def load_dataset():
	
	""" Loads datasets relavent to the project.

		Args:

	"""

	grl_dataset = pd.read_csv("data/GRL_ind_na.csv")
	grl_dataset.columns = ['code','iso','country','GRL.ind.1995','GRL.ind.1996','GRL.ind.1997','GRL.ind.1998','GRL.ind.1999','GRL.ind.2000','GRL.ind.2001','GRL.ind.2002','GRL.ind.2003','GRL.ind.2004','GRL.ind.2005','GRL.ind.2006','GRL.ind.2007','GRL.ind.2008','GRL.ind.2009','GRL.ind.2010','GRL.ind.2011','GRL.ind.2012','GRL.ind.2013','GRL.ind.2014','GRL.ind.2015','GRL.ind.2016','GRL.ind.2017','GRL.ind.2018','GRL.ind.2019','GRL.ind.2020']

	tcl_dataset = pd.read_csv("data/TCL_ind_na.csv")
	tcl_dataset.columns = ['code','iso','country','TCL.ind.1995','TCL.ind.1996','TCL.ind.1997','TCL.ind.1998','TCL.ind.1999','TCL.ind.2000','TCL.ind.2001','TCL.ind.2002','TCL.ind.2003','TCL.ind.2004','TCL.ind.2005','TCL.ind.2006','TCL.ind.2007','TCL.ind.2008','TCL.ind.2009','TCL.ind.2010','TCL.ind.2011','TCL.ind.2012','TCL.ind.2013','TCL.ind.2014','TCL.ind.2015','TCL.ind.2016','TCL.ind.2017','TCL.ind.2018','TCL.ind.2019','TCL.ind.2020']

	wtl_dataset = pd.read_csv("data/WTL_ind_na.csv")
	wtl_dataset.columns = ['code','iso','country','WTL.ind.1995','WTL.ind.1996','WTL.ind.1997','WTL.ind.1998','WTL.ind.1999','WTL.ind.2000','WTL.ind.2001','WTL.ind.2002','WTL.ind.2003','WTL.ind.2004','WTL.ind.2005','WTL.ind.2006','WTL.ind.2007','WTL.ind.2008','WTL.ind.2009','WTL.ind.2010','WTL.ind.2011','WTL.ind.2012','WTL.ind.2013','WTL.ind.2014','WTL.ind.2015','WTL.ind.2016','WTL.ind.2017','WTL.ind.2018','WTL.ind.2019','WTL.ind.2020']

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