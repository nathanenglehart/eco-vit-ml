import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import Ridge

# Nathan Englehart, Ishaq Kothari, Raul Segredo (Autumn 2021)

if __name__ == "__main__":

	verbose = True
	cv = True
	mode = 1

	# Create single dataset

	cda_dataset = pd.read_csv("data/CDA_ind_na.csv")
	cda_dataset.columns = ['code','iso','country','CDA.ind.1995','CDA.ind.1996','CDA.ind.1997','CDA.ind.1998','CDA.ind.1999','CDA.ind.2000','CDA.ind.2001','CDA.ind.2002','CDA.ind.2003','CDA.ind.2004','CDA.ind.2005','CDA.ind.2006','CDA.ind.2007','CDA.ind.2008','CDA.ind.2009','CDA.ind.2010','CDA.ind.2011','CDA.ind.2012','CDA.ind.2013','CDA.ind.2014','CDA.ind.2015','CDA.ind.2016','CDA.ind.2017','CDA.ind.2018','CDA.ind.2019','CDA.ind.2020']

	grl_dataset = pd.read_csv("data/GRL_ind_na.csv")
	grl_dataset.columns = ['code','iso','country','GRL.ind.1995','GRL.ind.1996','GRL.ind.1997','GRL.ind.1998','GRL.ind.1999','GRL.ind.2000','GRL.ind.2001','GRL.ind.2002','GRL.ind.2003','GRL.ind.2004','GRL.ind.2005','GRL.ind.2006','GRL.ind.2007','GRL.ind.2008','GRL.ind.2009','GRL.ind.2010','GRL.ind.2011','GRL.ind.2012','GRL.ind.2013','GRL.ind.2014','GRL.ind.2015','GRL.ind.2016','GRL.ind.2017','GRL.ind.2018','GRL.ind.2019','GRL.ind.2020']

	tcl_dataset = pd.read_csv("data/TCL_ind_na.csv")
	tcl_dataset.columns = ['code','iso','country','TCL.ind.1995','TCL.ind.1996','TCL.ind.1997','TCL.ind.1998','TCL.ind.1999','TCL.ind.2000','TCL.ind.2001','TCL.ind.2002','TCL.ind.2003','TCL.ind.2004','TCL.ind.2005','TCL.ind.2006','TCL.ind.2007','TCL.ind.2008','TCL.ind.2009','TCL.ind.2010','TCL.ind.2011','TCL.ind.2012','TCL.ind.2013','TCL.ind.2014','TCL.ind.2015','TCL.ind.2016','TCL.ind.2017','TCL.ind.2018','TCL.ind.2019','TCL.ind.2020']

	wtl_dataset = pd.read_csv("data/WTL_ind_na.csv")
	wtl_dataset.columns = ['code','iso','country','WTL.ind.1995','WTL.ind.1996','WTL.ind.1997','WTL.ind.1998','WTL.ind.1999','WTL.ind.2000','WTL.ind.2001','WTL.ind.2002','WTL.ind.2003','WTL.ind.2004','WTL.ind.2005','WTL.ind.2006','WTL.ind.2007','WTL.ind.2008','WTL.ind.2009','WTL.ind.2010','WTL.ind.2011','WTL.ind.2012','WTL.ind.2013','WTL.ind.2014','WTL.ind.2015','WTL.ind.2016','WTL.ind.2017','WTL.ind.2018','WTL.ind.2019','WTL.ind.2020']

	del grl_dataset['code']
	del tcl_dataset['code']
	del wtl_dataset['code']

	del grl_dataset['iso']
	del tcl_dataset['iso']
	del wtl_dataset['iso']

	del grl_dataset['country']
	del tcl_dataset['country']
	del wtl_dataset['country']

	cda_dataset = pd.concat([cda_dataset,grl_dataset],axis=1)
	cda_dataset = pd.concat([cda_dataset,tcl_dataset],axis=1)
	cda_dataset = pd.concat([cda_dataset,wtl_dataset],axis=1)

	# Countries to consider

	countries = np.array(['Australia','United States of America','Nigeria','China','India','Brazil','United Kingdom','Zambia'])

	country_rows = list()

	for i in range(len(cda_dataset)):

		row = np.array(cda_dataset.iloc[i])

		if row[2] in countries:
			country_rows.append(row)

	country_rows = np.array(country_rows)

	# 3 values at first then cda for 25, grl for 25, tcl for 25, wtl for 25

	countries = list()
	country = list()

	for entry in country_rows:

		t = []
		#X = [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]]
		X = [[1995,1996,1997,1998,1999,2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020]]
		col = []
		col_count = 0

		feature = 2

		for i in range(len(cda_dataset.columns)):

			if i < 29 and i >= 3:
				t.append(entry[i])
				if i == 28 and verbose:
					print("added t")
					print("added x1")

			if i >= 29:
				col.append(entry[i])
				col_count += 1

			if(col_count == 26):
				if verbose:
					msg = "added x" + str(feature)
					print(msg)
					feature += 1

				X.append(np.array(col))
				col.clear()
				col_count = 0

		if verbose:
			print("")

		t = np.array(t)
		X = np.array(X,dtype=object).T
		country.append(entry[0]) # code
		country.append(entry[1]) # iso
		country.append(entry[2]) # country
		country.append(t) # target vector
		country.append(X) # train matrix

		countries.append(np.array(country,dtype=object))

		country.clear()

	countries = np.array(countries)

	# indexing works as follows
	# countries[0] gives the first country from our list of selected countries in alphabetical order
	# e.g. if our list contains Australia, Brazil, and Columbia, countries[0] would give Australia
	# countries[0][1] gives the iso for the first country and countries[0][2] gives the country name
	# for the above example [0][1] would give AUS and [0][2] would give Austrialia
	# countries[0][3] gives the target vector, t, in the form of an np array (this array should be a column)
	# countries[0][4] gives the train martix, X, in the form of an np array of np arrays (each array is a column)
	# including a demo below

	print(countries[0][1]) # country iso
	print("len(t):",len(countries[0][3])) # target vector i.e. cda
	print("len(X):",np.size(countries[0][4],1)) # number of features
	print("len(x1):",len(countries[0][4][:,0])) # year
	print("len(x2):",len(countries[0][4][:,1])) # grl (grassland)
	print("len(x3):",len(countries[0][4][:,2])) # tcl (tree cover)
	print("len(x4):",len(countries[0][4][:,3])) # wtl (wetland)

	print(countries[0][4])

	if(mode == 1):

		if(verbose):
			print("mode 1: ridge regression")

		model = Ridge(alpha=1.0)	
		model.fit(countries[0][4],countries[0][3])

		year = 1995

		preds = list()
		years = list()

		for i in range(26):
			years.append(year)
			year += 1
			#preds.append(model.predict([np.array([year,43.97,24,72])]))
			preds.append(model.predict([np.array([year,43.89,26.90,66.79])]))


		preds = np.array(preds)
		years = np.array(years)
		cda = countries[0][3]

		plt.scatter(years, cda, color = 'g')
		plt.plot(years, preds, label="preds")
		plt.xlabel('Years')
		plt.ylabel('cda')
		plt.show()
		print(model.coef_)
		print(model.intercept_)
