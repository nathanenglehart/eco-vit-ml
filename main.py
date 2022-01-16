import numpy as np
import pandas as pd
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

	# Countries (which correspond to numerical labels) to consider

	countries = np.array(['Australia','United States of America','Nigeria','China','India','Brazil','United Kingdom'])

	country_rows = list()

	for i in range(len(cda_dataset)):
		
		row = np.array(cda_dataset.iloc[i])

		if row[2] in countries:
			country_rows.append(row)

	

	if(mode == 1):

		if(verbose):
			print("mode 1: ridge regression")

		model = Ridge(alpha=1.0)

		#t = np.array(dataset[''])
		
		#X = np.array(dataset['Year'])
		

	
	

