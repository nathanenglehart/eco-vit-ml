import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Nathan Englehart, Ishaq Kothari, Raul Segredo

if __name__ == "__main__":

	verbose = True
	cv = True
	mode = 1

	dataset = pd.read_csv("data/SYB64_319_202110_Ratio of girls to boys in education.csv", encoding="ISO-8859-1")
	dataset.columns = ['Region Code','Ratio of girls to boys','Year','Series','Value','Footnotes','Sources']

	print(dataset)

	if(mode == 1):

		if(verbose):
			print("mode 1: ridge regression")

		model = Ridge(alpha=1.0)

		t = np.array(dataset['Value'])
		
		X = np.array(dataset['Year'])
		

	
	

