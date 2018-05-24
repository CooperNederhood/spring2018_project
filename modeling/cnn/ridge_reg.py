import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn import linear_model

DATA = "../../data"
DHS = DATA+"/DHS/unzipped/"
SAT = DATA+"/satellite/"
REG = DATA+"/regression/"


def ridge_reg_analysis(reg_folder, pic_size,  model_h5):
	'''
	After running the build_feature_vecs script, we have the feature maps
	learned by the Neural Net and we can use these as the explanatory
	variables in a linear L2 ridge regression on the corresonding DHS cluster
	wealth index average

	Inputs:
		- reg_folder: (str) pointing to regression data
		- model_h5: (str) filename of trained CNN model

	Saves to disk:
		- feature_csv: (csv file) containing predicted feature vectors
	'''

	reg_path = REG + reg_folder
	cluster_avgs = pd.read_csv(reg_path+"/cluster_averages.csv")

	model_path = "output/"+model_h5

	feature_maps = pd.read_csv(model_path+"/cluster_features.csv")
	feature_maps.rename(columns={"Unnamed: 0": "DHSCLUSTER"}, inplace=True)

	reg_df = feature_maps.merge(cluster_avgs[["DHSCLUSTER", "hv271"]], how='left', on='DHSCLUSTER')

	x_vars = [c for c in reg_df.columns if c not in {"DHSCLUSTER", "hv271"} and reg_df[c].mean() != 0]

	X_data = reg_df[x_vars].values 
	Y_data = reg_df[['hv271']].values
	assert X_data.shape[0] == Y_data.shape[0]

	ridge_reg = linear_model.Ridge(alpha=1)
	ridge_reg.fit(X_data, Y_data)



	return ridge_reg, X_data, Y_data







if __name__ == '__main__':
	
	reg = "size34"
	size = 34
	mod = "mod_1"

	reg, X, Y = ridge_reg_analysis(reg, size, mod)




