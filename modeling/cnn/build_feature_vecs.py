from keras import layers
from keras import models
from keras.models import load_model

from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import os
import scipy.misc as misc
import numpy as np

DATA = "../../data"
DHS = DATA+"/DHS/unzipped/"
SAT = DATA+"/satellite/"
REG = DATA+"/regression/"


def build_features(reg_folder, pic_size,  model_h5):
	'''
	Given a model h5 file and a string pointing
	to the corresponding regression data, loads the trained 
	CNN in model_h5 then passes over the corrresponding regression
	data and constructs the average feature vector for each cluster.
	Saves out a csv file of each cluster's respective feature vector

	Inputs:
		- reg_folder: (str) pointing to regression data
		- model_h5: (str) filename of trained CNN model

	Saves to disk:
		- feature_csv: (csv file) containing predicted feature vectors
	'''

	model = load_model("output/" + model_h5 + "/model.h5")

	get_features = K.function([model.layers[0].input, K.learning_phase()],
					[model.layers[-2].output])

	reg_path = REG + reg_folder
	print(reg_path)
	folders = [x for x in os.listdir(reg_path) if ".csv" not in x]

	output_dict = {}

	for cluster in folders:
		all_pics = os.listdir(reg_path+"/"+cluster)

		pics_array = np.empty( (len(all_pics), pic_size, pic_size, 3) )
		row = 0

		for pic_file in all_pics:
			im = misc.imread(reg_path+"/"+cluster+"/"+pic_file)

			assert im.shape == (pic_size, pic_size, 3)

			pics_array[row] = im 
			row += 1

		feature_vecs = get_features( [pic_size, 0] )[0]

		avg_feature = feature_vecs.mean(axis=1)

		output_dict['cluster'] = avg_feature


	return output_dict

if __name__ == '__main__':
	
	reg = "size34"
	size = 34
	mod = "mod_1"

	x = build_features(reg, size, mod)