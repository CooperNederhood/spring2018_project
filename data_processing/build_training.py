import numpy as np
import numpy.random as rand 
import skimage.external.tifffile as tiff
import scipy.misc as misc
from matplotlib import pyplot as plt 
import pandas as pd 
import timeit 
import os


# hardcoded path is temporary
PATH = "/home/cooper/Documents/spring2018_project/data/satellite/nigeria2013"

IMG_SIZE = 128
FROM_CENTER = int(IMG_SIZE / 2)

class SatDataObs():
	"""
	A datapoint will be a 1-km nighlight luminosity value 
	paired with the corresponding 1-km Landsat imagery which
	is at 30m resolution.
	"""

	IMG_SIZE = 128
	FROM_CENTER = int(IMG_SIZE / 2)
	OBS = 0

	def __init__(self, center, source_file=None):
		'''
		Constructor method.

		Attributes:
			- center (tuple): denoting the center coords within the larger image
			- source_file (str): file name corresponding to the larger image
			- mean_lum (float): mean luminosity across the subimage
			- bin (int): bin based on mean luminosity
		'''

		self.center = center
		self.c_row = center[0]
		self.c_col = center[1]
		self.source_file = source_file
		self.mean_lum = None
		self.bin_lum = None 
		self.obs_num = SatDataObs.OBS

		SatDataObs.OBS += 1 


	def set_mean_lum(self, image_array):
		'''
		Calculate the mean luminosity of the region

		Inputs:
			- image_array (np array): of the larger image
		'''

		r0 = self.c_row - SatDataObs.FROM_CENTER
		r1 = self.c_row + SatDataObs.FROM_CENTER

		c0 = self.c_col - SatDataObs.FROM_CENTER
		c1 = self.c_col + SatDataObs.FROM_CENTER

		extract = image_array[r0:r1, c0:c1]


		self.mean_lum = extract.mean()

		if self.mean_lum < 3:
			b = 1

		elif self.mean_lum < 34:
			b = 2

		elif self.mean_lum <= 63:
			b = 3
		else:
			print(self.mean_lum)
			b = None 

		self.bin_lum = b 


	def save_extract(self, landsat_array, save_at):
		'''
		Saves out the corresponding extraction but from the 
		landsat image array in landsat_array. Saves the image
		under the corresponding location in save_at
		'''

		r0 = self.c_row - SatDataObs.FROM_CENTER
		r1 = self.c_row + SatDataObs.FROM_CENTER

		c0 = self.c_col - SatDataObs.FROM_CENTER
		c1 = self.c_col + SatDataObs.FROM_CENTER

		extract = landsat_array[r0:r1, c0:c1, :]

		filename = "obs_{}.png".format(self.obs_num)

		misc.imsave(arr=extract, name=save_at+"/"+filename)




	def __str__(self):

		s0 = "From {} w/ center = {}\n".format(self.source_file, self.center)
		s1 = "\t mean lum = {}".format(self.mean_lum)

		return s0+s1


def sample_from_image(image_array, source_file, N):
	'''
	Given a np image_array, returns a list of N
	SatDataObs objects sampled from the image_array

	Inputs:	
		-image_array: (np array) of a NIGHTLIGHTS photo from which we sample
		-source_file: (string) of the LANDSAT file name for image_array
		-N: (int) sample count to draw

	Returns:
		- rand_images: (list) of N SatDataObs instances
		- lum: (np array - N length) of luminisoty scores
		- lum_bin: (np array - N length) of luminisoty score bins
	'''

	shape = image_array.shape

	rand_rows = rand.randint(FROM_CENTER, shape[0]-FROM_CENTER, size=N)
	rand_cols = rand.randint(FROM_CENTER, shape[1]-FROM_CENTER, size=N)

	rand_images = []
	lum = np.empty(N)
	lum_bin = np.empty(N)

	for i in range(N):

		cur_center = (rand_rows[i], rand_cols[i])
		cur_extraction = SatDataObs(cur_center, source_file)
		cur_extraction.set_mean_lum(image_array)

		lum[i] = np.round(cur_extraction.mean_lum)
		lum_bin[i] = cur_extraction.bin_lum 

		rand_images.append(cur_extraction)

		print(cur_extraction)
		print()

	return rand_images, lum, lum_bin


def build_test_validate_train(image_array, landsat_array, source_file, counts_dict, zero_factor, save_at):
	'''
	Given a desired count structure for the training, validation, and
	testing size, samples accordingly from the image_array and builds
	the requisite folder structure for Keras

	Inputs:
		-image_array: LUMINOSITY we are sampling from
		-landsat_array: corresponding LANDSAT array
		-source_file: name of source file for luminosity array
		-counts_dict: counts for training, validation, test data
		-zero_factor: int describing rate to filter zero lum observations
		-save_at: directory to build the dataset
	'''

	train_count = counts_dict['training']
	test_count = counts_dict['validation']
	valid_count = counts_dict['test']

	total = train_count + test_count + valid_count 
	samples, mean_lum, bin_lum = sample_from_image(image_array, source_file, total*zero_factor)

	for t in counts_dict:
		for level in ["1", "2", "3"]:

			if not os.path.exists(save_at+t+"/"+level):
				os.makedirs(save_at+t+"/"+level)

	c = 0
	added_c = 0

	final_sample = []
	for pic in samples:

		if pic.mean_lum != 0 or np.mod(c, zero_factor) != 0:
			pic.obs_num = added_c
			final_sample.append(pic)

			added_c += 1

		c += 1

	for final_pic in final_sample[0:train_count]:
		final_pic.save_extract(landsat_array, save_at+"training/"+str(final_pic.bin_lum))

	for final_pic in final_sample[train_count: train_count+valid_count]:
		final_pic.save_extract(landsat_array, save_at+"validation/"+str(final_pic.bin_lum))

	for final_pic in final_sample[train_count+valid_count: total]:
		final_pic.save_extract(landsat_array, save_at+"test/"+str(final_pic.bin_lum))



def visualize_sample(lum_counts):
	'''
	Returns a pyplot of the mean luminosity
	and of the binned luminosity into the 3 bins
	'''

	counts = pd.value_counts(lum_counts).sort_index()
	plt.plot(counts.index, counts.values)

	plt.show() 

if __name__ == "__main__":


	nl_file = 'DMSPNL_2013GEO2.tif'
	landsat_file = 'landsat_2013GEO2.tif'

	nl_2013 = tiff.imread(PATH+"/"+nl_file)
	landsat_2013 = tiff.imread(PATH+"/"+landsat_file)

	structure = {'training': 1000, 'validation': 500, 'test': 500}

	build_test_validate_train(nl_2013, landsat_2013, nl_file, structure, 4, "lagos/")


