import numpy as np
import numpy.random as rand 
import skimage.external.tifffile as tiff
import scipy.misc as misc
from matplotlib import pyplot as plt 
import pandas as pd 
import timeit 

# hardcoded path is temporary
PATH = "/home/cooper/Documents/spring2018_project/data/satellite/nigeria2013"

IMG_SIZE = 34
FROM_CENTER = int(IMG_SIZE / 2)

class SatDataObs():
	"""
	A datapoint will be a 1-km nighlight luminosity value 
	paired with the corresponding 1-km Landsat imagery which
	is at 30m resolution.
	"""

	IMG_SIZE = 34
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


	def save_extract(self, landsat_array):
		'''
		Saves out the corresponding extraction but from the 
		landsat image array in landsat_array. Saves the image
		under the corresponding bin subfolder
		'''

		r0 = self.c_row - SatDataObs.FROM_CENTER
		r1 = self.c_row + SatDataObs.FROM_CENTER

		c0 = self.c_col - SatDataObs.FROM_CENTER
		c1 = self.c_col + SatDataObs.FROM_CENTER

		extract = landsat_array[r0:r1, c0:c1, :]

		filename = "obs_{}.png".format(self.obs_num)

		misc.imsave(arr=extract, name=PATH+"/y2013/"+str(self.bin_lum)+"/"+filename)




	def __str__(self):

		s0 = "From {} w/ center = {}\n".format(self.source_file, self.center)
		s1 = "\t mean lum = {}".format(self.mean_lum)

		return s0+s1


def sample_from_image(image_array, source_file, N):
	'''
	Given a np image_array, returns a list of N
	SatDataObs objects sampled from the image_array
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


def visualize_sample(lum_counts):
	'''
	Returns a pyplot of the mean luminosity
	and of the binned luminosity into the 3 bins
	'''

	counts = pd.value_counts(lum_counts).sort_index()
	plt.plot(counts.index, counts.values)

	plt.show() 

if __name__ == "__main__":

	sample_count = 10000

	nl_file = 'DMSPNL_2013GEO2.tif'
	landsat_file = 'landsat_2013GEO2.tif'

	nl_2013 = tiff.imread(PATH+"/"+nl_file)
	landsat_2013 = tiff.imread(PATH+"/"+landsat_file)

	samples, mean_lum, bin_lum = sample_from_image(nl_2013, nl_file, sample_count)


	t0 = timeit.default_timer()
	zero_factor = 4
	c = 0
	added_c = 0

	for pic in samples:
		begin = timeit.default_timer()

		if pic.mean_lum != 0 or np.mod(c, zero_factor) != 0:
			print("Adding pic:")
			print(pic)
			pic.save_extract(landsat_2013)

			elapsed_time = timeit.default_timer() - begin
			print("Took {} seconds".format(elapsed_time))
			print()

			added_c += 1

		c += 1

	total = timeit.default_timer() - t0
	print()
	print("#######################")
	print("Elapsed time to add {} of {} sample = {} minutes".format(added_c, sample_count, total/60))


