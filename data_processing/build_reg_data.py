import numpy as np
import numpy.random as rand 
import skimage.external.tifffile as tiff
import scipy.misc as misc
from matplotlib import pyplot as plt 
import pandas as pd 
import timeit 
import os

DATA = "../data"
DHS = DATA+"/DHS/unzipped/"
SAT = DATA+"/satellite/"
REG = DATA+"/regression/"

ADJ = int((10000 / 30) // 2)

def build_reg_data(img_size):
	'''
	Call the DHS data, filter to just those clusters
	within our current area of analysis. Avearge across
	cluster. For each cluster, save out a subfolder with 
	the Landsat photos in a 10km x 10km square around the 
	cluster center. The amount of photos is a function
	of the img_size
	'''

	ls_data = tiff.imread(SAT+"SAT_DATA_LANDSAT.tif")

	path = REG + str(img_size) + "/"

	dhs_df = pd.read_csv(SAT+"nigeria_2013.csv")
	cluster_avgs = pd.DataFrame( dhs_df.groupby('DHSCLUST').mean()['hv271'] )
	gps_df = pd.read_csv(SAT+"DHS_GPS.csv")

	output_csv = gps_df.merge(cluster_avgs, how='left', left_on='DHSCLUSTER', right_index=True)

	if not os.path.exists(REG+"size"+str(img_size)+"/"):
		os.makedirs(REG+"size"+str(img_size)+"/")


	output_csv.to_csv(REG+"size"+str(img_size)+"/"+"cluster_averages.csv")

	# for each cluster location, get it's 10km x 10km area within the Landsat image
	cluster_count = output_csv.shape[0]

	too_close = []

	for i in range(cluster_count):
		plot_i = int(output_csv.iloc[i]['array_i'])
		plot_j = int(output_csv.iloc[i]['array_j'])
		cluster = int(output_csv.iloc[i]['DHSCLUSTER'])

		if plot_i < ADJ or plot_j < ADJ:
			too_close.append( (plot_i, plot_j) )
			continue

		if plot_i + ADJ > ls_data.shape[0] or plot_j + ADJ > ls_data.shape[1]:
			too_close.append( (plot_i, plot_j) )
			continue

		sub = ls_data[plot_i-ADJ: plot_i+ADJ, plot_j-ADJ: plot_j+ADJ, :]
		print("Shape of subimage", sub.shape)

		if not os.path.exists(REG+"size"+str(img_size)+"/"+str(cluster)):
			os.makedirs(REG+"size"+str(img_size)+"/"+str(cluster))

		partition(sub, img_size, REG+"size"+str(img_size)+"/"+str(cluster))
		assert sub.shape == (332, 332, 3)


	return output_csv, too_close


def partition(main_image, sub_img_size, save_to):
	'''
	Partitions the main_image into sub images with size equal to sub_img_size
	and saves out all resulting subimages to folder save_to. All images
	are squares

	Inputs:
		- main_image: (np array) of image to partition
		- sub_img_size: (int) of dimensions of sub-images
		- save_to: (str) folder to save sub images as .jpg

	Saves to disk:
		- all resulting subimages, numbered accordingly
	'''

	img_count = main_image.shape[0] // sub_img_size

	qc_main = main_image[0: img_count*sub_img_size, 0: img_count*sub_img_size].sum()
	qc_partitions = 0

	pic_num = 0

	for x in range(img_count):
		for y in range(img_count):
			x_coord = x*sub_img_size
			y_coord = y*sub_img_size

			sub = main_image[x_coord: x_coord+sub_img_size, y_coord: y_coord+sub_img_size]
			qc_partitions += sub.sum() 

			filename = "pic_{}.png".format(pic_num)

			misc.imsave(arr=sub, name=save_to+"/"+filename)
			pic_num += 1

	assert np.abs(qc_main - qc_partitions) < 1

if __name__ == '__main__':

	df, too_close = build_reg_data(128)
	print("There were {} clusters on the boundary".format(len(too_close)))
	for x in too_close:
		print(x)

	df, too_close = build_reg_data(34)
	print("There were {} clusters on the boundary".format(len(too_close)))
	for x in too_close:
		print(x)


