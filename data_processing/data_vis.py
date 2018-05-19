'''
This script includes utilities to do basic 
summary stats and data visualizations for the main datasets:
(a) nighttime luminosity
(b) Landsat imagery
(c) DHS assets
'''
import numpy as np
import numpy.random as rand 
import skimage.external.tifffile as tiff
import scipy.misc as misc
from matplotlib import pyplot as plt 
import pandas as pd 
import timeit 
import os
from PIL import Image 

DATA = "../data"
DHS = DATA+"/DHS/unzipped/"
SAT = DATA+"/satellite/"

OUTPUT_FOLDER = "data_vis/"

RADIUS = 2

def tiff_to_png(file_list):
	'''
	Saves versions of tiff files as csv
	'''

	for f in file_list:	
		new_name = f.replace(".tif", ".png")

		i = tiff.imread(SAT+f)
		
		i_png = Image.fromarray(i)
		i_png.save("new_name")



def sat_data_with_clusters(jpg_file, gps_csv_file):
	'''
	Saves a new version of the jpg file but plots the 
	coordinates of the DHS cluster locations

	Inputs:
		- jpg_file (str): of jpg file
		- gps_csv_file (str): of GPS DHS locations (.csv extension)

	Saves to disk:
		- jpg_coords - new jpg with plots on it
	'''

	im = Image.open(SAT+jpg_file)
	draw = ImageDraw.Draw(im)

	gps_csv = pd.read_csv(SAT+gps_csv_file)
	N = gps_csv.shape[0]

	for i in range(N):
		plot_i = gps_csv.iloc[i][array_i]
		plot_j = gps_csv.iloc[j][array_j]
		bounds = [(plot_i-RADIUS, plot_j-RADIUS), (plot_i+RADIUS, plot_j+RADIUS)]

		draw.ellipse(bounds, fill=(255,0,0) )




# (a) Night-time luminosity
def summarize_luminosity(tiff_file, save_to):
	'''
	Given a tiff file of night-time luminosity,
	returns the
	'''

	pass


if __name__ == '__main__':

	files = ["SAT_DATA_LANDSAT.tif", "SAT_DATA_LUMINOSITY.tif"]

	tiff_to_png(files) 


