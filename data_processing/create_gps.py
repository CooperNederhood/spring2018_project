import fiona
import pandas as pd 

# hardcoded path is temporary
PATH = "/home/cooper/Documents/spring2018_project/data/DHS/unzipped/"

def create_gps_dataframe(year, file):
	'''
	Given a year of analysis, returns a pandas dataframe 
	of the GPS information corresponding to the DHS survey 
	of that year. GPS data is originally stored as shapefile

	Inputs:
		- year (int or str): year of DHS survey
		- file (str): file name

	Returns:
		- df (pandas DF): of GPS data
	'''


	shape = fiona.open(PATH+"/"+str(year)+"/"+file)
	obs_index = 0

	for obs in shape:
		ordered_dict = obs['properties']

		new_df = pd.DataFrame.from_records(ordered_dict, index=[obs_index])

		if obs_index == 0:
			df = new_df

		else:
			df = df.append(new_df)

		obs_index += 1

	return df 
	
def create_dhs_dataframe(year, file):
	'''
	Given a year and DHS survey filename returns the 
	corresponding pandas DF

	Inputs:
		- year (int or str): year of DHS survey
		- file (str): file name

	Returns:
		- df (pandas DF): of DHS data
	'''

	df = pd.read_stata(PATH+"/"+str(year)+"/"+file)

	return df 


def gen_DHS(year, file_DHS, file_GPS):

	df_DHS = create_dhs_dataframe(year, file_DHS)
	df_GPS = create_gps_dataframe(year, file_GPS)

	keep_DHS = ['hv001', 'hv270', 'hv271']

	df = df_DHS[ keep_DHS ].merge(df_GPS, how='left', left_on='hv001', right_on='DHSCLUST')

	# gen summary dataframe
	df_means = df[ ['hv001', 'hv271', 'LATNUM', 'LONGNUM'] ].groupby('hv001').mean()
	df_size = df['hv001'].value_counts()

	summ_stats = df_means.merge(pd.DataFrame(df_size), how='left', left_index=True, right_index=True)

	return df, summ_stats
