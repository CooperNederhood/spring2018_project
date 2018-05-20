import fiona
import pandas as pd 
import skimage.external.tifffile as tiff

# data paths
DATA = "../data"
DHS = DATA+"/DHS/unzipped/"
SAT = DATA+"/satellite/"

def create_gps_dataframe(year, file):
    '''
    Given a year of analysis, returns a pandas dataframe 
    of the GPS information corresponding to the DHS survey 
    of that year. GPS data is originally stored as shapefile

    Inputs:
        - year (int or str): year of DHS survey
        - file (str): file name (.shp file)

    Returns:
        - df (pandas DF): of GPS data
    '''


    shape = fiona.open(DHS+"/"+str(year)+"/"+file)
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
        - file (str): file name (.DTA file)

    Returns:
        - df (pandas DF): of DHS data
    '''

    df = pd.read_stata(DHS+"/"+str(year)+"/"+file)

    return df 


def gen_DHS(year, file_DHS, file_GPS):
    '''
    Returns a dataframe of the necessary DHS wealth indices, with
    the GPS data

    Inputs:
        - year (int or str): year of DHS survey
        - file_DHS (str): DHS survey file  (.DTA file)
        - file_GPS (str): DHS GPS file (.shp extension)
    '''

    df_DHS = create_dhs_dataframe(year, file_DHS)
    print("Loaded DHS survey for year:", year)
    print(df_DHS.head())

    df_GPS = create_gps_dataframe(year, file_GPS)
    print("Loaded DHS GPS data for year:", year)
    print(df_GPS.head())

    keep_DHS = ['hv001', 'hv270', 'hv271']

    df = df_DHS[ keep_DHS ].merge(df_GPS, how='left', left_on='hv001', right_on='DHSCLUST')

    # gen summary dataframe
    #df_means = df[ ['hv001', 'hv271', 'LATNUM', 'LONGNUM'] ].groupby('hv001').mean()
    #df_size = df['hv001'].value_counts()

    #summ_stats = df_means.merge(pd.DataFrame(df_size), how='left', left_index=True, right_index=True)

    return df

def write_EE_points(year, file, ee_script_name):
    '''
    Given a year of analysis, writes an Earth Engine compatable
    script to a txt file which creates Points corresponding
    to the GPS locations of the DHS clusters

    Inputs:
        - year (int or str): year of DHS survey
        - file (str): DHS GPS file (.shp extension)
        - ee_script_name (str): text file of EE script to create

    Saves to disk:
        - ee_script (txt file): containing EE code for making points
    '''

    gps_data = create_gps_dataframe(year, file)
    gps_data = gps_data[ ['DHSCLUST', 'LATNUM', 'LONGNUM'] ]

    N = gps_data.shape[0]

    with open(ee_script_name, 'w') as ee_script:
        ee_script.write('INSTRUCTIONS: below is code that can be pasted into Google Earth Engine to create boundaries for Kinshasa')
        ee_script.write('\t The code creates a list of Earth Engine geometries which can then be aggregated and cast as a raster file for export')
        ee_script.write('\n\n\n')
        #ee_script.write('var features = [')
        ee_script.write('\n\n')

        all_lines = []
        for i in range(N):
            lon = gps_data.iloc[i]['LONGNUM']
            lat = gps_data.iloc[i]['LATNUM']
            dhs_code = int(gps_data.iloc[i]['DHSCLUST'])
            #cur_line = "ee.Feature(ee.Geometry.Point( {}, {} ), {{ code:{} }}) ".format(lon, lat, dhs_code)
            cur_line = "var pt{} = ee.Feature(ee.Geometry.Point( {}, {} ), {{ code:{} }}); \n ".format(dhs_code, lon, lat, dhs_code)
            ee_script.write(cur_line)
            all_lines.append("pt{}".format(dhs_code))

        command_str = ", ".join(all_lines)
        ee_script.write("\n \n")
        final_line = "var ft_collection = ee.FeatureCollection([ " + command_str + " ]) ;"
        ee_script.write(final_line)
        ee_script.write("\n ] \n")
        ee_script.write("\n END OF GOOGLE EARTH ENGINE CODE")
        ee_script.close()

def GPS_tiff_to_csv(tif_file, output_file_name):
    '''
    The write_EE_points script generates an Earth Engine 
    script which then creates a tiff file indicating DHS cluster locations.
    Scan the tiff file and save out a csv file mapping a DHS Cluster ID
    to its coordinates within our tiff files

    Inputs:
        - tif_file (str): of tiff GPS file (.tiff extension)
        - output_file_name (str): of csv file to save to (.csv extension)

    Saves to disk:
        - see output_file_name
    '''

    data = tiff.imread(SAT+tif_file)
    i_size, j_size = data.shape 

    pd_data = []

    for i in range(i_size):
        for j in range(j_size):
            if data[i][j] != 0:

                obs_dict = {'DHSCLUSTER': data[i][j],
                            'array_i': i,
                            'array_j': j}
                print("Found cluster {} at i={}, j={}".format(data[i][j], i, j))
                pd_data.append(obs_dict)

    data_df = pd.DataFrame(pd_data)

    data_df.to_csv(SAT+output_file_name)


def do_EE_analysis():
    '''
    Actual lines of analysis run
    '''
    y = 2013
    dhs_file = "NGHR6AFL.DTA"
    gps_file = "NGGE6AFL.shp"


    write_EE_points(y, gps_file, "ee_gps.txt")

def do_GPS_tiff_do_csv():
    '''
    Actual lines of analysis run
    '''

    gps_tiff = "DHS_GPS.tif"
    gps_csv_file = "DHS_GPS.csv"

    GPS_tiff_to_csv(gps_tiff, gps_csv_file)

if __name__ == '__main__':

    year = 2013
    file_DHS = "NGHR6AFL.DTA"
    file_GPS = "NGGE6AFL.shp"

    df = gen_DHS(year, file_DHS, file_GPS)

