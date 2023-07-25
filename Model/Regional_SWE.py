import os
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
from pyproj import CRS, Transformer
from pystac_client import Client
import planetary_computer
from tqdm import tqdm
import math
import xarray
import rioxarray
import richdem as rd
import elevation
import pickle
import bz2
from datetime import date, datetime, timedelta

import National_Snow_Model_Regional

class Region_SWE_Simulation:
    """
    This class contains the necessary functions to run historical and up-to-date SWE simulations for a given shapefile region.
    Args:
        cwd (str): current working diectory. This should be the "Model" directory, along with the National_Snow_Model_Regional.py file.
        area (str): Name of the region to model. This should exactly match the name of the shapefile.
        year (str): 'YYYY' - Starting year of the Water Year to be modeled.
        shapefile_path (str): Path of the area shapefile.
        start_date (str): 'YYYY-MM-DD' - First date of model inference. If Regional_predict has not been executed before, this must be 'YYYY-10-01'.
        end_date (str): 'YYYY-MM-DD' - Final date of model infernce. 
        day_interval (int): Default = 7 days. Interval between model inferences. If changed, initial start_date must be 'YYYY-09-24' + interval
        plot (boolean): Default =True. Plot interactive map of modeled region SWE inline. Suggest setting False to improve speed and performance. 
    """ 
    def __init__(self, cwd, area, year):
        self = self
        self.cwd = cwd
        self.area = area
        self.year = year

        if not os.path.exists(self.cwd+'\\'+area):
            os.makedirs(self.cwd+'\\'+area)
            print('Created new directory:', self.cwd+'\\'+area)
        if not os.path.exists(self.cwd+'\\'+area+'\\Predictions'):
            os.makedirs(self.cwd+'\\'+area+'\\Predictions')
            print('Created new directory:', self.cwd+'\\'+area+'\\Predictions')
        if not os.path.exists(self.cwd+'\\'+area+'\\Data\\Processed'):
            os.makedirs(self.cwd+'\\'+area+'\\Data\\Processed')
            print('Created new directory:', self.cwd+'\\'+area+'\\Data\\Processed')
        if not os.path.exists(self.cwd+'\\'+area+'\\Data\\NetCDF'):
            os.makedirs(self.cwd+'\\'+area+'\\Data\\NetCDF')
            print('Created new directory:', self.cwd+'\\'+area+'\\Data\\NetCDF')
        if not os.path.exists(self.cwd+'\\'+area+'\\Data\\WBD'):
            os.makedirs(self.cwd+'\\'+area+'\\Data\\WBD')

    # @staticmethod
    def PreProcess(self, shapefile_path):
        """ Create a grid of points centered at 1 km grid cells within a shapefile boundary.
            This may take a long time for larger areas. Once Geo_df.csv is made for an area, it does not need to be executed again."""

        gdf_shapefile = gpd.read_file(shapefile_path)

        # Get bounding box coordinates
        minx, miny, maxx, maxy = gdf_shapefile.total_bounds

        #buffer the bounds
        minx = minx-1
        maxx = maxx+1
        miny = miny-1
        maxy = maxy+1

        # Define the source and target coordinate reference systems
        src_crs = CRS('EPSG:4326')  # WGS84

        if -126 < minx < -120:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32610', always_xy=True)
            target_crs = CRS('EPSG:32610') #UTM zone 10
        elif -120 < minx < -114:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32611', always_xy=True)
            target_crs = CRS('EPSG:32611') #UTM zone 11
        elif -114 < minx < -108:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32612', always_xy=True)
            target_crs = CRS('EPSG:32612') #UTM zone 12
        elif -108 < minx < -102:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32613', always_xy=True)
            target_crs = CRS('EPSG:32613') #UTM zone 13
        else:
            # transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            target_crs = CRS('EPSG:3857') #Web Mercator

        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

        # Convert the bounding box coordinates to Web Mercator
        minx, miny = transformer.transform(minx, miny)
        maxx, maxy = transformer.transform(maxx, maxy)

        # set the grid cell size in meters
        cell_size = 1000

        # Calculate the number of cells in x and y directions
        num_cells_x = int((maxx-minx)/cell_size)
        num_cells_y = int((maxy-miny)/cell_size)

        # Calculate the total grid width and height
        grid_width = num_cells_x*cell_size
        grid_height = num_cells_y*cell_size

        # Calculate the offset to center the grid
        offset_x = ((maxx-minx)-grid_width)/2
        offset_y = ((maxy-miny)-grid_height)/2

        # Generate latitude and longitude ranges
        lon_range = np.linspace(minx + offset_x, maxx - offset_x, num=num_cells_x)
        lat_range = np.linspace(miny + offset_y, maxy - offset_y, num=num_cells_y)

        # Create a grid of points
        points = []
        for lon in lon_range:
            for lat in lat_range:
                points.append((lon, lat))

        # Convert the coordinate pairs back to WGS84
        back_transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
        target_coordinates = [back_transformer.transform(lon, lat) for lon, lat in points]

        # Create a list of Shapely Point geometries
        coords = [Point(lon, lat) for lon, lat in target_coordinates]

        # Create a GeoDataFrame from the points
        gdf_points = gpd.GeoDataFrame(geometry=coords)
        # set CRS to WGS84
        gdf_points=gdf_points.set_crs('epsg:4326')

        # Clip the points to the shapefile boundary
        gdf_clipped_points = gpd.clip(gdf_points, gdf_shapefile)

        # Specify the output points shapefile path
        output_shapefile = self.cwd+'\\'+self.area+'\\'+self.area+'_points.shp'

        # Export the clipped points to a shapefile
        gdf_clipped_points.to_file(output_shapefile)

        print("Regional Grid Created")


        #Create Submission format .csv for SWE predictions
        gdf_clipped_points.index.names = ['cell_id']

        Geospatial_df = pd.DataFrame()
        Geospatial_df['lon']= gdf_clipped_points['geometry'].x
        Geospatial_df['lat']= gdf_clipped_points['geometry'].y



        ### Begin process to import geospatial features into DF

        min_lon = min(Geospatial_df['lon'])
        min_lat = min(Geospatial_df['lat'])

        # Define the source and target coordinate reference systems
        src_crs = CRS('EPSG:4326')  # WGS84

        if -126 < min_lon < -120:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32610', always_xy=True)
            target_crs = CRS('EPSG:32610') #UTM zone 10
        elif -120 < min_lon < -114:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32611', always_xy=True)
            target_crs = CRS('EPSG:32611') #UTM zone 11
        elif -114 < min_lon < -108:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32612', always_xy=True)
            target_crs = CRS('EPSG:32612') #UTM zone 12
        elif -108 < min_lon < -102:
            # transformer = Transformer.from_crs(src_crs, 'epsg:32613', always_xy=True)
            target_crs = CRS('EPSG:32613') #UTM zone 13
        else:
            # transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
            target_crs = CRS('EPSG:3857') #Web Mercator

        transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)

        # Convert the bounding box coordinates to Web Mercator
        Geospatial_df['lon_m'], Geospatial_df['lat_m'] = transformer.transform(Geospatial_df['lon'], Geospatial_df['lat'])

        geocols=['BR_Coord_Long', 'BR_Coord_Lat', 'UR_Coord_Long', 'UR_Coord_Lat',
            'UL_Coord_Long', 'UL_Coord_Lat', 'BL_Coord_Long', 'BL_Coord_Lat']

        Geospatial_df = Geospatial_df.reindex(columns=[*Geospatial_df.columns.tolist(), *geocols], fill_value=0)

        Geospatial_df = Geospatial_df.assign(BR_Coord_Long=lambda x: x.lon_m + 500,
                            BR_Coord_Lat=lambda x: x.lat_m - 500,
                            UR_Coord_Long=lambda x: x.lon_m + 500,
                            UR_Coord_Lat=lambda x: x.lat_m + 500,
                            UL_Coord_Long=lambda x: x.lon_m - 500,
                            UL_Coord_Lat=lambda x: x.lat_m + 500,
                            BL_Coord_Long=lambda x: x.lon_m - 500,
                            BL_Coord_Lat=lambda x: x.lat_m - 500,)
        
        transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
        Geospatial_df['BR_Coord_Long'], Geospatial_df['BR_Coord_Lat']=transformer.transform(Geospatial_df['BR_Coord_Long'], Geospatial_df['BR_Coord_Lat'])
        Geospatial_df['UR_Coord_Long'], Geospatial_df['UR_Coord_Lat']=transformer.transform(Geospatial_df['UR_Coord_Long'], Geospatial_df['UR_Coord_Lat']) 
        Geospatial_df['UL_Coord_Long'], Geospatial_df['UL_Coord_Lat']=transformer.transform(Geospatial_df['UL_Coord_Long'], Geospatial_df['UL_Coord_Lat']) 
        Geospatial_df['BL_Coord_Long'], Geospatial_df['BL_Coord_Lat']=transformer.transform(Geospatial_df['BL_Coord_Long'], Geospatial_df['BL_Coord_Lat']) 

        area_of_interest = {"type": "Polygon","coordinates": [
            [
                #lower left
                [Geospatial_df['BL_Coord_Long'].min(), Geospatial_df['BL_Coord_Lat'].min()],
                #upper left
                [Geospatial_df['UL_Coord_Long'].min(), Geospatial_df['UL_Coord_Lat'].max()],
                #upper right
                [Geospatial_df['UR_Coord_Long'].max(), Geospatial_df['UR_Coord_Lat'].max()],
                #lower right
                [Geospatial_df['BR_Coord_Long'].max(), Geospatial_df['BR_Coord_Lat'].min()],
                #lower left
                [Geospatial_df['BL_Coord_Long'].min(), Geospatial_df['BL_Coord_Lat'].min()],
                ]],}
        

        #Make a connection to get 90m Copernicus Digital Elevation Model (DEM) data with the Planetary Computer STAC API

        client = Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            ignore_conformance=True,
        )


        search = client.search(
            collections=["cop-dem-glo-90"],
            intersects=area_of_interest
        )

        tiles = list(search.get_items())

        #Make a DF to connect locations with the larger data tile, and then extract elevations
        regions = []

        print("Retrieving Copernicus 90m DEM tiles")
        for i in tqdm(range(0, len(tiles))):
            row = [i, tiles[i].id]
            regions.append(row)
        regions = pd.DataFrame(columns = ['sliceID', 'tileID'], data = regions)
        regions = regions.set_index(regions['tileID'])
        del regions['tileID']

        #added Long,Lat to get polygon points
        def GeoStat_func(i, Geospatial_df, regions, elev_L, slope_L, aspect_L, Long, Lat, tile):

            # convert coordinate to raster value
            lon = Geospatial_df.iloc[i][Long]
            lat = Geospatial_df.iloc[i][Lat]

            #connect point location to geotile
            tileid = 'Copernicus_DSM_COG_30_N' + str(math.floor(lat)) + '_00_W'+str(math.ceil(abs(lon))) +'_00_DEM'
            
            indexid = regions.loc[tileid]['sliceID']

            #Assing region
            signed_asset = planetary_computer.sign(tiles[indexid].assets["data"])
            #get elevation data in xarray object
            elevation = rioxarray.open_rasterio(signed_asset.href)

            #create copies to extract other geopysical information
            #Create Duplicate DF's
            slope = elevation.copy()
            aspect = elevation.copy()
                
            
            #transform projection
            transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
            xx, yy = transformer.transform(lon, lat)
            
            #extract elevation values into numpy array
            tilearray = np.around(elevation.values[0]).astype(int)

            #set tile geo to get slope and set at rdarray
            geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)
            tilearray = rd.rdarray(tilearray, no_data = -9999)
            tilearray.projection = 'EPSG:4326'
            tilearray.geotransform = geo

            #get slope, note that slope needs to be fixed, way too high
            #get aspect value
            slope_arr = rd.TerrainAttribute(tilearray, attrib='slope_degrees')
            aspect_arr = rd.TerrainAttribute(tilearray, attrib='aspect')

            #save slope and aspect information 
            slope.values[0] = slope_arr
            aspect.values[0] = aspect_arr
            
            elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
            slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
            asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])
            
            #add point values to list
            elev_L.append(elev)
            slope_L.append(slop)
            aspect_L.append(asp)

        print("Interpolating Grid Cell Spatial Features")
        ###---------------------------------------------------------- Need to Parallelize This ----------------------------------------------------------
        BLelev_L = []
        BLslope_L = []
        BLaspect_L = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, Geospatial_df, regions, BLelev_L, BLslope_L, BLaspect_L,
                        'BL_Coord_Long', 'BL_Coord_Lat', tiles) for i in tqdm(range(0, len(Geospatial_df)))]

        #Save each points elevation in DF
        Geospatial_df['BL_Elevation_m'] = BLelev_L
        Geospatial_df['BL_slope_Deg'] = BLslope_L
        Geospatial_df['BLaspect_L'] = BLaspect_L

        ULelev_L = []
        ULslope_L = []
        ULaspect_L = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, Geospatial_df, regions, ULelev_L, ULslope_L, ULaspect_L,
                        'UL_Coord_Long', 'UL_Coord_Lat', tiles) for i in tqdm(range(0,len(Geospatial_df)))]

        #Save each points elevation in DF
        Geospatial_df['UL_Elevation_m'] = ULelev_L
        Geospatial_df['UL_slope_Deg'] = ULslope_L
        Geospatial_df['ULaspect_L'] = ULaspect_L

        URelev_L = []
        URslope_L = []
        URaspect_L = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, Geospatial_df, regions, URelev_L, URslope_L, URaspect_L,
                        'UR_Coord_Long', 'UR_Coord_Lat', tiles) for i in tqdm(range(0,len(Geospatial_df)))]

        #Save each points elevation in DF
        Geospatial_df['UR_Elevation_m'] = URelev_L
        Geospatial_df['UR_slope_Deg'] = URslope_L
        Geospatial_df['URaspect_L'] = URaspect_L

        BRelev_L = []
        BRslope_L = []
        BRaspect_L = []

        #run the elevation function, added tqdm to show progress
        [GeoStat_func(i, Geospatial_df, regions, BRelev_L, BRslope_L, BRaspect_L,
                        'BR_Coord_Long', 'BR_Coord_Lat', tiles) for i in tqdm(range(0,len(Geospatial_df)))]

        #Save each points elevation in DF
        Geospatial_df['BR_Elevation_m'] = BRelev_L
        Geospatial_df['BR_slope_Deg'] = BRslope_L
        Geospatial_df['BRaspect_L'] = BRaspect_L

        #get mean Geospatial data
        def mean_Geo(df, geo):
            BL = 'BL'+geo
            UL = 'UL'+geo
            UR = 'UR'+geo
            BR = 'BR'+geo
            
            df[geo] = (df[BL] + df[UL]+ df[UR] + df[BR]) /4

        Geo_df = Geospatial_df.copy()

        #Get geaspatial means
        geospatialcols = ['_Coord_Long', '_Coord_Lat', '_Elevation_m', '_slope_Deg' , 'aspect_L']

        #Training data
        [mean_Geo(Geo_df, i) for i in geospatialcols]

        #list of key geospatial component means
        geocol = ['_Coord_Long','_Coord_Lat','_Elevation_m','_slope_Deg','aspect_L']
        Geo_df = Geo_df[geocol].copy()

        #adjust column names to be consistent with snotel
        Geo_df = Geo_df.rename( columns = {'_Coord_Long':'Long', '_Coord_Lat':'Lat', '_Elevation_m': 'elevation_m',
                                    '_slope_Deg':'slope_deg' , 'aspect_L': 'aspect'})
        
        #This function defines northness: :  sine(Slope) * cosine(Aspect). this gives you a northness range of -1 to 1.
        #Note you'll need to first convert to radians. 
        #Some additional if else statements to get around sites with low obervations
        def northness(df):    
            
            if len(df) == 8: #This removes single value observations, need to go over and remove these locations from training too
                #Determine northness for site
                #convert to radians
                df = pd.DataFrame(df).T
                
                df['aspect_rad'] = df['aspect']*0.0174533
                df['slope_rad'] = df['slope_deg']*0.0174533
                
                df['northness'] = -9999
                for i in range(0, len(df)):
                    df['northness'].iloc[i] = math.sin(df['slope_rad'].iloc[i])*math.cos(df['aspect_rad'].iloc[i])

                #remove slope and aspects to clean df up
                df = df.drop(columns = ['aspect', 'slope_deg', 'aspect_rad', 'slope_rad', 'Region'])
                
                return df
                
            else:
                #convert to radians
                df['aspect_rad'] = df['aspect']*0.0174533
                df['slope_rad'] = df['slope_deg']*0.0174533
                
                df['northness'] = -9999
                for i in range(0, len(df)):
                    df['northness'].iloc[i] = math.sin(df['slope_rad'].iloc[i])*math.cos(df['aspect_rad'].iloc[i])

                
                #remove slope and aspects to clean df up
                df = df.drop(columns = ['aspect', 'slope_deg', 'aspect_rad', 'slope_rad'])
                
                return df
            
        Geo_df = northness(Geo_df)

        Geo_df.to_csv(self.cwd+'\\'+self.area+'\\'+self.area+'_Geo_df.csv', index= True)


    def Prepare_Prediction(self):
        """ Create the necessary files to store SWE estimates. Estimates will begin on YYYY-09-24."""

        Geo_df = pd.read_csv(self.cwd+'\\'+self.area+'\\'+self.area+'_Geo_df.csv', index_col='cell_id')

        submission_format = Geo_df.drop(['Long','Lat','elevation_m','northness'], axis=1)
        submission_format.to_csv(self.cwd+'\\'+self.area+'\Predictions\submission_format_'+self.area+'.csv')

        #also need to save initial 9/24 submission format
        submission_format.to_csv(self.cwd+'\\'+self.area+'\Predictions\submission_format_'+self.area+'_'+str(self.year)+'-09-24.csv')

        print("Prediction CSV Created")

        Geo_df['Region']=""
        print('Defining Encompassing Model Regions')
        National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, str(self.year)+'-10-01', self.year).Region_id(Geo_df)     


        Region_df_dict = Geo_df.groupby("Region")

        # Convert the GroupBy DataFrame to a dictionary of DataFrames
        grouped_dict = {}
        for group_label, group_df in Region_df_dict:
            group_df=group_df.drop("Region",axis=1)
            grouped_dict[group_label] = group_df


        Region_list=National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, str(self.year)+'-10-01', self.year).Region_list

        init_dict = {}
        Region_Pred={}
        for region in Region_list:
            init_dict[region] = pd.read_hdf(self.cwd+'\Predictions\initialize_predictions.h5', key = region)
        
        for i in init_dict.keys():
            init_dict[i] = init_dict[i].drop(init_dict[i].index[1:])
            init_dict[i][str(self.year)+'-09-24'] = 0
            if i in grouped_dict.keys():
                init_dict[i] = init_dict[i].append(grouped_dict[i])
                init_dict[i]['WYWeek']=52
                init_dict[i] = init_dict[i].fillna(0)

            if len(init_dict[i]) > 1:
                init_dict[i] = init_dict[i].tail(-1)
            Region_Pred[i] = init_dict[i].iloc[:,:4].reset_index()

        Region_Pred['S_Sierras']=Region_Pred['S_Sierras_High'].merge(Region_Pred['S_Sierras_Low'], how='outer')
        del Region_Pred['S_Sierras_High'], Region_Pred['S_Sierras_Low']

        for region in Region_list:
            init_dict[region].to_hdf(self.cwd+'\\'+self.area+'\\Predictions\predictions'+str(self.year)+'-09-24.h5', key = region)


        o_path=self.cwd+'\\'+self.area+'\\Data\\Processed\Prediction_DF_'+str(self.year)+'-09-24.pkl'
        outfile = bz2.BZ2File(o_path, 'wb')
        pickle.dump(init_dict, outfile)
        outfile.close()

        outfile = self.cwd+'\\'+self.area+'\\Data\\Processed\Region_Pred.pkl'
        with open(outfile, 'wb') as pickle_file:
            pickle.dump(Region_Pred, pickle_file)
        pickle_file.close()

        GM_template = pd.read_csv(self.cwd+'/Data/Pre_Processed_DA/ground_measures_features_template.csv')
        GM_template = GM_template.rename(columns = {'Unnamed: 0': 'station_id', 'Date': str(self.year)+'-09-24'})
        GM_template.index =GM_template['station_id']
        cols = [str(self.year)+'-09-24']
        GM_template =GM_template[cols]
        GM_template[cols] = GM_template[cols].fillna(0)
        GM_template.to_csv(self.cwd+'\Data\Pre_Processed_DA\ground_measures_features_'+str(self.year)+'-09-24.csv')

        DA_template = pd.read_csv(self.cwd+'\Data\Processed\DA_ground_measures_features_template.csv')
        DA_template['Date'] = str(self.year)+'-09-24'
        DA_template.to_csv(self.cwd+'\\'+self.area+'\Data\Processed\DA_ground_measures_features_'+str(self.year)+'-09-24.csv')

        print('All initial files created for predictions beginning', 'October 1', str(self.year))



    def Regional_Predict(self, start_date, end_date, day_interval=7, plot =True):
        """ Create SWE estimates for the desired area. If first time executing, start_date must be YYYY-10-01."""
        # day interval can be altered to create list every n number of days by changing 7 to desired skip length.

        def daterange(start_date, end_date):
            for n in range(0, int((end_date - start_date).days) + 1, day_interval):
                yield start_date + timedelta(n)
                
        #create empty list to store dates
        datelist = []
        start_date = datetime.strptime(start_date, ("%Y-%m-%d"))
        end_date = datetime.strptime(end_date, ("%Y-%m-%d"))
        start_dt = date(start_date.year, start_date.month, start_date.day)
        end_dt = date(end_date.year, end_date.month, end_date.day)
        #append dates to list
        for dt in daterange(start_dt, end_dt):
            dt=dt.strftime('%Y-%m-%d')
            datelist.append(dt)
        
        #run the model through all time (data acqusition already completed)
        for date_ in datelist:
            print('Updating SWE predictions for ', date_)
            #connect interactive script to Wasatch Snow module, add model setup input for temporal resolution here.(eg. self.resolution = 7 days)
            Snow = National_Snow_Model_Regional.SWE_Prediction(self.cwd, self.area, date_, self.year, day_interval=day_interval)
            
            #Go get SNOTEL observations -- currently saving to csv, change to H5,
            #dd if self.data < 11-1-2022 and SWE = -9999, 
            Snow.Get_Monitoring_Data_Threaded()

            #Process observations into Model prediction ready format,
            Snow.Data_Processing()

            #Sometimes need to run twice for some reason, has a hard time loading the model (distributed computation engines for each region (multithreaded, cpus, GPUs))
            Snow.SWE_Predict()

            #Make CONUS netCDF file, compressed.
            Snow.netCDF_compressed(plot=False)
            if plot == True:
                #Make GeoDataframe and plot, self.Geo_df() makes the geo df.
                Snow.Geo_df()
                Snow.plot_interactive_SWE_comp(pinlat = 39.3, pinlong = -107, web = False)