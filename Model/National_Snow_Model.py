#Wasatch snow model
#Author: Ryan C. Johnson
#Date: 2022-6-29
#This script assimilates SNOTEL observations, processes the data into a model friendly
#format, then uses a calibrated multi-layered perceptron network to make 1 km x 1 km
#CONUS scale SWE estimates. 


#required modules
import copy
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle 
from pickle import dump
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import contextily as cx
import rasterio
import geopandas as gpd
from shapely.geometry import Point
import xarray as xr
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LinearSegmentedColormap
import folium
from folium import plugins
import branca.colormap as cm
import rioxarray as rxr
import earthpy as et
import earthpy.spatial as es
import datetime as dt
from netCDF4 import date2num,num2date
from osgeo import osr
import warnings
from pyproj import CRS
import requests
import geojson
import pandas as pd
from multiprocessing import Pool, cpu_count
from shapely.ops import unary_union
import json
import geopandas as gpd, fiona, fiona.crs
import webbrowser
warnings.filterwarnings("ignore")


class SWE_Prediction():
    def __init__(self, cwd, date, prevdate):
        self = self
        self.date = date
        self.prevdate = prevdate
        self.cwd = cwd
        
         #make other date tags
        m = self.date[0:2]
        d = self.date[3:5]
        y = self.date[-4:]

        pm = self.prevdate[0:2]
        p = self.prevdate[3:5]
        py = self.prevdate[-4:]

        self.datekey = m[1]+'/'+d+'/'+y
        self.datecol = y+'-'+m+'-'+d
        self.prevcol = py+'-'+pm+'-'+p
        
        #Define Model Regions
        self.Region_list = ['N_Sierras',
                       'S_Sierras_High',
                       'S_Sierras_Low',
                       'Greater_Yellowstone',
                       'N_Co_Rockies',
                       'SW_Mont',
                       'SW_Co_Rockies',
                       'GBasin',
                       'N_Wasatch',
                       'N_Cascade',
                       'S_Wasatch',
                       'SW_Mtns',
                       'E_WA_N_Id_W_Mont',
                       'S_Wyoming',
                       'SE_Co_Rockies',
                       'Sawtooth',
                       'Ca_Coast',
                       'E_Or',
                       'N_Yellowstone',
                       'S_Cascade',
                       'Wa_Coast',
                       'Greater_Glacier',
                       'Or_Coast'
                      ]
        
        #Original Region List needed to remove bad features
        self.OG_Region_list = ['N_Sierras',
                   'S_Sierras',
                   'Greater_Yellowstone',
                   'N_Co_Rockies',
                   'SW_Mont',
                   'SW_Co_Rockies',
                   'GBasin',
                   'N_Wasatch',
                   'N_Cascade',
                   'S_Wasatch',
                   'SW_Mtns',
                   'E_WA_N_Id_W_Mont',
                   'S_Wyoming',
                   'SE_Co_Rockies',
                   'Sawtooth',
                   'Ca_Coast',
                   'E_Or',
                   'N_Yellowstone',
                   'S_Cascade',
                   'Wa_Coast',
                   'Greater_Glacier',
                   'Or_Coast'
                  ]
      

    #make Region identifier. The data already includes Region, but too many 'other' labels

    def Region_id(self, df):
        
        
        #put obervations into the regions
        for i in tqdm(range(0, len(df))):

            #Sierras
            #Northern Sierras
            if -122.5 <= df['Long'][i] <=-119 and 39 <=df['Lat'][i] <= 42:
                loc = 'N_Sierras'
                df['Region'].iloc[i] = loc

            #Southern Sierras
            if -121.2 <= df['Long'][i] <=-117 and 35 <=df['Lat'][i] <= 39:
                loc = 'S_Sierras'
                df['Region'].iloc[i] = loc

            #West Coast    
            #CACoastal (Ca-Or boarder)
            if df['Long'][i] <=-122.5 and df['Lat'][i] <= 42:
                loc = 'Ca_Coast'
                df['Region'].iloc[i] = loc

            #Oregon Coastal (Or)?
            if df['Long'][i] <=-122.7 and 42<= df['Lat'][i] <= 46:
                loc = 'Or_Coast'
                df['Region'].iloc[i] = loc

            #Olympis Coastal (Wa)
            if df['Long'][i] <=-122.5 and 46<= df['Lat'][i]:
                loc = 'Wa_Coast'
                df['Region'].iloc[i] = loc    

            #Cascades    
             #Northern Cascades
            if -122.5 <= df['Long'][i] <=-119.4 and 46 <=df['Lat'][i]:
                loc = 'N_Cascade'
                df['Region'].iloc[i] = loc

            #Southern Cascades
            if -122.7 <= df['Long'][i] <=-121 and 42 <=df['Lat'][i] <= 46:
                loc = 'S_Cascade'
                df['Region'].iloc[i] = loc

            #Eastern Cascades and Northern Idaho and Western Montana
            if -119.4 <= df['Long'][i] <=-116.4 and 46 <=df['Lat'][i]:
                loc = 'E_WA_N_Id_W_Mont'
                df['Region'].iloc[i] = loc
            #Eastern Cascades and Northern Idaho and Western Montana
            if -116.4 <= df['Long'][i] <=-114.1 and 46.6 <=df['Lat'][i]:
                loc = 'E_WA_N_Id_W_Mont'
                df['Region'].iloc[i] = loc

            #Eastern Oregon
            if -121 <= df['Long'][i] <=-116.4 and 43.5 <=df['Lat'][i] <= 46:
                loc = 'E_Or'
                df['Region'].iloc[i] = loc

            #Great Basin
            if -121 <= df['Long'][i] <=-112 and 42 <=df['Lat'][i] <= 43.5:
                loc = 'GBasin'
                df['Region'].iloc[i] = loc

            if -119 <= df['Long'][i] <=-112 and 39 <=df['Lat'][i] <= 42:
                loc = 'GBasin'
                df['Region'].iloc[i] = loc
                #note this section includes mojave too
            if -117 <= df['Long'][i] <=-113.2 and df['Lat'][i] <= 39:
                loc = 'GBasin'
                df['Region'].iloc[i] = loc


            #SW Mtns (Az and Nm)
            if -113.2 <= df['Long'][i] <=-107 and df['Lat'][i] <= 37:
                loc = 'SW_Mtns'
                df['Region'].iloc[i] = loc


            #Southern Wasatch + Utah Desert Peaks
            if -113.2 <= df['Long'][i] <=-109 and 37 <= df['Lat'][i] <= 39:
                loc = 'S_Wasatch'
                df['Region'].iloc[i] = loc
            #Southern Wasatch + Utah Desert Peaks
            if -112 <= df['Long'][i] <=-109 and 39 <= df['Lat'][i] <= 40:
                loc = 'S_Wasatch'
                df['Region'].iloc[i] = loc

            #Northern Wasatch + Bear River Drainage
            if -112 <= df['Long'][i] <=-109 and 40 <= df['Lat'][i] <= 42.5:
                loc = 'N_Wasatch'
                df['Region'].iloc[i] = loc

            #YellowStone, Winds, Big horns
            if -111 <= df['Long'][i] <=-106.5 and 42.5 <= df['Lat'][i] <= 45.8:
                loc = 'Greater_Yellowstone'
                df['Region'].iloc[i] = loc

            #North of YellowStone to Boarder
            if -112.5 <= df['Long'][i] <=-106.5 and 45.8 <= df['Lat'][i]:
                loc = 'N_Yellowstone'
                df['Region'].iloc[i] = loc

             #SW Montana and nearby Idaho
            if -112 <= df['Long'][i] <=-111 and 42.5 <= df['Lat'][i] <=45.8:
                loc = 'SW_Mont'
                df['Region'].iloc[i] = loc 
             #SW Montana and nearby Idaho
            if -113 <= df['Long'][i] <=-112 and 43.5 <= df['Lat'][i] <=45.8:
                loc = 'SW_Mont'
                df['Region'].iloc[i] = loc
            #SW Montana and nearby Idaho
            if -113 <= df['Long'][i] <=-112.5 and 45.8 <= df['Lat'][i] <=46.6:
                loc = 'SW_Mont'
                df['Region'].iloc[i] = loc
             #Sawtooths, Idaho
            if -116.4 <= df['Long'][i] <=-113 and 43.5 <= df['Lat'][i] <=46.6:
                loc = 'Sawtooth'
                df['Region'].iloc[i] = loc

            #Greater Glacier
            if -114.1 <= df['Long'][i] <=-112.5 and 46.6 <= df['Lat'][i]:
                loc = 'Greater_Glacier'
                df['Region'].iloc[i] = loc 

             #Southern Wyoming 
            if -109 <= df['Long'][i] <=-104.5 and 40.99 <= df['Lat'][i] <= 42.5 :
                loc = 'S_Wyoming'
                df['Region'].iloc[i] = loc 
            #Southern Wyoming
            if -106.5 <= df['Long'][i] <=-104.5 and 42.5 <= df['Lat'][i] <= 43.2:
                loc = 'S_Wyoming'
                df['Region'].iloc[i] = loc 
                
             #Northern Colorado Rockies
            if -109 <= df['Long'][i] <=-104.5 and 38.3 <= df['Lat'][i] <= 40.99:
                loc = 'N_Co_Rockies'
                df['Region'].iloc[i] = loc 

             #SW Colorado Rockies
            if -109 <= df['Long'][i] <=-106 and 36.99 <= df['Lat'][i] <= 38.3:
                loc = 'SW_Co_Rockies'
                df['Region'].iloc[i] = loc 

            #SE Colorado Rockies + Northern New Mexico
            if -106 <= df['Long'][i] <=-104.5 and 34 <= df['Lat'][i] <= 38.3:
                loc = 'SE_Co_Rockies'
                df['Region'].iloc[i] = loc  
            #SE Colorado Rockies + Northern New Mexico
            if -107 <= df['Long'][i] <=-106 and 34 <= df['Lat'][i] <= 36.99:
                loc = 'SE_Co_Rockies'
                df['Region'].iloc[i] = loc 
                
                
                
                
    #Data Assimilation script, takes date and processes to run model.            
    def Data_Processing(self):
          

        #load ground truth values (SNOTEL): Testing
        obs_path = self.cwd+'/Data/Pre_Processed/ground_measures_features_' + self.date + '.csv'
        self.GM_Test = pd.read_csv(obs_path)

        #load ground truth values (SNOTEL): previous week, these have Na values filled by prev weeks obs +/- mean region Delta SWE
        obs_path = self.cwd+'/Data/Processed/DA_ground_measures_features_' + self.prevdate + '.csv'
        self.GM_Prev = pd.read_csv(obs_path)
        colrem = ['Region','Prev_SWE','Delta_SWE']
        self.GM_Prev = self.GM_Prev.drop(columns =colrem)


        #All coordinates of 1 km polygon used to develop ave elevation, ave slope, ave aspect
        path = self.cwd+'/Data/Processed/RegionVal.pkl'
        #load regionalized geospatial data
        self.RegionTest = open(path, "rb")
        self.RegionTest = pickle.load(self.RegionTest)

        ### Load H5 previous prediction files into dictionary

        self.prev_SWE= {}
        for region in self.Region_list:
            self.prev_SWE[region] = pd.read_hdf(self.cwd+'/Predictions/predictions'+ self.prevcol+'.h5', region)  #this was
            self.prev_SWE[region] = pd.DataFrame(self.prev_SWE[region][self.prevcol])
            self.prev_SWE[region] = self.prev_SWE[region].rename(columns = {self.prevcol: 'prev_SWE'})

        #change first column to station id
        self.GM_Test = self.GM_Test.rename(columns = {'Unnamed: 0':'station_id'})
        self.GM_Prev = self.GM_Prev.rename(columns = {'Unnamed: 0':'station_id'})


        #Fill NA observations
        self.GM_Test[self.datekey] = self.GM_Test[self.datekey].fillna(-9999)

        #drop na and put into modeling df format
        self.GM_Test = self.GM_Test.melt(id_vars=["station_id"]).dropna()

        #change variable to Date and value to SWE
        self.GM_Test = self.GM_Test.rename(columns ={'variable': 'Date', 'value':'SWE'})

        #load ground truth meta
        self.GM_Meta = pd.read_csv(self.cwd+'/Data/Pre_Processed/ground_measures_metadata.csv')

        #merge testing ground truth location metadata with snotel data
        self.GM_Test = self.GM_Meta.merge(self.GM_Test, how='inner', on='station_id')
        self.GM_Test = self.GM_Test.set_index('station_id')
        self.GM_Prev = self.GM_Prev.set_index('station_id')

        self.GM_Test.rename(columns={'name': 'location', 'latitude': 'Lat', 'longitude': 'Long', 'value': 'SWE'}, inplace=True)

        #drop NA columns from initial observations
        prev_index = self.GM_Prev.index
        self.GM_Test = self.GM_Test.loc[prev_index]

        #Make a dictionary for current snotel observations
        self.Snotel = self.GM_Test.copy()
        self.Snotel['Region'] = 'other'
        self.Region_id(self.Snotel)
        self.RegionSnotel  = {name: self.Snotel.loc[self.Snotel['Region'] == name] for name in self.Snotel.Region.unique()}

        #Make a dictionary for previous week's snotel observations
        self.prev_Snotel = self.GM_Prev.copy()
        self.prev_Snotel['Region'] = 'other'
        self.Region_id(self.prev_Snotel)
        self.prev_RegionSnotel  = {name: self.prev_Snotel.loc[self.prev_Snotel['Region'] == name] for name in self.prev_Snotel.Region.unique()}
        



        #add week number to observations
        for i in self.RegionTest.keys():
            self.RegionTest[i] = self.RegionTest[i].reset_index(drop=True)
            self.RegionTest[i]['Date'] = pd.to_datetime(self.RegionSnotel[i]['Date'][0])
            self.week_num(i)

        #set up dataframe to save to be future GM_Pred
        col = list(self.GM_Test.columns)+['Region']
        self.Future_GM_Pred = pd.DataFrame(columns = col)

        for region in self.OG_Region_list:
            self.NaReplacement(region)
            self.RegionSnotel[region]['Prev_SWE'] =self.prev_RegionSnotel[region]['SWE']
            self.RegionSnotel[region]['Delta_SWE'] = self.RegionSnotel[region]['SWE'] - self.RegionSnotel[region]['Prev_SWE']

            #make dataframe to function as next forecasts GM_Prev
            self.Future_GM_Pred = self.Future_GM_Pred.append(self.RegionSnotel[region])

        #Need to save 'updated non-na' df's
        GM_path = self.cwd+'/Data/Processed/DA_ground_measures_features_'+ self.date + '.csv'

        self.Future_GM_Pred.to_csv(GM_path)


        #This needs to be here to run in next codeblock
        self.Regions = list(self.RegionTest.keys()).copy()


        #Make dictionary in Regions dict for each region's dictionary of Snotel sites
    #Regions = list(RegionTrain.keys()).copy()

        for i in tqdm(self.Regions):

            snotel = i+'_Snotel'
            self.RegionTest[snotel] = {site: self.RegionSnotel[i].loc[site] for site in self.RegionSnotel[i].index.unique()}

           #get training and testing sites that are the same
            test = self.RegionTest[snotel].keys()

            for j in test:
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].to_frame().T
            #remove items we do not need
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].drop(columns = ['Long', 'Lat', 'location',
                                                                              'elevation_m', 'state', 'Region'])
            #make date index
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].set_index('Date')

            #rename columns to represent site info
                colnames = self.RegionTest[snotel][j].columns
                sitecolnames = [x +'_'+ j for x in colnames]
                names = dict(zip(colnames, sitecolnames))
                self.RegionTest[snotel][j] = self.RegionTest[snotel][j].rename(columns = names)

            #make a df for training each region, 

        for R in tqdm(self.Regions):
            snotels = R+'_Snotel'  
           # RegionTest[R] = RegionTest[R].reset_index()
           # print(R)
            sites = list(self.RegionTest[R]['cell_id'])
            sitelen = len(sites)-1
            self.RegionTest[R] = self.RegionTest[R].set_index('cell_id')


            for S in self.RegionTest[snotels].keys():
               # print(S)
                self.RegionTest[snotels][S] = self.RegionTest[snotels][S].append([self.RegionTest[snotels][S]]*sitelen, ignore_index = True)
                self.RegionTest[snotels][S].index = sites
                self.RegionTest[R]= pd.concat([self.RegionTest[R], self.RegionTest[snotels][S].reindex(self.RegionTest[R].index)], axis=1)

            self.RegionTest[R] = self.RegionTest[R].fillna(-9999.99)
            del self.RegionTest[R]['Date']


            #Perform the splitting for S_Sierras High and Low elevations
        self.RegionTest['S_Sierras_High'] =self.RegionTest['S_Sierras'].loc[self.RegionTest['S_Sierras']['elevation_m'] > 2500].copy()
        self.RegionTest['S_Sierras_Low'] = self.RegionTest['S_Sierras'].loc[self.RegionTest['S_Sierras']['elevation_m'] <= 2500].copy()
        del self.RegionTest['S_Sierras']


        #Add previous Cell SWE
        for region in self.Region_list:
            self.RegionTest[region] = pd.concat([self.RegionTest[region], self.prev_SWE[region]], axis =1, join = 'inner')


            #save dictionaries as pkl
        # create a binary pickle file 

        path = self.cwd+'/Data/Processed/ValidationDF_'+self.date + '.pkl'

        RVal = open(path,"wb")


        # write the python object (dict) to pickle file
        pickle.dump(self.RegionTest,RVal)


        # close file
        RVal.close()


    #Get the week number of the observations, from beginning of water year    
    def week_num(self, region):
        #week of water year
        weeklist = []

        for i in tqdm(range(0,len(self.RegionTest[region]))):
            if self.RegionTest[region]['Date'][i].month<11:
                y = self.RegionTest[region]['Date'][i].year-1
            else:
                y = self.RegionTest[region]['Date'][i].year

            WY_start = pd.to_datetime(str(y)+'-10-01')
            deltaday = self.RegionTest[region]['Date'][i]-WY_start
            deltaweek = round(deltaday.days/7)
            weeklist.append(deltaweek)


        self.RegionTest[region]['WYWeek'] = weeklist
        
        
   #NA Replacement script for necessary SNOTEL sites without observations     
    def NaReplacement(self, region):
        
        #Make NA values mean snowpack values
        meanSWE = np.mean(self.RegionSnotel[region]['SWE'][self.RegionSnotel[region]['SWE']>0])
        self.RegionSnotel[region]['SWE'][self.RegionSnotel[region]['SWE']<0]= meanSWE


        prev_meanSWE = np.mean(self.prev_RegionSnotel[region]['SWE'][self.prev_RegionSnotel[region]['SWE']>0])
        self.prev_RegionSnotel[region]['SWE'][self.prev_RegionSnotel[region]['SWE']<0]= prev_meanSWE

        delta = self.RegionSnotel[region]['SWE']-self.prev_RegionSnotel[region]['SWE']
        delta = pd.DataFrame(delta)
        delta = delta.rename(columns = {'SWE':'Delta'})

        #get values that are not affected by NA
        delta = delta[delta['Delta']> -9000]

        #Get mean Delta to adjust observed SWE
        meanD = np.mean(delta['Delta'])

        #go and fix current SWE observations
        #Get bad obsevations and snotel sites
        badSWE_df = self.RegionSnotel[region][self.RegionSnotel[region]['SWE']< 0].copy()
        bad_sites = list(badSWE_df.index)


        #remove bad observations from SWE obsevations
        self.RegionSnotel[region] = self.RegionSnotel[region][self.RegionSnotel[region]['SWE'] >= 0]

        #Fix bad observatoins by taking previous obs +/- mean delta SWE
        print('Fixing these bad sites in ', region, ':')
        for badsite in bad_sites:
            print(badsite)
            badSWE_df.loc[badsite,'SWE']=self.prev_RegionSnotel[region].loc[badsite]['SWE'] + meanD

        #Add observations back to DF
        self.RegionSnotel[region] = pd.concat([self.RegionSnotel[region], badSWE_df])


    #Take in and make prediction
    def SWE_Predict(self, plot):
        
        self.plot = plot
        #load first SWE observation forecasting dataset with prev and delta swe for observations. 
        path = self.cwd+'/Data/Processed/ValidationDF_'+self.date + '.pkl'


        #load regionalized forecast data
        self.Forecast = open(path, "rb")

        self.Forecast = pickle.load(self.Forecast)




        #load RFE optimized features
        self.Region_optfeatures= pickle.load(open(self.cwd+"/Model/Prev_SWE_Models_Final/opt_features_prevSWE.pkl", "rb"))


        #Reorder regions
        self.Forecast = {k: self.Forecast[k] for k in self.Region_list}


        #Make and save predictions for each reagion
        self.Prev_df = pd.DataFrame()
        self.predictions ={}
        print ('Making predictions for: ', self.datecol)

        for Region in self.Region_list:
            print(Region)
            self.predictions[Region] = self.Predict(Region)
            self.predictions[Region] = pd.DataFrame(self.predictions[Region])
            
            if self.plot == True:
                del self.predictions[Region]['geometry']
            self.Prev_df = self.Prev_df.append(pd.DataFrame(self.predictions[Region][self.datecol]))
            self.Prev_df = pd.DataFrame(self.Prev_df)

            self.predictions[Region].to_hdf(self.cwd+'/Predictions/predictions'+self.datecol+'.h5', key = Region)


        #load submission DF and add predictions
        self.subdf = pd.read_csv(self.cwd+'/Predictions/submission_format_'+self.prevcol+'.csv')
        self.subdf.index = list(self.subdf.iloc[:,0].values)
        self.subdf = self.subdf.iloc[:,1:]

        self.sub_index = self.subdf.index
        #reindex predictions
        self.Prev_df = self.Prev_df.loc[self.sub_index]
        self.subdf[self.datecol] = self.Prev_df[self.datecol].astype(float)
        #subdf.index.names = [' ']
        self.subdf.to_csv(self.cwd+'/Predictions/submission_format_'+self.datecol+'.csv')

        
            #set up model prediction function
    def Predict(self, Region):

        #region specific features
        features = self.Region_optfeatures[Region]

        #Make prediction dataframe
        forecast_data = self.Forecast[Region].copy()
        forecast_data = forecast_data[features]


        #change all na values to prevent scaling issues
        forecast_data[forecast_data< -9000]= -10


        #load and scale data

        #set up model checkpoint to be able to extract best models
        checkpoint_filepath = self.cwd+'/Model/Prev_SWE_Models_Final/' +Region+ '/'
        model = checkpoint_filepath+Region+'_model.h5'
        print(model)
        model=load_model(model)


        #load SWE scaler
        SWEmax = np.load(checkpoint_filepath+Region+'_SWEmax.npy')
        SWEmax = SWEmax.item()

        #load features scaler
        #save scaler data here too
        scaler =  pickle.load(open(checkpoint_filepath+Region+'_scaler.pkl', 'rb'))
        scaled = scaler.transform(forecast_data)
        x_forecast = pd.DataFrame(scaled, columns = forecast_data.columns)



         #make predictions and rescale
        y_forecast = (model.predict(x_forecast))

        y_forecast[y_forecast < 0 ] = 0
        y_forecast = (SWEmax * y_forecast)
        self.Forecast[Region][self.datecol] = y_forecast
         

        if self.plot == True:
            #plot predictions    
            plt.scatter( self.Forecast[Region]['elevation_m'],self.Forecast[Region][self.datecol], s=5, color="blue", label="Predictions")
            plt.xlabel('elevation m')
            plt.ylabel('Predicted SWE')
            plt.legend()


            #plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
            plt.title(Region)
            plt.show()


            #plot geolocation information
            _geom = [Point(xy) for xy in zip(self.Forecast[Region]['Long'], self.Forecast[Region]['Lat'])]
            _geom_df = gpd.GeoDataFrame(self.Forecast[Region], crs="EPSG:4326", geometry=_geom)

            dfmax = max(self.Forecast[Region][self.datecol])*1.05

            # fig, ax = plt.subplots(figsize=(14,6))
            ax = _geom_df.plot(self.datecol, cmap="cool", markersize=30,figsize=(25,25), legend=True, vmin=0, vmax=dfmax)#vmax=test_preds['delta'].max(), vmin=test_preds['delta'].min())
            cx.add_basemap(ax, alpha = .7, crs=_geom_df.crs.to_string())


            plt.show()
        
        return self.Forecast[Region]
    
    
    # construct a full grid
    def expand_grid(self, lat, lon):
        '''list all combinations of lats and lons using expand_grid(lat,lon)'''
        test = [(A,B) for A in lat for B in lon]
        test = np.array(test)
        test_lat = test[:,0]
        test_lon = test[:,1]
        full_grid = pd.DataFrame({'Long': test_lon, 'Lat': test_lat})
        full_grid = full_grid.sort_values(by=['Lat','Long'])
        full_grid = full_grid.reset_index(drop=True)
        return full_grid
    
    def netCDF(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.datecol]

        for region in self.Forecast:
            self.NA_SWE = self.NA_SWE.append(self.Forecast[region][columns])

        self.NA_SWE = self.NA_SWE.rename(columns = {self.datecol:'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(min(self.NA_SWE['Long'])-1, max(self.NA_SWE['Long'])+2, 0.01)
        self.latrange = np.arange(min(self.NA_SWE['Lat'])-1, max(self.NA_SWE['Lat']), 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]


        #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        target_variable_2D = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

        #put into xarray formate
        target_variable_xr = xr.DataArray(target_variable_2D, coords=[('lat', self.latrange),('lon', self.lonrange)])

        #set target variable name
        target_variable_xr = target_variable_xr.rename("SWE")

        #save as netCDF
        target_variable_xr.to_netcdf(self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+self.datecol+'.nc')
        
        #show plot
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()
            
    def netCDF2(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.datecol]

        for region in self.Forecast:
            self.NA_SWE = self.NA_SWE.append(self.Forecast[region][columns])

        self.NA_SWE = self.NA_SWE.rename(columns = {self.datecol:'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(min(self.NA_SWE['Long'])-1, max(self.NA_SWE['Long'])+2, 0.01)
        self.latrange = np.arange(min(self.NA_SWE['Lat'])-1, max(self.NA_SWE['Lat']), 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]


        #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        self.SWE_array = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

       # create nc filepath
        fn = self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+self.datecol+'.nc'
        
        # make nc file, set lat/long, time
        ds = nc.Dataset(fn, 'w', format = 'NETCDF4')
        lat = ds.createDimension('lat', len(self.latrange))
        lon = ds.createDimension('lon', len(self.lonrange)) 
        time = ds.createDimension('time', None)
        
        #make nc file metadata
        ds.title = 'SWE interpolation for ' + self.datecol

        lat = ds.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'

        lon = ds.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'

        time = ds.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'

        SWE = ds.createVariable('SWE', np.float64, ('time', 'lat', 'lon',))
        SWE.units = 'inches'
        SWE.standard_name = 'snow_water_equivalent'
        SWE.long_name = 'Interpolated SWE product @1-km'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        SWE.coordinates = 'lon lat'
        

        # Write latitudes, longitudes.
        lat[:] = self.latrange
        lon[:] = self.lonrange

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        SWE[:,:,:] = self.SWE_array 
        
        #Set date/time information
        times_arr = time[:]
        dates = [dt.datetime(int(self.datecol[0:4]),int(self.datecol[5:7]),int(self.datecol[8:]),0)]
        times = date2num(dates, time.units)
        time[:] = times
        
        print(ds)
        ds.close()
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()
            
            
            
            
    def netCDF_CONUS(self, plot):

        #get all SWE regions data into one DF

        self.NA_SWE = pd.DataFrame()
        columns = ['Long', 'Lat', 'elevation_m', 'northness', self.datecol]

        for region in self.Forecast:
            self.NA_SWE = self.NA_SWE.append(self.Forecast[region][columns])

        self.NA_SWE = self.NA_SWE.rename(columns = {self.datecol:'SWE'})
        

        #round to 2 decimals
        self.NA_SWE['Lat'] = round(self.NA_SWE['Lat'],2)
        self.NA_SWE['Long'] = round(self.NA_SWE['Long'],2)

        #NA_SWE = NA_SWE.set_index('Date')

        #Get the range of lat/long to put into xarray
        self.lonrange = np.arange(-124.75, -66.95, 0.01)
        self.latrange = np.arange(25.52, 49.39, 0.01)

        self.lonrange = [round(num, 2) for num in self.lonrange]
        self.latrange = [round(num, 2) for num in self.latrange]

         #Make grid of lat long
        FG = self.expand_grid(self.latrange, self.lonrange)

        #Merge SWE predictions with gridded df
        self.DFG = pd.merge(FG, self.NA_SWE, on = ['Long','Lat'], how = 'left')

        #drop duplicate lat/long
        self.DFG = self.DFG.drop_duplicates(subset = ['Long', 'Lat'], keep = 'last').reset_index(drop = True)
        
        #fill NaN values with 0
        self.DFG['SWE'] = self.DFG['SWE'].fillna(0)

        #Reshape DFG DF
        self.SWE_array = self.DFG['SWE'].values.reshape(1,len(self.latrange),len(self.lonrange))

       # create nc filepath
        fn = self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+self.datecol+'_CONUS.nc'
        
        # make nc file, set lat/long, time
        ds = nc.Dataset(fn, 'w', format = 'NETCDF4')
        lat = ds.createDimension('lat', len(self.latrange))
        lon = ds.createDimension('lon', len(self.lonrange)) 
        time = ds.createDimension('time', None)
        
        #make nc file metadata
        ds.title = 'SWE interpolation for ' + self.datecol

        lat = ds.createVariable('lat', np.float32, ('lat',))
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'

        lon = ds.createVariable('lon', np.float32, ('lon',))
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'

        time = ds.createVariable('time', np.float64, ('time',))
        time.units = 'hours since 1800-01-01'
        time.long_name = 'time'

        SWE = ds.createVariable('SWE', np.float64, ('time', 'lat', 'lon',))
        SWE.units = 'inches'
        SWE.standard_name = 'snow_water_equivalent'
        SWE.long_name = 'Interpolated SWE product @1-km'

        #add projection information
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326) # GCS_WGS_1984
        SWE.esri_pe_string = proj.ExportToWkt()

        #set lat lon info in file
        SWE.coordinates = 'lon lat'
        

        # Write latitudes, longitudes.
        lat[:] = self.latrange
        lon[:] = self.lonrange

        # Write the data.  This writes the whole 3D netCDF variable all at once.
        SWE[:,:,:] = self.SWE_array 
        
        #Set date/time information
        times_arr = time[:]
        dates = [dt.datetime(int(self.datecol[0:4]),int(self.datecol[5:7]),int(self.datecol[8:]),0)]
        times = date2num(dates, time.units)
        time[:] = times
        
        print(ds)
        ds.close()
        print('File conversion to netcdf complete')
        
        if plot == True:
            print('Plotting results')
            self.plot_netCDF()

        
           
        
    def plot_netCDF(self):
        
        #set up colormap that is transparent for zero values
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)

        
        
        
        #load file
        fn = self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+ self.datecol+'.nc'
        SWE = nc.Dataset(fn)

        #Get area of interest
        lats = SWE.variables['lat'][:]
        lons = SWE.variables['lon'][:]
        swe = SWE.variables['SWE'][:]

        #get basemap
        plt.figure(figsize=(20,10))
        map = Basemap(projection='merc',llcrnrlon=-130.,llcrnrlat=30,urcrnrlon=-100,urcrnrlat=50.,resolution='i')

        map.drawcoastlines()
        map.drawstates()
        map.drawcountries()
        map.drawlsmask(land_color='Linen', ocean_color='#CCFFFF') # can use HTML names or codes for colors
        map.drawcounties()

        #put lat / long into appropriate projection grid
        lons, lats = np.meshgrid(lons, lats)
        x,y = map(lons, lats)
        map.pcolor(x, y,swe, cmap= map_object)
        plt.colorbar()
        
        
     #produce an interactive plot using Folium
    def plot_interactive(self, pinlat, pinlong, web):
        
        #set up colormap that is transparent for zero values
        # get colormap
        ncolors = 256
        color_array = plt.get_cmap('viridis')(range(ncolors))

        # change alpha values
        color_array[:,-1] = np.linspace(0.0,1.0,ncolors)

        # create a colormap object
        map_object = LinearSegmentedColormap.from_list(name='viridis_alpha',colors=color_array)

        # register this new colormap with matplotlib
        plt.register_cmap(cmap=map_object)
        
        
        #load file
        fn = self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+ self.datecol+'_CONUS.nc'
        
        #open netcdf file with rioxarray
        xr = rxr.open_rasterio(fn)

        xr.rio.write_crs("epsg:4326", inplace=True)

        #replace x and y numbers with coordinate rangres
        xr.coords['x'] = self.lonrange
        xr.coords['y'] = self.latrange

        # Create a variable for destination coordinate system 
        dst_crs = 'EPSG:4326' 

        #scale the array from 0 - 255
        scaled_xr = es.bytescale(xr.values[0])
        
        #set max for color map
        maxSWE = xr.values[0].max()
        #set color range for color map
        SWErange = np.arange(0,maxSWE+1, maxSWE/5).tolist()

        m = folium.Map(location=[pinlat, pinlong],
               tiles = 'Stamen Terrain', zoom_start = 9, control_scale=True)

        #map bounds, must be minutally adjusted for correct lat/long placement
        map_bounds = [[min(self.latrange)-0.86, min(self.lonrange)], 
                      [max(self.latrange)-0.86, max(self.lonrange), 0.01]]

        rasterlayer = folium.FeatureGroup(name = 'SWE')

        rasterlayer.add_child(folium.raster_layers.ImageOverlay(
                                image=scaled_xr,
                                bounds=map_bounds,
                                interactive=True,
                                cross_origin=False,
                                zindex=1,
                                colormap=map_object,
                                opacity=1
                                    ))
        
        #add colorbar
        colormap = cm.LinearColormap(colors=['violet', 'darkblue', 'blue', 'cyan', 'green', 'yellow'],
                                     index=SWErange, vmin=0.1, vmax=xr.values[0].max(),
                                     caption='Snow Water Equivalent (SWE) in inches')

        m.add_child(rasterlayer)
        m.add_child(folium.LayerControl())
        m.add_child(colormap)
        
        #code for webbrowser app
        if web == True:
            output_file =  self.cwd +'/Data/NetCDF/SWE_'+self.datecol+'.html'
            m.save(output_file)
            webbrowser.open(output_file, new=2)
            
        else:
            display(m)
            
        xr.close()
      

     #produce an interactive plot using Folium
    def plot_interactive_SWE(self, pinlat, pinlong, web):
        fnConus = self.cwd +'/Data/NetCDF/SWE_MAP_1km_'+self.datecol+'_CONUS.nc'

        #xr = rxr.open_rasterio(fn)
        xrConus = rxr.open_rasterio(fnConus)
        
        #Convert rxr df to geodataframe
        x, y, elevation = xrConus.x.values, xrConus.y.values, xrConus.values
        x, y = np.meshgrid(x, y)
        x, y, elevation = x.flatten(), y.flatten(), elevation.flatten()

        print("Converting to GeoDataFrame...")
        SWE_pd = pd.DataFrame.from_dict({'SWE': elevation, 'x': x, 'y': y})
        SWE_threshold = 0.1
        SWE_pd = SWE_pd[SWE_pd['SWE'] > SWE_threshold]
        SWE_gdf = gpd.GeoDataFrame(
            SWE_pd, geometry=gpd.points_from_xy(SWE_pd.x, SWE_pd.y), crs=4326)

        SWE_gdf.geometry = SWE_gdf.geometry.buffer(0.01, cap_style=3)
        SWE_gdf.geometry = SWE_gdf.geometry.to_crs(epsg= 4326)

        SWE_gdf =  SWE_gdf.reset_index(drop = True)
        SWE_gdf['geoid'] = SWE_gdf.index.astype(str)
        Chorocols = ['geoid', 'SWE', 'geometry']
        SWE_gdf = SWE_gdf[Chorocols]
        SWE_gdf.crs = CRS.from_epsg(4326)

        print('File conversion complete, creating mapping instance')
        # Create a Map instance
        m = folium.Map(location=[pinlat, pinlong], tiles = 'Stamen Terrain', zoom_start=10, 
                       control_scale=True)

        # Plot a choropleth map
        # Notice: 'geoid' column that we created earlier needs to be assigned always as the first column
        folium.Choropleth(
            geo_data=SWE_gdf,
            name='SWE estimates',
            data=SWE_gdf,
            columns=['geoid', 'SWE'],
            key_on='feature.id',
            fill_color='YlGnBu_r',
            fill_opacity=0.7,
            line_opacity=0.2,
            line_color='white', 
            line_weight=0,
            highlight=False, 
            smooth_factor=1.0,
            #threshold_scale=[100, 250, 500, 1000, 2000],
            legend_name= 'SWE in inches for '+ self.datecol).add_to(m)

        # Convert points to GeoJson
        folium.features.GeoJson(SWE_gdf,  
                                name='Snow Water Equivalent',
                                style_function=lambda x: {'color':'transparent','fillColor':'transparent','weight':0},
                                tooltip=folium.features.GeoJsonTooltip(fields=['SWE'],
                                                                        aliases = ['Snow Water Equivalent (in) for '+ self.datecol+ ':'],
                                                                        labels=True,
                                                                        sticky=True,
                                                                         localize=True
                                                                                    )
                               ).add_to(m)

        
         #code for webbrowser app
        if web == True:
            output_file =  self.cwd +'/Data/NetCDF/SWE_'+self.datecol+'_Interactive.html'
            m.save(output_file)
            webbrowser.open(output_file, new=2)

        else:
            display(m)
            
        xrConus.close()
        

