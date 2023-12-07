#Standardized Snow Water Equivalent Evaluation Tool (SWEET)
#created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
# SWEET supported by the University of Alabama and the Alabama Water Institute
# 10-19-2023

#Load packages
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import h5py
import tables
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.metrics import mean_squared_error
import hydroeval as he
from pickle import dump
import pickle 
from tqdm import tqdm
import geopandas as gpd
import folium
from folium import features
from folium.plugins import StripePattern
import branca.colormap as cm
import vincent
from vincent import AxisProperties, PropertySet, ValueRef, Axis
import hvplot.pandas
import holoviews as hv
from holoviews import dim, opts, streams
from bokeh.models import HoverTool
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os
import json
import warnings; warnings.filterwarnings("ignore")

#A function to load model predictions
def load_Predictions(Region_list):
    #Regions = ['N_Sierras','S_Sierras_Low', 'S_Sierras_High']
    RegionTest = {}
    for Region in Region_list:
        RegionTest[Region] = pd.read_hdf("./Predictions/Testing/Predictions.h5", key = Region)
        
        #convert predictions/observations to SI units, aka, cm
        RegionTest[Region]['y_test'] = RegionTest[Region]['y_test']*2.54
        RegionTest[Region]['y_pred'] = RegionTest[Region]['y_pred']*2.54
        RegionTest[Region]['y_pred_fSCA'] = RegionTest[Region]['y_pred_fSCA']*2.54
        
        #get SWE obs columns to convert to si
        obscols = [match for match in RegionTest[Region] if 'SWE_' in match]
        for col in obscols:
            RegionTest[Region][col] = RegionTest[Region][col]*2.54
        
        
    return RegionTest


#Function to convert predictions into parity plot plus evaluation metrics
def parityplot(RegionTest):
    
    #Make sure dates are in datetime formate
    for key in RegionTest.keys():
        RegionTest[key]['Date'] = pd.to_datetime(RegionTest[key]['Date'])
    
    #Get regions
    Regions = list(RegionTest.keys())
    
    #put y_pred, y_pred_fSCA, y_test, Region into one DF for parity plot
    Compare_DF = pd.DataFrame()
    cols = ['Region', 'y_test', 'y_pred', 'y_pred_fSCA']
    for Region in Regions:
        df = RegionTest[Region][cols]
        Compare_DF = pd.concat([Compare_DF, df])
    
    
    
    
    #Plot the results in a parity plot
    sns.set(style='ticks')
    SWEmax = max(Compare_DF['y_test'])

    sns.relplot(data=Compare_DF, x='y_test', y='y_pred_fSCA', hue='Region', hue_order=Regions, aspect=1.61)
    plt.plot([0,SWEmax], [0,SWEmax], color = 'red', linestyle = '--')
    plt.xlabel('Observed SWE (cm)')
    plt.ylabel('Predicted SWE (cm)')
    plt.show()

    #Run model evaluate functions
    #Regional
    Performance = pd.DataFrame()
    for Region in Regions:
        y_test = RegionTest[Region]['y_test']
        y_pred = RegionTest[Region]['y_pred']
        y_pred_fSCA = RegionTest[Region]['y_pred_fSCA']
        
        r2 = sklearn.metrics.r2_score(y_test, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
        kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
        pbias = he.evaluator(he.pbias, y_pred, y_test)
    
        r2_fSCA = sklearn.metrics.r2_score(y_test, y_pred_fSCA)
        rmse_fSCA = sklearn.metrics.mean_squared_error(y_test, y_pred_fSCA, squared = False)
        kge_fSCA, r_fSCA, alpha_fSCA, beta_fSCA = he.evaluator(he.kge, y_pred_fSCA, y_test)
        pbias_fSCA = he.evaluator(he.pbias, y_pred_fSCA, y_test)
        
        error_data = np.array([Region, 
                               round(r2,2),  
                               round(rmse,2), 
                               round(kge[0],2),
                               round(pbias[0],2),
                               round(r2_fSCA,2),
                               round(rmse_fSCA,2),
                              round(kge_fSCA[0],2),
                              round(pbias_fSCA[0],2)])
        
        error = pd.DataFrame(data = error_data.reshape(-1, len(error_data)), 
                             columns = ['Region', 
                                        'R2',
                                        'RMSE',
                                        'KGE', 
                                        'PBias', 
                                        'R2_fSCA',
                                        'RMSE_fSCA',
                                        'KGE_fSCA', 
                                        'PBias_fSCA',
                                       ])
        Performance = Performance.append(error, ignore_index = True)
      

    #All Sierras
    #y_test = Compare_DF['y_test']
    #y_pred = Compare_DF['y_pred']
    #y_pred_fSCA = Compare_DF['y_pred_fSCA']
    #r2 = sklearn.metrics.r2_score(y_test, y_pred)
    #rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False) 
    #kge, r, alpha, beta = he.evaluator(he.kge, y_pred, y_test)
    #pbias = he.evaluator(he.pbias, y_pred, y_test)
    
    #r2_fSCA = sklearn.metrics.r2_score(y_test, y_pred_fSCA)
    #rmse_fSCA = sklearn.metrics.mean_squared_error(y_test, y_pred_fSCA, squared = False)
    #kge_fSCA, r_fSCA, alpha_fSCA, beta_fSCA = he.evaluator(he.kge, y_pred_fSCA, y_test)
    #pbias_fSCA = he.evaluator(he.pbias, y_pred_fSCA, y_test)
    
    
    #error_data = np.array(['Sierras_All',
 #                           round(r2,2),  
   #                            round(rmse,2), 
    #                           round(kge[0],2),
     #                          round(pbias[0],2),
      #                         round(r2_fSCA,2),
       #                        round(rmse_fSCA,2),
        #                      round(kge_fSCA[0],2),
         #                     round(pbias_fSCA[0],2)])

    #error = pd.DataFrame(data = error_data.reshape(-1, len(error_data)), 
     #                        columns = ['Region', 
      #                                  'R2',
       #                                 'RMSE',
        #                                'KGE', 
         #                               'PBias', 
          #                              'R2_fSCA',
           #                             'RMSE_fSCA',
            #                            'KGE_fSCA', 
             #                           'PBias_fSCA'
              #                         ])
    #Performance = Performance.append(error, ignore_index = True)
#    display(Performance)
    return Performance
    
    
    
#Plot the error/prediction compared to different variables
def Model_Vs(RegionTest,metric,model_output):
    #load access key
    home = os.path.expanduser('~')
    keypath = "apps/AWSaccessKeys.csv"
    access = pd.read_csv(f"{home}/{keypath}")

    #start session
    session = boto3.Session(
        aws_access_key_id=access['Access key ID'][0],
        aws_secret_access_key=access['Secret access key'][0],
    )
    s3 = session.resource('s3')
    #AWS bucket information
    bucket_name = 'national-snow-model'
    #s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)

    
    #Get regions
    Regions = list(RegionTest.keys())
     
    #put y_pred, y_pred_fSCA, y_test, Region into one DF for parity plot
    Compare_DF = pd.DataFrame()
    cols = ['Region', 'y_test', 'y_pred', 'y_pred_fSCA', metric]
    
    #RegionTrain_SCA_path = f"{datapath}/data/RegionTrain_SCA.pkl"   
    file_key = 'data/RegionTrain_SCA.pkl'
    obj = bucket.Object(file_key)
    body = obj.get()['Body']
    df_Region = pd.read_pickle(body)
    
    # load regionalized forecast data
    #current forecast
    #df_Region = open(RegionTrain_SCA_path, "rb")
    #df_Region = pickle.load(df_Region)
    
    
    for Region in Regions:
        #check if metric is included and add if not
        try:
            df = RegionTest[Region][cols]
        except:
            df_Region[Region] = pd.DataFrame(df_Region[Region][metric])
            df_Region[Region].drop_duplicates(inplace = True)
            df_Region[Region].reset_index(inplace = True)
            df.reset_index(inplace = True)
            df = pd.merge(RegionTest[Region], df_Region[Region],on = 'index', how = 'left')
            df.set_index('index', inplace = True, drop = True)
            df = df[cols]

        Compare_DF = pd.concat([Compare_DF, df])
        
    #Calculate error
    Compare_DF['error'] = Compare_DF['y_test'] - Compare_DF['y_pred']
    Compare_DF['error_fSCA'] = Compare_DF['y_test'] - Compare_DF['y_pred_fSCA']
    Compare_DF['Perc_error_fSCA'] = ((Compare_DF['y_test'] - Compare_DF['y_pred_fSCA'])/Compare_DF['y_test'])*100
    Compare_DF['Perc_error_fSCA'] = Compare_DF['Perc_error_fSCA'].fillna(0)
    #change error > 100 to 100
    Compare_DF.loc[Compare_DF['Perc_error_fSCA'] >100, 'Perc_error_fSCA'] = 100
    Compare_DF.loc[Compare_DF['Perc_error_fSCA'] < -100, 'Perc_error_fSCA'] = -100
    
    if model_output == 'Prediction':
        Y = 'y_pred_fSCA'
        ylabel ='SWE (cm)'
    
    if model_output == 'Error':
        Y = 'error_fSCA'
        ylabel ='SWE (cm)'
        
    if model_output == 'Percent_Error':
        Y = 'Perc_error_fSCA'
        ylabel = 'SWE % Error'
        
    if metric == 'northness':
        xlabel = 'Northness'
        
    if metric == 'elevation_m':
        xlabel = 'Elevation (m)'
    if metric == 'WYWeek':
        xlabel = 'Water Year Week (From Oct 1st)'
    if metric == 'prev_SWE':
        xlabel = 'Previous SWE Estimate'
    if metric == 'Lat':
        xlabel = 'Latitude'
    if metric == 'prev_SWE_error':
        xlabel = 'Error in Previous SWE Estimate'
    
    sns.set(style='ticks')
    sns.relplot(data=Compare_DF, x=metric, y=Y, hue='Region', hue_order=Regions, aspect=1.61)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(f"{model_output} by {xlabel}", fontsize = 20)
    plt.show()
    
    
    
#create geopandas dataframes to map predictions and obs
def createGeoSpatial(Sites, Region_list):
     #load access key
    home = os.path.expanduser('~')
    keypath = "apps/AWSaccessKeys.csv"
    access = pd.read_csv(f"{home}/{keypath}")

    #start session
    session = boto3.Session(
        aws_access_key_id=access['Access key ID'][0],
        aws_secret_access_key=access['Secret access key'][0],
    )
    s3 = session.resource('s3')
    #AWS bucket information
    bucket_name = 'national-snow-model'
    #s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)  
    
    #Load RegionTrain and Snotel Geospatial DFs
    #GeoSpatial = open(f"{datapath}/data/RegionTrain.pkl", "rb")
    #GeoSpatial = pickle.load(GeoSpatial)
    
    file_key = 'data/RegionTrain.pkl'
    obj = bucket.Object(file_key)
    body = obj.get()['Body']
    GeoSpatial = pd.read_pickle(body)
    
    #Snotel = open(f"{datapath}/data/RegionSnotel_Train.pkl", "rb")
    #Snotel = pickle.load(Snotel)
    file_key = 'data/RegionSnotel_Train.pkl'
    obj = bucket.Object(file_key)
    body = obj.get()['Body']
    Snotel = pd.read_pickle(body)
    
    #separate Sierras high and low
    GeoSpatial['S_Sierras_High'] = GeoSpatial['S_Sierras'][GeoSpatial['S_Sierras']['elevation_m'] > 2500]
    GeoSpatial['S_Sierras_Low'] = GeoSpatial['S_Sierras'][GeoSpatial['S_Sierras']['elevation_m'] <= 2500]

    Snotel['S_Sierras_High'] = Snotel['S_Sierras'][Snotel['S_Sierras']['elevation_m'] > 2500]
    Snotel['S_Sierras_Low'] = Snotel['S_Sierras'][Snotel['S_Sierras']['elevation_m'] <= 2500]
    
    #Regions = ['N_Sierras','S_Sierras_Low', 'S_Sierras_High']
    
    GeoPred = pd.DataFrame()
    GeoObs = pd.DataFrame()
    
    for Region in Region_list:
        #Create Geospatial prediction point DF
        Pred_Geo = GeoSpatial[Region].copy()
        Snotel_Geo = Snotel[Region].copy()
        
        Pred_Geo = Pred_Geo.reset_index().drop_duplicates(subset='cell_id', keep='last').set_index('cell_id').sort_index()
        Snotel_Geo = Snotel_Geo.reset_index().drop_duplicates(subset='station_id', keep='last').set_index('station_id').sort_index()
        
        #Convert the Prediction Geospatial dataframe into a geopandas dataframe
        Pred_Geo = gpd.GeoDataFrame(Pred_Geo, geometry = gpd.points_from_xy(Pred_Geo.Long, Pred_Geo.Lat))
        Snotel_Geo = gpd.GeoDataFrame(Snotel_Geo, geometry = gpd.points_from_xy(Snotel_Geo.Long, Snotel_Geo.Lat))

        Pcols = ['Long','Lat','elevation_m','slope_deg','aspect', 'geometry']
        Obscols = ['Long','Lat','elevation_m','slope_deg','aspect', 'geometry']
        Pred_Geo= Pred_Geo[Pcols].reset_index()
        Snotel_Geo = Snotel_Geo[Obscols].reset_index()
        
        #add to respective dataframe
        GeoPred = pd.concat([GeoPred, Pred_Geo])
        GeoObs = pd.concat([GeoObs, Snotel_Geo])
        
    #Select sites used for prediction
    GeoObs = GeoObs.set_index('station_id').T[Sites].T.reset_index()
    
    return GeoPred, GeoObs



#Get SNOTEL sites used as features
#load RFE optimized features
def InSitu_Locations(RegionTest):
    
     #load access key
    home = os.path.expanduser('~')
    keypath = "apps/AWSaccessKeys.csv"
    access = pd.read_csv(f"{home}/{keypath}")

    #start session
    session = boto3.Session(
        aws_access_key_id=access['Access key ID'][0],
        aws_secret_access_key=access['Secret access key'][0],
    )
    s3 = session.resource('s3')
    #AWS bucket information
    bucket_name = 'national-snow-model'
    #s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))
    bucket = s3.Bucket(bucket_name)  

    #Region_optfeatures= pickle.load(open(f"{datapath}/data/Optimal_Features.pkl", "rb"))
    file_key = 'data/Optimal_Features.pkl'
    obj = bucket.Object(file_key)
    body = obj.get()['Body']
    Region_optfeatures = pd.read_pickle(body)
    
    
    Sites = {}
    Regions = list(RegionTest.keys())
    Sites_list = []
    for Region in Regions:
        Sites[Region] = [match for match in Region_optfeatures[Region] if 'SWE_'  in match]

        for i in np.arange(0, len(Sites[Region]),1):
            Sites[Region][i] = Sites[Region][i].replace('Delta_SWE_', '') 
            Sites[Region][i] = Sites[Region][i].replace('SWE_', '') 
            Sites[Region][i] = Sites[Region][i].replace('Prev_SWE_', '') 
            Sites[Region][i] = Sites[Region][i].replace('Prev_', '') 
            res = []
            [res.append(x) for x in Sites[Region] if x not in res]
        #Sites[Region] = res
        
        #make one DF for Sites
        Sites_list = Sites_list+res
        Sites_list.sort()
        
    return Sites_list



#need to put the predictions, obs, error in a time series format
def ts_pred_obs_err(Compare_DF):
    print('Processing Dataframe into timeseries format: predictions, observations, error.')
    x = Compare_DF.copy()
    x.index = x.index.set_names(['cell_id'])

    #predictions
    x_pred = x.copy()
    cols = ['Date','y_pred']
    x_pred = x_pred[cols].reset_index().set_index('Date').sort_index()
    x_pred = df_transpose(x_pred, 'y_pred')

    #observations
    x_obs = x.copy()
    cols = ['Date','y_test']
    x_obs = x_obs[cols].reset_index().set_index('Date').sort_index()
    x_obs = df_transpose(x_obs, 'y_test')

    #error
    x_err = x.copy()
    cols = ['Date','error']
    x_err = x_err[cols].reset_index().set_index('Date').sort_index()
    x_err = df_transpose(x_err, 'error')
    
    return x_pred, x_obs, x_err


def map_data_prep(RegionTest):
    #Get regions
    Regions = list(RegionTest.keys())
    
    #put y_pred, y_pred_fSCA, y_test, Region into one DF for parity plot
    Compare_DF = pd.DataFrame()
    cols = ['Date', 'Lat', 'Long', 'elevation_m', 'y_test', 'y_pred', 'y_pred_fSCA', 'Region']
    for Region in Regions:
        df = RegionTest[Region][cols]
        Compare_DF = pd.concat([Compare_DF, df])
        
    #Calculate error
    Compare_DF['error'] = Compare_DF['y_test'] - Compare_DF['y_pred']
    Compare_DF['error_fSCA'] = Compare_DF['y_test'] - Compare_DF['y_pred_fSCA']
    
    return Compare_DF

def df_transpose(df, obs):
    #get index
    date_idx = df.index.unique()
    #get columns names, aka sites
    sites = df['cell_id'].values
    #make dataframe
    DF =pd.DataFrame(index = date_idx)
    for site in tqdm(sites):
        s = pd.DataFrame(df[df['cell_id'] == site][obs])
        DF = DF.join(s)
        DF = DF.rename(columns ={obs: site})
    DF = DF.loc[:,~DF.columns.duplicated()].copy()
    return DF


#Map locations and scoring of sites
#def Map_Plot_Eval(self, freq, df, size):
def Map_Plot_Eval(RegionTest, yaxis, error_metric, Region_list):   
    
    #Make sure dates are in datetime formate
    for key in RegionTest.keys():
        RegionTest[key]['Date'] = pd.to_datetime(RegionTest[key]['Date'])
    
    #correctly configure dataframes for plotting
    pred, obs, err = ts_pred_obs_err(map_data_prep(RegionTest))
    
    #Get SNOTEL sites used as features
    #load RFE optimized features
    Sites = InSitu_Locations(RegionTest)

    #Get the geometric DF for prediction locations and in situ obs
    GeoDF, Snotel = createGeoSpatial(Sites, Region_list)

    print('Plotting monitoring station locations')
    cols =  ['cell_id', 'Lat', 'Long', 'geometry']

    df_map = GeoDF[cols].copy()

    #Get Centroid of watershed
    centeroid = df_map.dissolve().centroid

    # Create a Map instance
   # m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], tiles = 'Stamen Terrain', zoom_start=8, 
    #               control_scale=True)
    m = folium.Map(location=[centeroid.y[0], centeroid.x[0]], zoom_start=8, control_scale=True)
    #add legend to map
    if error_metric == 'KGE':
        colormap = cm.StepColormap(colors = ['darkred', 'r', 'orange', 'g'], vmin = 0, vmax = 1, index = [0,0.4,0.6,0.85,1])
        colormap.caption = 'Model Performance (KGE)'
        
    elif error_metric == 'cm':
        colormap = cm.StepColormap(colors = ['g', 'orange', 'r', 'darkred'], vmin = 0, vmax = 20, index = [0,6,12,25,50])
        colormap.caption = 'Model Error (cm)'
        
    elif error_metric == '%':
        colormap = cm.StepColormap(colors = ['g', 'orange', 'r', 'darkred'], vmin = 0, vmax = 50, index = [0,10,20,30,50])
        colormap.caption = 'Model Error (%)'
        
    
    m.add_child(colormap)

    ax = AxisProperties(
    labels=PropertySet(
        angle=ValueRef(value=300),
        align=ValueRef(value='right')
            )
        )

    for i in obs.columns:


        #get site information
        site = i
        Obs_site = 'Observations'#_' + site
        Pred_site = 'Predictions'#_' + site
        Err_site = 'Errors'#_' + site


        #get modeled, observed, and error information for each site
        df = pd.DataFrame(obs[site])
        df = df.rename(columns = {site: Obs_site})
        df[Pred_site] = pd.DataFrame(pred[site])
        df[Err_site] = pd.DataFrame(err[site])
        
        #drop na values
        df.dropna(inplace = True)

        if error_metric == 'KGE':
            #set the color of marker by model performance
            kge, r, alpha, beta = he.evaluator(he.kge, df[Pred_site].astype('float32'), df[Obs_site].astype('float32'))

            if kge[0] > 0.85:
                color = 'green'

            elif kge[0] > 0.6:
                color = 'orange'

            elif kge[0] > 0.40:
                color = 'red'

            else:
                color = 'darkred'
                
        #error in absolute value and inches       
        elif error_metric == 'cm':
            error = np.abs(np.mean(df[Obs_site] - df[Pred_site]))
            if error < 6:
                color = 'green'

            elif error < 12:
                color = 'orange'

            elif error <25:
                color = 'red'

            else:
                color = 'darkred'
        
        #mean percentage error
        elif error_metric == '%':
            #make all predictions and observations below 1", 1" to remove prediction biases, it does not matter if there
            #is 0.5" or 0.9" of SWE but the percentage error here will be huge and overpowering
            df[df[Obs_site]<1] = 1
            df[df[Pred_site]<1] = 1
            
            error = np.mean(np.abs(df[Obs_site] - df[Pred_site])/df[Obs_site])*100
            if error < 10:
                color = 'green'

            elif error < 20:
                color = 'orange'

            elif error <30:
                color = 'red'

            else:
                color = 'darkred'
                

        title_size = 14
        
        #display(df)

        #create graph and convert to json
        graph = vincent.Scatter(df, height=300, width=500)
        graph.axis_titles(x='Datetime', y=yaxis)
        graph.legend(title= 'Legend')
        graph.colors(brew='Paired')
        graph.x_axis_properties(title_size=title_size, title_offset=35,
                      label_angle=300, label_align='right', color=None)
        graph.y_axis_properties(title_size=title_size, title_offset=-30,
                      label_angle=None, label_align='right', color=None)

        data = json.loads(graph.to_json())

        #Add marker with point to map, https://fontawesome.com/v4/cheatsheet  - needs to be v4.6 or less
        lat_long = df_map[df_map['cell_id'] == i]
        lat = lat_long['Lat'].values[0]
        long = lat_long['Long'].values[0]

        mk = features.Marker([lat, long], icon=folium.Icon(color=color, icon = ' fa-ge', prefix = 'fa'))
        p = folium.Popup()
        v = features.Vega(data, width="100%", height="100%")

        mk.add_child(p)
        p.add_child(v)
        m.add_child(mk)
        
        
    # add SNOTEL marker one by one on the map
    for i in range(0,len(Snotel)):
        

        folium.Marker(
          location=[Snotel.iloc[i]['Lat'], Snotel.iloc[i]['Long']],
            icon=folium.Icon(color='blue', icon = 'fa-area-chart', prefix = 'fa'),
            tooltip =  str(Snotel.iloc[i]['station_id']),
          popup= str(Snotel.iloc[i]['elevation_m']) + "m",
       ).add_to(m)

    display(m)
    


     