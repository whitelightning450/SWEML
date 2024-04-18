# Import packages
# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely
from shapely import wkt
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer

# Raster Packages
import rioxarray as rxr
import rasterio
from rasterio.mask import mask
from rioxarray.merge import merge_arrays
import rasterstats as rs
from osgeo import gdal
from osgeo import gdalconst

# Data Access Packages
import earthaccess as ea
import h5py
import pickle
from pystac_client import Client
import richdem as rd
import planetary_computer
from planetary_computer import sign

# General Packages
import os
import re
import shutil
import math
from datetime import datetime
import glob
from pprint import pprint
from typing import Union
from pathlib import Path
from tqdm import tqdm
import time
import requests
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import dask
import dask.dataframe as dd
from dask.distributed import progress
from dask.distributed import Client
from dask.diagnostics import ProgressBar
from retrying import retry
import fiona
import re
import s3fs

#need to mamba install gdal, earthaccess 
#pip install pystac_client, richdem, planetary_computer, dask, distributed, retrying

#connecting to AWS
import warnings; warnings.filterwarnings("ignore")
import boto3
from botocore import UNSIGNED
from botocore.client import Config

import NSIDC_Data
'''
To create .netrc file:
import earthaccess
earthaccess.login(persist=True)
open file and change machine to https://urs.earthdata.nasa.gov

'''

#load access key
HOME = os.path.expanduser('~')
KEYPATH = "SWEML/AWSaccessKeys.csv"
ACCESS = pd.read_csv(f"{HOME}/{KEYPATH}")

#start session
SESSION = boto3.Session(
    aws_access_key_id=ACCESS['Access key ID'][0],
    aws_secret_access_key=ACCESS['Secret access key'][0],
)
S3 = SESSION.resource('s3')
#AWS BUCKET information
BUCKET_NAME = 'national-snow-model'
#S3 = boto3.resource('S3', config=Config(signature_version=UNSIGNED))
BUCKET = S3.Bucket(BUCKET_NAME)

def load_aso_snotel_geometry(aso_swe_file, folder_path):
    print('Loading ASO/SNOTEL Geometry')
    
    aso_file = pd.read_csv(os.path.join(folder_path, aso_swe_file))
    aso_file.set_index('cell_id', inplace=True)
    aso_geometry = [Point(xy) for xy in zip(aso_file['x'], aso_file['y'])]
    aso_gdf = gpd.GeoDataFrame(aso_file, geometry=aso_geometry)
    
    return aso_gdf

# Calculating nearest SNOTEL sites
def calculate_nearest_snotel(aso_gdf, snotel_gdf, n=6, distance_cache=None):

    if distance_cache is None:
        distance_cache = {}

    nearest_snotel = {}
    for idx, aso_row in tqdm(aso_gdf.iterrows()):
        cell_id = idx
        # Check if distances for this cell_id are already calculated and cached
        if cell_id in distance_cache:
            nearest_snotel[idx] = distance_cache[cell_id]
        else:
            # Calculate Haversine distances between the grid cell and all SNOTEL locations
            distances = haversine_vectorized(
                aso_row.geometry.y, aso_row.geometry.x,
                snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)

            # Store the nearest stations in the cache
            nearest_snotel[idx] = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
            distance_cache[cell_id] = nearest_snotel[idx]
    #save nearest snotel information as a dictionary.pkl file for future use.
    with open(f"{HOME}/SWEML/data/NWMv2.0/data/TrainingDFs/nearest_SNOTEL.pkl", 'wb') as handle:
        pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return nearest_snotel

def haversine_vectorized(lat1, lon1, lat2, lon2):
    
    lon1 = np.radians(lon1)
    lon2 = np.radians(lon2)
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    r = 6371.0
    # Distance calculation
    distances = r * c

    return distances

def calculate_distances_for_cell(aso_row, snotel_gdf, n=6):
   
    distances = haversine_vectorized(
        aso_row.geometry.y, aso_row.geometry.x,
        snotel_gdf.geometry.y.values, snotel_gdf.geometry.x.values)
    
    nearest_sites = list(snotel_gdf['station_id'].iloc[distances.argsort()[:n]])
    
    return nearest_sites

def create_polygon(row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])

def fetch_snotel_sites_for_cellids(region):
    aso_swe_files_folder_path = f"{HOME}/SWEML/data/NSMv2.0/data/Processed_SWE/{region}"
    metadata_path = f"{HOME}/SWEML/data/NSMv2.0/data/TrainingDFs/grid_cells_meta.csv"
    snotel_path = f"{HOME}/SWEML/data/NSMv2.0/data/SNOTEL_Data/"
    
    #Get metadata for sites/snotel and observatins from 2013-2019
    #Get site metadata
    try:
        metadata_df = pd.read_csv(metadata_path)
        #metadata_df['geometry'] = metadata_df['geometry'].apply(wkt.loads)
    except:
        print("metadata not found, retreiving from AWS S3")
        key = "NSMv2.0"+metadata_path.split("NSMv2.0",1)[1]        
        S3.meta.client.download_file(BUCKET_NAME, key,metadata_path)
        metadata_df = pd.read_csv(metadata_path)

     #get Snotel site meta
    Snotelmeta_path = f"{snotel_path}ground_measures_metadata.csv"
    
    try:
        snotel_file = pd.read_csv(Snotelmeta_path)
    except:
        print("Snotel meta not found, retreiving from AWS S3")
        key = "NSMv2.0"+Snotelmeta_path.split("NSMv2.0",1)[1]       
        S3.meta.client.download_file(BUCKET_NAME, key,Snotelmeta_path)
        snotel_file = pd.read_csv(Snotelmeta_path)

    #get Snotel observations
    Snotelobs_path = f"{snotel_path}ground_measures_train_featuresALLDATES.parquet"
    try:
        snotel_data = pd.read_csv(Snotelobs_path)
    except:
        print("Snotel obs not found, retreiving from AWS S3")
        key = "NSMv2.0"+Snotelobs_path.split("NSMv2.0",1)[1]        
        S3.meta.client.download_file(BUCKET_NAME, key,Snotelobs_path)
        snotel_data = pd.read_csv(Snotelobs_path)

    print('Processing prediction location geometry')
    metadata_df = metadata_df.drop(columns=['Unnamed: 0'], axis=1)
    metadata_df['geometry'] = metadata_df.apply(create_polygon, axis=1)
    
    metadata = gpd.GeoDataFrame(metadata_df, geometry='geometry')

    date_columns = snotel_data.columns[1:]
    new_column_names = {col: pd.to_datetime(col, format='%Y-%m-%d').strftime('%Y%m%d') for col in date_columns}
    snotel_data_f = snotel_data.rename(columns=new_column_names)

    print('Processing snotel geometry')
    snotel_geometry = [Point(xy) for xy in zip(snotel_file['longitude'], snotel_file['latitude'])]
    snotel_gdf = gpd.GeoDataFrame(snotel_file, geometry=snotel_geometry)

    final_df = pd.DataFrame()

    #RJs implementation to only do sites once. still need to figure out how to extend to the RegionVal.pkl file..., likely add to this file...
    print('Finding unique sites to locate nearest in situ observation stations.')
    aso_gdf = pd.DataFrame()

    for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):
        aso_file = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))
        aso_gdf = pd.concat([aso_gdf, aso_file])

    aso_gdf.drop_duplicates(subset=['cell_id'], inplace=True)

    # for aso_swe_file in tqdm(os.listdir(aso_swe_files_folder_path)):      #This will need to be added later in order to connect obs to sites

    #     if os.path.isdir(os.path.join(aso_swe_files_folder_path, aso_swe_file)):
    #         continue

        # timestamp = aso_swe_file.split('_')[-1].split('.')[0]

        # aso_gdf = load_aso_snotel_geometry(aso_swe_file, aso_swe_files_folder_path)
        aso_swe_data = pd.read_csv(os.path.join(aso_swe_files_folder_path, aso_swe_file))

        # Calculating nearest SNOTEL sites
        nearest_snotel, distance_cache = calculate_nearest_snotel(aso_gdf, snotel_gdf, n=6)
        
        transposed_data = {}

        if timestamp in new_column_names.values():
            for idx, aso_row in aso_gdf.iterrows():    
                cell_id = idx
                station_ids = nearest_snotel[cell_id]
                selected_snotel_data = snotel_data_f[['station_id', timestamp]].loc[snotel_data_f['station_id'].isin(station_ids)]
                station_mapping = {old_id: f"nearest site {i+1}" for i, old_id in enumerate(station_ids)}
                
                # Rename the station IDs in the selected SNOTEL data
                selected_snotel_data['station_id'] = selected_snotel_data['station_id'].map(station_mapping)

                # Transpose and set the index correctly
                transposed_data[cell_id] = selected_snotel_data.set_index('station_id').T

            transposed_df = pd.concat(transposed_data, axis=0)
            
            # Reset index and rename columns
            transposed_df = transposed_df.reset_index()
            transposed_df.rename(columns={'level_0': 'cell_id', 'level_1': 'Date'}, inplace = True)
            transposed_df['Date'] = pd.to_datetime(transposed_df['Date'])
        
            aso_swe_data['Date'] = pd.to_datetime(timestamp)
            aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]
            merged_df = pd.merge(aso_swe_data, transposed_df, how='left', on=['cell_id', 'Date'])
        
            final_df = pd.concat([final_df, merged_df], ignore_index=True)
        
        else:
            aso_swe_data['Date'] = pd.to_datetime(timestamp)
            aso_swe_data = aso_swe_data[['cell_id', 'Date', 'swe']]
    
            # No need to merge in this case, directly concatenate
            final_df = pd.concat([final_df, aso_swe_data], ignore_index=True)


    # Merge with metadata
    req_cols = ['cell_id', 'lat', 'lon', 'BR_Coord_Long', 'BR_Coord_Lat', 'UR_Coord_Long', 'UR_Coord_Lat',
                'UL_Coord_Long', 'UL_Coord_Lat', 'BL_Coord_Long', 'BL_Coord_Lat', 'geometry']
    Result = final_df.merge(metadata[req_cols], how='left', on='cell_id')

    # Column renaming and ordering
    Result.rename(columns={'swe': 'ASO_SWE_in'}, inplace=True)
    Result = Result[['cell_id', 'Date', 'ASO_SWE_in', 'lat', 'lon', 'nearest site 1', 'nearest site 2',
                     'nearest site 3', 'nearest site 4', 'nearest site 5', 'nearest site 6',
                     'BR_Coord_Long', 'BR_Coord_Lat', 'UR_Coord_Long', 'UR_Coord_Lat',
                     'UL_Coord_Long', 'UL_Coord_Lat', 'BL_Coord_Long', 'BL_Coord_Lat']]

    # Save the merged data to a new file
    output_filename = f"{HOME}/SWEML/data/NSMv2.0/data/TrainingDFs/Merged_aso_snotel_data.parquet"
    Result.to_csv(output_filename, index=False)
    display(Result.head(10))
    print("Processed and saved data")