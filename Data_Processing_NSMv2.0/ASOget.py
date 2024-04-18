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
from tensorflow.keras.models import load_model
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


class ASODataTool:
    def __init__(self, short_name, version, polygon='', filename_filter=''):
        self.short_name = short_name
        self.version = version
        self.polygon = polygon
        self.filename_filter = filename_filter
        self.url_list = []
        self.CMR_URL = 'https://cmr.earthdata.nasa.gov'
        self.CMR_PAGE_SIZE = 2000
        self.CMR_FILE_URL = ('{0}/search/granules.json?provider=NSIDC_ECS'
                             '&sort_key[]=start_date&sort_key[]=producer_granule_id'
                             '&scroll=true&page_size={1}'.format(self.CMR_URL, self.CMR_PAGE_SIZE))

    def cmr_search(self, time_start, time_end, bounding_box):
        try:
            if not self.url_list:
                self.url_list = NSIDC_Data.cmr_search(
                    self.short_name, self.version, time_start, time_end,
                    bounding_box=self.bounding_box, polygon=self.polygon,
                    filename_filter=self.filename_filter, quiet=False)
            return self.url_list
        except KeyboardInterrupt:
            quit()

    def cmr_download(self, directory):
        dpath = f"{HOME}/SWEML/data/NSMv2.0/data/ASO/{directory}"
        if not os.path.exists(dpath):
            os.makedirs(dpath, exist_ok=True)

        NSIDC_Data.cmr_download(self.url_list, dpath, False)

    @staticmethod
    def get_bounding_box(region):
        try:
            regions = pd.read_pickle(f"{HOME}/SWEML/data/PreProcessed/RegionVal.pkl")
        except:
            print('File not local, getting from AWS S3.')
            key = f"data/PreProcessed/RegionVal.pkl"            
            S3.meta.client.download_file(BUCKET_NAME, key,f"{HOME}/SWEML/data/PreProcessed/RegionVal.pkl")
            regions = pd.read_pickle(f"{HOME}/SWEML/data/PreProcessed/RegionVal.pkl")


        
        superset = []

        superset.append(regions[region])
        superset = pd.concat(superset)
        superset = gpd.GeoDataFrame(superset, geometry=gpd.points_from_xy(superset.Long, superset.Lat, crs="EPSG:4326"))
        bounding_box = list(superset.total_bounds)

        return f"{bounding_box[0]},{bounding_box[1]},{bounding_box[2]},{bounding_box[3]}"

class ASODownload(ASODataTool):
    def __init__(self, short_name, version, polygon='', filename_filter=''):
        super().__init__(short_name, version, polygon, filename_filter)
        self.region_list =    [ 'N_Sierras',
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
                                'Or_Coast'  ]

    def select_region(self):
        print("Select a region by entering its index:")
        for i, region in enumerate(self.region_list, start=1):
            print(f"{i}. {region}")

        try:
            user_input = int(input("Enter the index of the region: "))
            if 1 <= user_input <= len(self.region_list):
                selected_region = self.region_list[user_input - 1]
                self.bounding_box = self.get_bounding_box(selected_region)
                print(f"You selected: {selected_region}")
                print(f"Bounding Box: {self.bounding_box}")
            else:
                print("Invalid index. Please select a valid index.")
        except ValueError:
            print("Invalid input. Please enter a valid index.")
            


class ASODataProcessing:
    
    @staticmethod
    def processing_tiff(input_file, output_res):
        try:
            date = os.path.splitext(input_file)[0].split("_")[-1]
            
            # Define the output file path
            output_folder = os.path.join(os.getcwd(), "Processed_Data")
            os.makedirs(output_folder, exist_ok=True)
            output_file = os.path.join(output_folder, f"ASO_100M_{date}.tif")
    
            ds = gdal.Open(input_file)
            if ds is None:
                print(f"Failed to open '{input_file}'. Make sure the file is a valid GeoTIFF file.")
                return None
            
            # Reproject and resample
            gdal.Warp(output_file, ds, dstSRS="EPSG:4326", xRes=output_res, yRes=-output_res, resampleAlg="bilinear")
    
            # Read the processed TIFF file using rasterio
            rds = rxr.open_rasterio(output_file)
            rds = rds.squeeze().drop("spatial_ref").drop("band")
            rds.name = "data"
            df = rds.to_dataframe().reset_index()
            return df
    
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None
        
    @staticmethod
    def convert_tiff_to_csv(input_folder, output_res):

        #curr_dir = os.getcwd()
        dir = f"{HOME}/SWEML/data/NSMv2.0/data/ASO/"
        folder_path = os.path.join(dir, input_folder)
        
        # Check if the folder exists and is not empty
        if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
            print(f"The folder '{folder_path}' does not exist.")
            return
        
        if not os.listdir(folder_path):
            print(f"The folder '{input_folder}' is empty.")
            return
    
        tiff_files = [filename for filename in os.listdir(folder_path) if filename.endswith(".tif")]
        print(f"Converting {len(tiff_files)} ASO tif files to csv'")
        # Create CSV files from TIFF files
        for tiff_filename in tqdm(tiff_files):
            
            # Open the TIFF file
            tiff_filepath = os.path.join(folder_path, tiff_filename)
            df = ASODataProcessing.processing_tiff(tiff_filepath, output_res)
    
            if df is not None:
                # Get the date from the TIFF filename
                date = os.path.splitext(tiff_filename)[0].split("_")[-1]
    
                # Define the CSV filename and folder
                csv_filename = f"ASO_SWE_{date}.csv"
                csv_folder = os.path.join(dir, "SWE_csv")
                os.makedirs(csv_folder, exist_ok=True)
                csv_filepath = os.path.join(csv_folder, csv_filename)
    
                # Save the DataFrame as a CSV file
                df.to_csv(csv_filepath, index=False)
    
                #print(f"Converted '{tiff_filename}' to '{csv_filename}'")
                
    def create_polygon(self, row):
        return Polygon([(row['BL_Coord_Long'], row['BL_Coord_Lat']),
                        (row['BR_Coord_Long'], row['BR_Coord_Lat']),
                        (row['UR_Coord_Long'], row['UR_Coord_Lat']),
                        (row['UL_Coord_Long'], row['UL_Coord_Lat'])])

    def process_folder(self, input_folder, metadata_path, output_folder):
        # Import the metadata into a pandas DataFrame
        '''
        input_folder = f"{HOME}/data/NSMv2.0/data/Processed_Data/SWE_csv"
        metadata_path = f"{HOME}/data/NSMv2.0/data/Provided_Data/grid_cells_meta.csv"
        output_folder = f"{HOME}/data/NSMv2.0/data/Processed_SWE"
        '''
        try:
            pred_obs_metadata_df = pd.read_csv(metadata_path)
        except:
            key = "NSMv2.0"+metadata_path.split("NSMv2.0",1)[1]        
            S3.meta.client.download_file(BUCKET_NAME, key,metadata_path)
            pred_obs_metadata_df = pd.read_csv(metadata_path)


        # Get all SWE_csv into the input folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]

            
    
        # Assuming create_polygon is defined elsewhere, we add a column with polygon geometries
        pred_obs_metadata_df = pred_obs_metadata_df.drop(columns=['Unnamed: 0'], axis=1)
        pred_obs_metadata_df['geometry'] = pred_obs_metadata_df.apply(self.create_polygon, axis=1)
    
        # Convert the DataFrame to a GeoDataFrame
        metadata = gpd.GeoDataFrame(pred_obs_metadata_df, geometry='geometry')
    
        # Drop coordinates columns
        metadata_df = metadata.drop(columns=['BL_Coord_Long', 'BL_Coord_Lat', 
                                             'BR_Coord_Long', 'BR_Coord_Lat', 
                                             'UR_Coord_Long', 'UR_Coord_Lat', 
                                             'UL_Coord_Long', 'UL_Coord_Lat'], axis=1)
    
        # List all CSV files in the input folder
        csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    
        for csv_file in csv_files:
            input_aso_path = os.path.join(input_folder, csv_file)
            output_aso_path = os.path.join(output_folder, csv_file)
    
            # Check if the output file already exists
            if os.path.exists(output_aso_path):
                print(f"CSV file {csv_file} already exists in the output folder.")
                continue
    
            # Process each CSV file
            aso_swe_df = pd.read_csv(input_aso_path)
    
            # Convert the "aso_swe_df" into a GeoDataFrame with point geometries
            geometry = [Point(xy) for xy in zip(aso_swe_df['x'], aso_swe_df['y'])]
            aso_swe_geo = gpd.GeoDataFrame(aso_swe_df, geometry=geometry)

            result = gpd.sjoin(aso_swe_geo, metadata_df, how='left', predicate='within', op = 'intersects')
    
            # Select specific columns for the final DataFrame
            Final_df = result[['y', 'x', 'data', 'cell_id']]
            Final_df.rename(columns={'data': 'swe'}, inplace=True)
    
            # Drop rows where 'cell_id' is NaN
            if Final_df['cell_id'].isnull().values.any():
                Final_df = Final_df.dropna(subset=['cell_id'])
    
            # Save the processed DataFrame to a CSV file
            Final_df.to_csv(output_aso_path, index=False)
            print(f"Processed {csv_file}")
            
    def converting_ASO_to_standardized_format(self, input_folder, output_csv):
        
        # Initialize an empty DataFrame to store the final transformed data
        final_df = pd.DataFrame()
    
        # Iterate through all CSV files in the directory
        for filename in os.listdir(input_folder):
            if filename.endswith(".csv"):
                file_path = os.path.join(input_folder, filename)
    
                # Extract the time frame from the filename
                time_frame = filename.split('_')[-1].split('.')[0]
    
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)
    
                # Rename the 'SWE' column to the time frame for clarity
                df = df.rename(columns={'SWE': time_frame})
    
                # Merge or concatenate the data into the final DataFrame
                if final_df.empty:
                    final_df = df
                else:
                    final_df = pd.merge(final_df, df, on='cell_id', how='outer')
    
        # Save the final transformed DataFrame to a single CSV file
        final_df.to_csv(output_csv, index=False)