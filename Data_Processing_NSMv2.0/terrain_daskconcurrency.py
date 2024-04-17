# Import packages
# Dataframe Packages
import numpy as np
from numpy import gradient, rad2deg, arctan2
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely
from shapely.geometry import Point, Polygon
from pyproj import CRS, Transformer

# Raster Packages
import rioxarray as rxr
from rioxarray.merge import merge_arrays
#import rasterstats as rs
from osgeo import gdal
from osgeo import gdalconst

# Data Access Packages
import pystac_client
import planetary_computer

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
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, TimeoutError
import dask
import dask.dataframe as dd
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from requests.exceptions import HTTPError
import dask_jobqueue
from dask.distributed import Client

#Processing using gdal
#In my opinion working gdal is faster compared to richdem
def process_single_location(args):
    lat, lon, index_id, tiles = args

    signed_asset = planetary_computer.sign(tiles[int(index_id)].assets["data"])
    elevation = rxr.open_rasterio(signed_asset.href)
    
    slope = elevation.copy()
    aspect = elevation.copy()

    transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
    xx, yy = transformer.transform(lon, lat)

    tilearray = np.around(elevation.values[0]).astype(int)
    geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)

    driver = gdal.GetDriverByName('MEM')
    temp_ds = driver.Create('', tilearray.shape[1], tilearray.shape[0], 1, gdalconst.GDT_Float32)

    temp_ds.GetRasterBand(1).WriteArray(tilearray)

    tilearray_np = temp_ds.GetRasterBand(1).ReadAsArray()
    grad_y, grad_x = gradient(tilearray_np)

    # Calculate slope and aspect
    slope_arr = np.sqrt(grad_x**2 + grad_y**2)
    aspect_arr = rad2deg(arctan2(-grad_y, grad_x)) % 360 
    
    slope.values[0] = slope_arr
    aspect.values[0] = aspect_arr

    elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
    slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
    asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])

    return elev, slop, asp

#Processing using RichDEM
def process_single_location(args):
    lat, lon, index_id, tiles = args

    signed_asset = sign(tiles[index_id].assets["data"])
    elevation = rxr.open_rasterio(signed_asset.href, masked=True)
    
    slope = elevation.copy()
    aspect = elevation.copy()
    
    transformer = Transformer.from_crs("EPSG:4326", elevation.rio.crs, always_xy=True)
    xx, yy = transformer.transform(lon, lat)
    tilearray = np.around(elevation.values[0]).astype(int)

    #set tile geo to get slope and set at rdarray
    geo = (math.floor(float(lon)), 90, 0.0, math.ceil(float(lat)), 0.0, -90)
    
    tilearray = rd.rdarray(tilearray, no_data = -9999)
    tilearray.projection = 'EPSG:4326'
    tilearray.geotransform = geo
    
    slope_arr = rd.TerrainAttribute(tilearray, attrib='slope_degrees')
    aspect_arr = rd.TerrainAttribute(tilearray, attrib='aspect')

    slope.values[0] = slope_arr
    aspect.values[0] = aspect_arr

    elev = round(elevation.sel(x=xx, y=yy, method="nearest").values[0])
    slop = round(slope.sel(x=xx, y=yy, method="nearest").values[0])
    asp = round(aspect.sel(x=xx, y=yy, method="nearest").values[0])    
  
    return elev, slop, asp

def process_data_in_chunks(df, tiles, num_workers = 16):
    chunk_results = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_single_location, (row.lat, row.lon, row.index_id, tiles)) for index, row in df.iterrows()]
        
        #for future in tqdm(as_completed(futures), total=len(futures)):
        for future in tqdm(futures):
            try:
                chunk_results.append(future.result(timeout=10))
            except TimeoutError:
                print("Processing location timed out.")
                continue
            except HTTPError as e:
                print(f"Failed to process a location due to HTTPError: {e}")
                continue  

    return pd.DataFrame(chunk_results, columns=['Elevation_m', 'Slope_Deg', 'Aspect_L'])     

###Dask implementation to process the dataframe in chunks###
def extract_terrain_data(geospatial_df, num_workers=16):

    cores_per_worker = 1  
    memory_per_worker = "3GB"    #Adjust based on the necessity
    
    cluster = dask_jobqueue.SLURMCluster(cores=cores_per_worker,
                                        memory=memory_per_worker, 
                                        local_directory=r'/home/vgindi/slurmcluster')

    # Scale the cluster to workers based on the requirement
    cluster.scale(num_workers)
    dask_client = Client(cluster)
    dask_client
    
    client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)

    #coordinates obtained from the grid_cells_meta data
    area_of_interest = {'type': 'Polygon', 
                        'coordinates': [[[-120.37693519839556, 36.29213061937931],
                                        [-120.37690215328962, 38.8421802805432], 
                                        [-118.29165268221286, 38.84214595220293],
                                        [-118.2917116398743, 36.29209713778364], 
                                        [-120.37693519839556, 36.29213061937931]]]}
    
    search = client.search(collections=["cop-dem-glo-90"], intersects=area_of_interest)
    
    tiles = list(search.items())
    geospatial_df = geospatial_df[['lat', 'lon', 'index_id']]

    dask_df = dd.from_pandas(geospatial_df, npartitions=5)
    results = dask_df.map_partitions(process_data_in_chunks, tiles=tiles, num_workers=num_workers,
                                 meta=pd.DataFrame(columns=['Elevation_m', 'Slope_Deg', 'Aspect_L']))


    final_result = results.compute(scheduler='processes')
        
    #result_df = pd.concat([geospatial_df.reset_index(drop=True), final_result.reset_index(drop=True)], axis=1) 
    return final_result

