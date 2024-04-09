from dask.distributed import Client, LocalCluster
from concurrent.futures import ProcessPoolExecutor
from dask import delayed
import dask.dataframe as dd
import pandas as pd
import math
from pyproj import Transformer
import rioxarray as rxr
import numpy as np
from osgeo import gdal, gdalconst
from math import floor, ceil, sqrt, atan2, rad2deg
from numpy import rad2deg, arctan2, gradient
import dask_jobqueue
import pystac_client
import planetary_computer
from tqdm import tqdm
from dask.diagnostics import ProgressBar
from retrying import retry

def process_single_location(args):
    lat, lon, elevation = args
    
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

def process_data_in_chunks(tile_group, tiles):
    results = []

    index_id = int(tile_group['index_id'].iloc[0])
    tile = tiles[index_id]

    @retry(stop_max_attempt_number=3, wait_fixed=2000, retry_on_exception=lambda exception: isinstance(exception, IOError))
    def fetch_and_process_elevation():
        signed_asset = planetary_computer.sign(tile.assets["data"])
        elevation = rxr.open_rasterio(signed_asset.href)

        for lat, lon in zip(tile_group['lat'], tile_group['lon']):
            try:
                result = process_single_location((lat, lon, elevation))
                results.append(result)
            except Exception as e:
                print(f"Error processing location (lat: {lat}, lon: {lon}) due to {e}. Skipping...")
    
    fetch_and_process_elevation()
    return pd.DataFrame(results, columns=['Elevation_m', 'Slope_Deg', 'Aspect_L'])

def process_data_in_chunks_dask(tile_group, tiles):

    index_id = int(tile_group['index_id'].iloc[0])
    tile = tiles[index_id]
    
    signed_asset = planetary_computer.sign(tile.assets["data"])
    elevation = rxr.open_rasterio(signed_asset.href)
    
    results = [delayed(process_single_location)((lat, lon, elevation)) for lat, lon in zip(tile_group['lat'], tile_group['lon'])]
    return pd.DataFrame(results, columns=['Elevation_m', 'Slope_Deg', 'Aspect_L'])

def extract_terrain_data(df):

    
    #cluster = dask_jobqueue.SLURMCluster(local_directory=r'/home/vgindi/slurmcluster') 
    #cluster.scale(num_workers)
    #dask_client = Client(cluster)
    
    client = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1", ignore_conformance=True)
    
    area_of_interest = {'type': 'Polygon', 
                        'coordinates': [[[-120.37693519839556, 36.29213061937931],
                                        [-120.37690215328962, 38.8421802805432], 
                                        [-118.29165268221286, 38.84214595220293],
                                        [-118.2917116398743, 36.29209713778364], 
                                        [-120.37693519839556, 36.29213061937931]]]}
    
    search = client.search(collections=["cop-dem-glo-90"], intersects=area_of_interest)
    
    tiles = list(search.items())
    df = df[['lat', 'lon', 'index_id']]

    # Convert the DataFrame to a Dask DataFrame for distributed computing
    dask_df = dd.from_pandas(df, npartitions = df['index_id'].nunique())
    
    # Process each partition (grouped by 'index_id') in parallel
    results = dask_df.groupby('index_id').apply(lambda group: process_data_in_chunks(group, tiles), 
                                                meta=pd.DataFrame(columns=['Elevation_m', 'Slope_Deg', 'Aspect_L']))
    with ProgressBar():
        result_df = results.compute()

    #dask_client.close()
    #cluster.close()
    
    final_df = pd.concat([df.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    return result_df
