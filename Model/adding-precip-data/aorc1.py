#!/usr/bin/env python3

"""
Author: Tony Castronova <acastronova@cuahsi.org>
Date: October 2023

This file contains helper functions for working with AORC v1.0 meteorological data
that has Kerchunk headers. The Kerchunk headers that we'll be using are created and
maintained by the Alabama Water Institute.
"""

import sys
import dask
import zarr
import numpy
import xarray
import pyproj
from s3fs import S3FileSystem
from dask.distributed import Client
from dask.distributed import progress
from kerchunk.combine import MultiZarrToZarr

BUCKET = 's3://ciroh-nwm-zarr-retrospective-data-copy/noaa-nwm-retrospective-2-1-zarr-pds/forcing/'
WRFHYDRO_META = 'http://thredds.hydroshare.org/thredds/dodsC/hydroshare/resources/2a8a3566e1c84b8eb3871f30841a3855/data/contents/WRF_Hydro_NWM_geospatial_data_template_land_GIS.nc'

def load_aorc_dataset(year, month=None, day=None):

    # find AORC files on AWS
    s3 = S3FileSystem(anon=True)
    if (month is not None) and (day is None):
        files = s3.glob(f'{BUCKET}{year}/{year}{month:02}*')
    elif day is not None:
        files = s3.glob(f'{BUCKET}{year}/{year}{month:02}{day:02}*')
    else:
        files = s3.glob(f'{BUCKET}{year}/{year}*')
    
    json_list = []
    for f in files:
        parts = f.split('/')
        parts[0] += '.s3.amazonaws.com'
        parts.insert(0, 'https:/')
        new_name = '/'.join(parts)
        json_list.append(new_name)

    # build and load json heade urls for these files
    #json_list = new_files[0:217] 
    mzz = MultiZarrToZarr(json_list,
        remote_protocol='s3',
        remote_options={'anon':True},
        concat_dims=['valid_time'])
    d = mzz.translate()
    backend_args = {"consolidated": False,
                    "storage_options": {"fo": d},
                    "consolidated": False}
    ds = xarray.open_dataset("reference://", engine="zarr", backend_kwargs=backend_args)

    # clean the dataset, add spatial metadata, create lat/lon coordinates
    ds = ds.squeeze(dim='Time')
    ds_meta = xarray.open_dataset(WRFHYDRO_META)
    
    leny = len(ds_meta.y)
    x = ds_meta.x.values
    y = ds_meta.y.values
    
    ds = ds.rename({'valid_time': 'time', 'south_north':'y', 'west_east':'x'})
    #ds.rename_dims(south_north='y', west_east='x', valid_time='time')
    
    X, Y = numpy.meshgrid(x, y)
    
    # define the input crs
    wrf_proj = pyproj.Proj(proj='lcc',
                           lat_1=30.,
                           lat_2=60., 
                           lat_0=40.0000076293945, lon_0=-97., # Center point
                           a=6370000, b=6370000)
    
    # define the output crs
    wgs_proj = pyproj.Proj(proj='latlong', datum='WGS84')
    
    # transform X, Y into Lat, Lon
    transformer = pyproj.Transformer.from_crs(wrf_proj.crs, wgs_proj.crs)
    lon, lat = transformer.transform(X, Y)
    
    ds = ds.assign_coords(lon = (['y', 'x'], lon))
    ds = ds.assign_coords(lat = (['y', 'x'], lat))
    ds = ds.assign_coords(x = x)
    ds = ds.assign_coords(y = y)
    
    ds.x.attrs['axis'] = 'X'
    ds.x.attrs['standard_name'] = 'projection_x_coordinate'
    ds.x.attrs['long_name'] = 'x-coordinate in projected coordinate system'
    ds.x.attrs['resolution'] = 1000.  # cell size
    
    ds.y.attrs['axis'] = 'Y' 
    ds.y.attrs['standard_name'] = 'projection_y_coordinate'
    ds.y.attrs['long_name'] = 'y-coordinate in projected coordinate system'
    ds.y.attrs['resolution'] = 1000.  # cell size
    
    ds.lon.attrs['units'] = 'degrees_east'
    ds.lon.attrs['standard_name'] = 'longitude' 
    ds.lon.attrs['long_name'] = 'longitude'
    
    ds.lat.attrs['units'] = 'degrees_north'
    ds.lat.attrs['standard_name'] = 'latitude' 
    ds.lat.attrs['long_name'] = 'latitude'
    
    # add crs to netcdf file
    ds.rio.write_crs(ds_meta.crs.attrs['spatial_ref'], inplace=True
                    ).rio.set_spatial_dims(x_dim="x",
                                           y_dim="y",
                                           inplace=True,
                                           ).rio.write_coordinate_system(inplace=True);
    return ds



    