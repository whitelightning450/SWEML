#created by Josh Christensen and Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH) REU Summer 2023
# SWEET supported by the University of Alabama and the Alabama Water Institute
# 10-19-2023

import requests
import getpass
import socket
import json
import zipfile
import io
import math
import os
import shutil
import pprint
import re
import time
import geopandas as gpd
import fiona
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, mapping
from shapely.geometry.polygon import orient
from statistics import mean
from requests.auth import HTTPBasicAuth
from xml.etree import ElementTree as ET

import netrc
from pprint import pprint
from typing import Union
from datetime import datetime
from pathlib import Path

def get_credentials():
    """
    Get credentials from .netrc file
    """
    try:
        info = netrc.netrc()
        username, account, password = info.authenticators("urs.earthdata.nasa.gov")
        print('credentials not found, please enter them')
    except Exception as e:
        username = input("Earthdata Login Username: ")
        password = getpass.getpass("Earthdata Login Password: ")
        account = input("Earthdata Login Email: ")
    return username, password, account

def get_latest_version(short_name: str):
    """
    Get the latest version of a data product
    """
    params = {
        'short_name': short_name
    }

    cmr_collections_url = 'https://cmr.earthdata.nasa.gov/search/collections.json'
    response = requests.get(cmr_collections_url, params=params)
    results = json.loads(response.content)
    pprint(results)

    return

def format_date(start_date: datetime, end_date: datetime = None):
    """
        Format date for use in CMR search

        Parameters
            start_date (datetime): Start date
            end_date (datetime): End date (optional)

        Returns
            str: Formatted date string
    """
    if end_date is None:
        end_date = start_date.replace(hour=23, minute=59, second=59)

    return f"{start_date.strftime('%Y-%m-%dT%H:%M:%SZ')},{end_date.strftime('%Y-%m-%dT%H:%M:%SZ')}"

def format_boundingbox(bbox: Union[list, tuple]):
    """
        Format bounding box for use in CMR search

        Parameters
            bbox (list or tuple): Bounding box coordinates
            [lower left longitude, lower left latitude, upper right longitude, upper right latitude]

        Returns
            str: Formatted bounding box string
    """
    return f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"

def discoverGranules(short_name: str, version: str, temporal: str, bbox: str):
    """
        Discover the granules for a given search

        Parameters
            short_name (str): Short name of the data product
            version (str): Version of the data product, must be formatted correctly
            temporal (str): Temporal range of the data product, must be formated correctly
            bbox (str): Bounding box of the data product, must be formatted correctly

        Returns
            granules (list): List of granules
    """
    granule_search_url = 'https://cmr.earthdata.nasa.gov/search/granules'
    search_params = {
        'short_name': short_name,
        'version': version,
        'temporal': temporal,
        'page_size': 100,
        'page_num': 1,
        'bounding_box': bbox
    }

    found_granules = []
    headers = {"Accept": "application/json"}

    while True:
        # try:
        response = requests.get(granule_search_url, params=search_params, headers=headers)
        results = json.loads(response.content)
        # except Exception as e:
        #     print(e)
        #     print(f"Response content: {response.content}")
        #     break  # Exit the loop if there is an error

        if len(results['feed']['entry']) == 0:
            break  # Exit the loop if there are no more granules

        # Add the granules from the current page to the list
        found_granules.extend(results['feed']['entry'])
        search_params['page_num'] += 1

    return found_granules

def getDownloadSize(granules: list):
    """
        Get the total download size of a list of granules

        Parameters
            granules (list): List of granules

        Returns
            size (float): Total download size in MB
    """
    size = 0
    for granule in granules:
        size += float(granule['granule_size'])

    return size

def download(short_name: str, version: str, temporal:str, bbox:str,
             folder: Union[str, Path],
             mode: str = None,
             format:str = "GeoTIFF",
             subset:str = "/VIIRS_Grid_IMG_2D/CGF_NDSI_Snow_Cover", #need to change this to adjust for the appropriate v1/v2 data products
             # threading: bool = False
             ):
    """
        Download data from NSIDC. Credit to NSIDC and their Data Access Notebook (https://github.com/nsidc/NSIDC-Data-Access-Notebook)
        

        Parameters:
            short_name (str): Short name of the data product
            version (str): Version of the data product, must be formatted correctly
            temporal (str): Temporal range of the data product, must be formated correctly
            bbox (str): Bounding box of the data product, must be formatted correctly
            folder (str or Path): Folder to save the data to
            mode (str): Method of downloading ["stream", "async"] defaults to auto (less than 100 done synchronously)
            format (str): Format of the data product, defaults to GeoTIFF
            subset (str): Subset of the data product, defaults to "/VIIRS_Grid_IMG_2D/CGF_NDSI_Snow_Cover", must be comma separated - change to '/NPP_Grid_IMG_2D/VNP10A1_NDSI_Snow_Cover'  for dates < 2018
            # threading (bool): Whether to use threading, defaults to False  # TODO not implemented
    """

    # Get credentials
    username, password, account = get_credentials()

    # Get granules
    granules = discoverGranules(short_name, version, temporal, bbox)

    if mode is None:
        if len(granules) < 100:
            mode = "stream"
        else:
            mode = "async"

    if mode == "async":
        page_size = 2000
    else:
        page_size = 100

    page_num = math.ceil(len(granules) / page_size)

    base_url = "https://n5eil02u.ecs.nsidc.org/egi/request"  # NSIDC API base URL

    param_dict = {
        'short_name': short_name,
        'version': version,
        'temporal': temporal,
        'page_size': page_size,
        'bounding_box': bbox,
        'projection': 'GEOGRAPHIC',
        # 'bbox': bbox,  # crops results to bbox
        'format': format,
        'agent': '',
        'email': account,
        'Coverage': subset,
        'request_mode': mode
    }

    # # Convert to string
    # param_string = '&'.join("{!s}={!r}".format(k, v) for (k, v) in param_dict.items())
    # param_string = param_string.replace("'", "")
    #
    # endpoint_list = []
    # for i in range(page_num):
    #     page_val = i + 1  # page numbers start at 1
    #     API_request = api_request = f'{base_url}?{param_string}&page_num={page_val}'
    #     endpoint_list.append(API_request)

    # Create folder if it doesn't exist
    path = str(folder)
    if not os.path.exists(path):
        os.mkdir(path)

    session = requests.session()

    if mode == "async":
        # Request data service for each page number, and unzip outputs
        for i in range(page_num):
            start = time.time()  # Start timer
            page_val = i + 1
            print('Order: ', page_val)

            # For all requests other than spatial file upload, use get function
            request = session.get(base_url, params=param_dict, auth=(username, password))

            # print('Request HTTP response: ', request.status_code)

            # Raise bad request: Loop will stop for bad response code.
            request.raise_for_status()
            # print('Order request URL: ', request.url)
            esir_root = ET.fromstring(request.content)
            # print('Order request response XML content: ', request.content)

            # Look up order ID
            orderlist = []
            for order in esir_root.findall("./order/"):
                orderlist.append(order.text)
            orderID = orderlist[0]
            print('order ID: ', orderID)

            # Create status URL
            statusURL = base_url + '/' + orderID
            print('status URL: ', statusURL)

            # Find order status
            request_response = session.get(statusURL)
            # print('HTTP response from order response URL: ', request_response.status_code)

            # Raise bad request: Loop will stop for bad response code.
            request_response.raise_for_status()
            request_root = ET.fromstring(request_response.content)
            statuslist = []
            for status in request_root.findall("./requestStatus/"):
                statuslist.append(status.text)
            status = statuslist[0]
            print('Data request ', page_val, ' is submitting...')
            print('Initial request status is ', status)

            # Continue loop while request is still processing
            while status == 'pending' or status == 'processing':
                time.sleep(10)
                loop_response = session.get(statusURL)

                # Raise bad request: Loop will stop for bad response code.
                loop_response.raise_for_status()
                loop_root = ET.fromstring(loop_response.content)

                # find status
                statuslist = []
                for status in loop_root.findall("./requestStatus/"):
                    statuslist.append(status.text)
                status = statuslist[0]
                print("\r", f'{int((time.time() - start)//60)}min - Request status is: ', status, end='')
                if status == 'pending' or status == 'processing':
                    continue

            # Order can either complete, complete_with_errors, or fail:
            # Provide complete_with_errors error message:
            if status == 'complete_with_errors' or status == 'failed':
                messagelist = []
                for message in loop_root.findall("./processInfo/"):
                    messagelist.append(message.text)
                print('error messages:')
                pprint.pprint(messagelist)

            # Download zipped order if status is complete or complete_with_errors
            if status == 'complete' or status == 'complete_with_errors':
                downloadURL = 'https://n5eil02u.ecs.nsidc.org/esir/' + orderID + '.zip'
                print('Zip download URL: ', downloadURL)
                print('Beginning download of zipped output...')
                zip_response = session.get(downloadURL)
                # Raise bad request: Loop will stop for bad response code.
                zip_response.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(zip_response.content)) as z:
                    z.extractall(path)
                print('Data request', page_val, 'is complete.')
            else:
                print('Request failed.')
    else:
        for i in range(page_num):
            page_val = i + 1
            print('Order: ', page_val)
            print('Requesting...')
            request = session.get(base_url, params=param_dict)
            print('HTTP response from order response URL: ', request.status_code)
            request.raise_for_status()
            d = request.headers['content-disposition']
            fname = re.findall('filename=(.+)', d)
            dirname = os.path.join(path, fname[0].strip('\"'))
            print('Downloading...')
            open(dirname, 'wb').write(request.content)
            print('Data request', page_val, 'is complete.')

            # Unzip outputs
        for z in os.listdir(path):
            if z.endswith('.zip'):
                zip_name = path + "/" + z
                zip_ref = zipfile.ZipFile(zip_name)
                zip_ref.extractall(path)
                zip_ref.close()
                os.remove(zip_name)

    # Move files from subfolders to parent folder
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                os.replace(os.path.join(root, file), os.path.join(path, file))
            except OSError:
                pass
        for name in dirs:
            os.rmdir(os.path.join(root, name))

    time.sleep(5)  # TODO investigate further, this is a hack to prevent the filesystem from not finding the moved files


if __name__ == '__main__':
    folder = "/Users/jmac/Documents/Programming/REU/National-Snow-Model/Data_Processing_Assimilation/download_test"
    short_name = "VNP10A1F"
    version = "2"
    temporal = format_date(datetime(2019,1,1), datetime(2019,1,1,23,59,59))
    boundingbox = "-123.34078531,33.35825379,-105.07803558,48.97106571" # CONUS

    print("Discovering...")
    granules = discoverGranules(short_name, version, temporal, boundingbox)

    print(granules)
    print(len(granules))
    print(getDownloadSize(granules))

    print("Downloading...")
    download(short_name, version, temporal, boundingbox, folder, mode="async")
    print("DONE!")