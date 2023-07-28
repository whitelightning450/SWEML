# Import packages

# NSM Packages
from National_Snow_Model import SWE_Prediction

# Dataframe Packages
import numpy as np
import xarray as xr
import pandas as pd

# Vector Packages
import geopandas as gpd
import shapely

# Raster Packages
import rioxarray as rxr
from rioxarray.merge import merge_arrays
import rasterstats as rs

# Data Access Packages
import earthaccess as ea
from Data_Processing_Assimilation.nsidc_fetch import download, format_date, format_boundingbox
import h5py
import pickle
from tensorflow.keras.models import load_model

# General Packages
import re
from datetime import datetime
import glob
from pprint import pprint
from typing import Union
from pathlib import Path
from tqdm import tqdm
import time


class NSM_SCA(SWE_Prediction):

    def __init__(self, cwd: Union[str, Path], date: Union[str, datetime], delta=7, timeDelay=3, threshold=0.2):
        """
            Initializes the NSM_SCA class by calling the superclass constructor.

            Parameters:
                cwd (str): The current working directory.
                date (str): The date of the prediction.
                delta (int): How many days back to go for Last SWE.
                timeDelay (int): Use the SCA rasters from [timeDelay] days ago. Simulates operations in the real world.
                threshold (float): The threshold for NDSI, if greater than this value, it is considered to be snow.
        """
        if type(cwd) != Path:
            cwd = Path(cwd)  # Convert to Path object if necessary

        if type(date) != datetime:
            date = datetime.strptime(date, "%Y-%m-%d")  # Convert to datetime object if necessary

        # Call superclass constructor
        SWE_Prediction.__init__(self, cwd=str(cwd), date=date.strftime("%Y-%m-%d"), delta=delta)

        self.timeDelay = timeDelay
        self.delayedDate = date - pd.Timedelta(days=timeDelay)

        self.SCA_folder = self.cwd + "/Data/VIIRS_SCA/"
        self.threshold = threshold * 100  # Convert percentage to values used in VIIRS NDSI

        self.auth = ea.login(strategy="netrc")
        if self.auth is None:
            print("Error logging into Earth Data account. Things will probably break")

    def initializeGranules(self, bbox: list[float, float, float, float],
                           dataFolder: Union[str, Path]):
        """
            Initalizes SCA information by fetching granules and merging them.

            Parameters:
                bbox (list[float, float, float, float]): The bounding box to fetch granules for.
                dataFolder (str): The folder with the granules.

            Returns:
                None - Initializes the following class variables: extentDF, granules, raster
        """
        self.extentDF = calculateGranuleExtent(bbox, self.delayedDate)  # Get granule extent
        self.granules = fetchGranules(bbox, dataFolder, self.delayedDate, self.extentDF)  # Fetch granules
        self.raster = createMergedRxr(self.granules["filepath"])  # Merge granules


    def getPredictionExtent(self):
        """
            Gets the extent of the prediction dataframe.

            Returns:
                extent (list[float, float, float, float]): The extent of the prediction dataframe.
        """
        regions = pd.read_pickle(self.cwd + "/Data/Processed/RegionVal.pkl")

        superset = []

        for region in regions:
            superset.append(regions[region])

        superset = pd.concat(superset)
        superset = gpd.GeoDataFrame(superset, geometry=gpd.points_from_xy(superset.Long, superset.Lat, crs="EPSG:4326"))

        return superset.total_bounds

    def augment_SCA(self, region: str):
        """
            Augments the region's forecast dataframe with SCA data.

            Parameters:
                region (str): The region to augment.

            Returns:
                adf (GeoDataFrame): The augmented dataframe.
        """

        # Load forecast dataframe
        try:
            self.Forecast  # Check if forecast dataframe has been initialized
        except AttributeError:
            path = self.cwd + "/Data/Processed/Prediction_DF_" + self.date + ".pkl"
            self.Forecast = pd.read_pickle(path)

        region_df = self.Forecast[region]
        geoRegionDF = gpd.GeoDataFrame(region_df, geometry=gpd.points_from_xy(region_df.Long, region_df.Lat,
                                                                              crs="EPSG:4326"))  # Convert to GeoDataFrame

        try:
            regional_raster = self.raster  # Check if raster has been initialized
        except AttributeError:
            # Fetch granules and merge them
            # region_extentDF = calculateGranuleExtent(geoRegionDF.total_bounds, self.delayedDate)  # Get granule extent TODO fix delayedDate
            region_granules = fetchGranules(geoRegionDF.total_bounds, self.SCA_folder,
                                            self.delayedDate)  # Fetch granules
            regional_raster = createMergedRxr(region_granules["filepath"])  # Merge granules

        adf = augmentGeoDF(geoRegionDF, regional_raster, buffer=500, threshold=self.threshold)  # Buffer by 500 meters -> 1km square
        # adf.drop(columns=["geometry"], inplace=True)  # Drop geometry column

        return adf

    def augmentPredictionDFs(self):
        """
            Augments the forecast dataframes with SCA data.
        """
        print("Calculating mean SCA for each geometry in each region...")
        self.Forecast = pd.read_pickle(self.cwd + "/Data/Processed/Prediction_DF_" + self.date + ".pkl")

        # Augment each Forecast dataframes
        for region in tqdm(self.Region_list):
            self.Forecast[region] = self.augment_SCA(region).drop(columns=["geometry"])

        # Save augmented forecast dataframes
        path = self.cwd + "/Data/Processed/Prediction_DF_SCA_" + self.date + ".pkl"
        file = open(path, "wb")

        # write the python object (dict) to pickle file
        pickle.dump(self.Forecast, file)

        # close file
        file.close()

    def SWE_Predict(self, SCA=True):
        # load first SWE observation forecasting dataset with prev and delta swe for observations.

        if SCA:
            path = self.cwd + "/Data/Processed/Prediction_DF_SCA_" + self.date + ".pkl"
        else:
            path = self.cwd + '/Data/Processed/Prediction_DF_' + self.date + '.pkl'

        # load regionalized forecast data
        self.Forecast = open(path, "rb")

        self.Forecast = pickle.load(self.Forecast)

        # load RFE optimized features
        self.Region_optfeatures = pickle.load(
            open(self.cwd + "/Model/Prev_SWE_Models_Final/opt_features_prevSWE.pkl", "rb"))

        # Reorder regions
        self.Forecast = {k: self.Forecast[k] for k in self.Region_list}

        # Make and save predictions for each region
        self.Prev_df = pd.DataFrame()
        self.predictions = {}
        print('Making predictions for: ', self.date)

        for Region in self.Region_list:
            print(Region)
            self.predictions[Region] = self.Predict(Region)
            self.predictions[Region] = pd.DataFrame(self.predictions[Region])

            #  if self.plot == True:
            #     del self.predictions[Region]['geometry']
            self.Prev_df = pd.concat([self.Prev_df, self.predictions[Region][[self.date]]])  # pandas 2.0 update
            # self.Prev_df = self.Prev_df.append(pd.DataFrame(self.predictions[Region][self.date]))
            self.Prev_df = pd.DataFrame(self.Prev_df)

            self.predictions[Region].to_hdf(self.cwd + '/Predictions/predictions' + self.date + '.h5', key=Region)

        # load submission DF and add predictions, if locations are removed or added, this needs to be modified
        self.subdf = pd.read_csv(self.cwd + '/Predictions/submission_format_' + self.prevdate + '.csv')
        self.subdf.index = list(self.subdf.iloc[:, 0].values)
        self.subdf = self.subdf.iloc[:, 1:]  # TODO replace with drop("cell_id")

        self.sub_index = self.subdf.index
        # reindex predictions
        self.Prev_df = self.Prev_df.loc[self.sub_index]
        self.subdf[self.date] = self.Prev_df[self.date].astype(float)
        # subdf.index.names = [' ']
        self.subdf.to_csv(self.cwd + '/Predictions/submission_format_' + self.date + '.csv')

        # set up model prediction function

    def Predict(self, Region, SCA=True):
        """
            Run model inference on a region

            Parameters:
                Region (str): The region to run inference on
                SCA (bool): Whether or not to use SCA data

            Returns:
                Forcast[Region] (DataFrame): The forecast df for the region
        """
        # region specific features
        features = self.Region_optfeatures[Region]

        # Make prediction dataframe
        forecast_data = self.Forecast[Region].copy()

        if SCA:
            # drop all rows that have a False value in "hasSnow", i.e. no snow, so skip inference
            inference_locations = forecast_data.drop(forecast_data[~forecast_data["hasSnow"]].index)
        else:
            # keep all rows
            inference_locations = forecast_data

        forecast_data = inference_locations[features]  # Keep only features needed for inference

        if len(inference_locations) == 0:  # makes sure that we don't run inference on empty regions
            print("No snow in region: ", Region)
            self.Forecast[Region][self.date] = 0.0
        else:
            # change all na values to prevent scaling issues
            forecast_data[forecast_data < -9000] = -10

            # load and scale data

            # set up model checkpoint to be able to extract best models
            checkpoint_filepath = self.cwd + '/Model/Prev_SWE_Models_Final/' + Region + '/'
            model = checkpoint_filepath + Region + '_model.h5'
            print(model)
            model = load_model(model)

            # load SWE scaler
            SWEmax = np.load(checkpoint_filepath + Region + '_SWEmax.npy')
            SWEmax = SWEmax.item()

            # load features scaler
            # save scaler data here too
            scaler = pickle.load(open(checkpoint_filepath + Region + '_scaler.pkl', 'rb'))
            scaled = scaler.transform(forecast_data)
            x_forecast = pd.DataFrame(scaled, columns=forecast_data.columns)

            # make predictions and rescale
            y_forecast = (model.predict(x_forecast))
            y_forecast[y_forecast < 0] = 0
            y_forecast = (SWEmax * y_forecast)
            # remove forecasts less than 0.5 inches SWE
            y_forecast[y_forecast < 0.5] = 0  # TODO address this with research, try smaller values/no value

            # add predictions to forecast dataframe

            self.Forecast[Region][self.date] = 0.0  # initialize column
            forecast_data[self.date] = y_forecast  # add column
            self.Forecast[Region][self.date].update(forecast_data[self.date])  # update forecast dataframe

        return self.Forecast[Region]


def calculateGranuleExtent(boundingBox: list[float, float, float, float],
                           day: Union[datetime, str] = datetime(2018, 7, 7)):
    """
        Fetches relevant VIIRS granules from NASA's EarthData's CMR API.

        Parameters:
            boundingBox (list[float, float, float, float]): The bounding box of the region of interest.

                lower_left_lon – lower left longitude of the box (west)
                lower_left_lat – lower left latitude of the box (south)
                upper_right_lon – upper right longitude of the box (east)
                upper_right_lat – upper right latitude of the box (north)

            day (datetime, str): The day to query granules for.

        Returns:
            cells (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries

    """

    if not isinstance(day, datetime):
        day = datetime.strptime(day, "%Y-%m-%d")

    # Get params situated
    datasetName = "VNP10A1F"  # NPP-SUOMI VIIRS, but JPSS1 VIIRS also exists
    version = "2" if day > datetime(2018, 1, 1) else "1"  # TODO v1 supports 2013-on, but v2 currently breaks <2018???

    query = (ea.granule_query()
             .short_name(datasetName)
             .version(version)
             .bounding_box(*boundingBox)
             .temporal(day.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d"))
             # Grab one day's worth of data, we only care about spatial extent
             )

    results = query.get(100)  # The Western CONUS is usually 7, so this is plenty

    cells = []
    for result in results:
        geometry = shapely.geometry.Polygon(
            [(x["Longitude"], x["Latitude"]) for x in
             result["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]["Geometry"]["GPolygons"][0]["Boundary"][
                 "Points"]]
        )
        cell = {
            "h": result["umm"]["AdditionalAttributes"][1]["Values"][0],  # HORIZONTAL TILE NUMBER
            "v": result["umm"]["AdditionalAttributes"][2]["Values"][0],  # VERTICAL TILE NUMBER
            "geometry": geometry
        }
        cells.append(cell)

    geo = gpd.GeoDataFrame(cells, geometry="geometry", crs="EPSG:4326")
    return geo


def createGranuleGlobpath(dataRoot: str, date: datetime, h: int, v: int) -> str:
    """
        Creates a filepath for a VIIRS granule.

        Parameters:
            dataRoot (str): The root folder for the data.
            date (str): The date of the data.
            h (int): The horizontal tile number.
            v (int): The vertical tile number.

        Returns:
            filepath (str): The filepath of the granule.
    """
    dayOfYear = date.strftime("%Y%j")  # Format date as YearDayOfYear

    WY_split = datetime(date.year, 10, 1)  # Split water years on October 1st

    # if day is after water year, then we need to adjust the year
    if date.month < 10:
        year = date.year - 1
        next_year = date.year
    else:
        year = date.year
        next_year = date.year + 1

    return str(Path(dataRoot, f"{year}-{next_year}NASA", f"VNP10A1F_A{dayOfYear}_h{h}v{v}_*.tif"))


def granuleFilepath(filepath: str) -> str:
    """
        return matched filepath if it exists, otherwise return empty string
    """
    result = glob.glob(filepath)
    if result:
        return result[0]  # There should only be one match
    else:
        return ''


def fetchGranules(boundingBox: list[float, float, float, float],
                  dataFolder: Union[Path, str],
                  date: Union[datetime, str],
                  extentDF: gpd.GeoDataFrame = None,
                  shouldDownload: bool = True) -> gpd.GeoDataFrame:
    """
            Fetches VIIRS granules from local storage.

            Parameters:
                boundingBox (list[float, float, float, float]): The bounding box of the region of interest. (west, south, east, north)
                date (datetime, str): The start date of the data to fetch.
                dataFolder (Path, str): The folder to save the data to, also used to check for existing data.
                extentDF (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries
                shouldDownload (bool): Whether to fetch the data from the API or not.

            Returns:
                df (GeoDataFrame): A dataframe of the granules that intersect with the bounding box
        """

    if extentDF is None:
        cells = calculateGranuleExtent(boundingBox, date)  # Fetch granules from API, no need to check bounding box
    else:
        # Find granules that intersect with the bounding box
        cells = extentDF.cx[boundingBox[0]:boundingBox[2],
                boundingBox[1]:boundingBox[3]]  # FIXME if there is only one point, this will fail

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if not isinstance(dataFolder, Path):
        dataFolder = Path(dataFolder)

    day = date.strftime("%Y-%m-%d")
    cells["date"] = date  # record the date
    cells["filepath"] = cells.apply(
        lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
        axis=1
    )  # add filepath if it exists, otherwise add empty string

    missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")
    attempts = 3  # how many times it will try and download the missing granules
    while shouldDownload and len(missingCells) > 0 and attempts > 0:
        # TODO test function that fetches missing granules from NASA
        print(f"Missing {len(missingCells)} granules for {day}, downloading")
        temporal = format_date(date)  # Format date as YYYY-MM-DD
        bbox = format_boundingbox(boundingBox)  # Format bounding box as "W,S,E,N"
        version = "2" if date > datetime(2018, 1, 1) else "1"  # Use version 1 if date is before 2018
        year = date.year if date.month >= 10 else date.year - 1  # Water years start on October 1st
        download("VNP10A1F", version, temporal, bbox, dataFolder.joinpath(f"{year}-{year + 1}NASA"), mode="async")
        cells["filepath"] = cells.apply(
            lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
            axis=1
        )  # add filepath if it exists, otherwise add empty string
        missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")
        if len(missingCells) > 0:
            attempts -= 1
            print(f"still missing {len(missingCells)} granules for {day}, retrying in 30 seconds, {attempts} tries left")
            time.sleep(30)
            print("retrying")
            cells["filepath"] = cells.apply(
                lambda x: granuleFilepath(createGranuleGlobpath(dataFolder, date, x['h'], x['v'])),
                axis=1
            )  # add filepath if it exists, otherwise add empty string
            missingCells = cells[cells["filepath"] == ''][["h", "v"]].to_dict("records")  # before we try again, double check



    return cells


def fetchGranulesRange(boundingBox: list[float, float, float, float],
                       dataFolder: str,
                       startDate: Union[datetime, str],
                       endDate: Union[datetime, str] = None,
                       extentDF: gpd.GeoDataFrame = None,
                       frequency: str = "D",
                       fetch: bool = True) -> dict:
    """
        Fetches VIIRS granules from local storage.

        Parameters:
            boundingBox (list[float, float, float, float]): The bounding box of the region of interest. (west, south, east, north)
            startDate (str): The start date of the data to fetch.
            endDate (str): The end date of the data to fetch. Defaults to same day as startDate.
            dataFolder (str): The folder to save the data to, also used to check for existing data.
            extentDF (GeoDataFrame): A dataframe containing the horizontal and vertical tile numbers and their boundaries

        Returns:
            dfs (dict): A dictionary of dataframes the granules that intersect with the bounding box by day
    """
    if type(startDate) != datetime:
        startDate = datetime.strptime(startDate, "%Y-%m-%d")  # If start date is specified, convert to datetime

    if endDate is None:
        endDate = startDate  # If no end date is specified, assume we only want one day
    elif type(endDate) != datetime:
        endDate = datetime.strptime(endDate, "%Y-%m-%d")  # If end date is specified, convert to datetime

    if extentDF is None:
        cells = calculateGranuleExtent(boundingBox, startDate)  # Fetch granules from API, no need to check bounding box
    else:
        # Find granules that intersect with the bounding box
        cells = extentDF.cx[boundingBox[0]:boundingBox[2], boundingBox[1]:boundingBox[3]]

    missing = {}
    dfs = {}
    # check and see if we already have the data for that day
    for date in pd.date_range(startDate, endDate, freq=frequency):
        # for each granule, check if we have the data
        granules = fetchGranules(boundingBox, dataFolder, date, cells, shouldDownload=False)

        missingCells = granules[granules["filepath"] == ''][["h", "v"]].to_dict("records")
        if len(missingCells) > 0:
            missing[date.strftime("%Y-%m-%d")] = missingCells
        dfs[date.strftime("%Y-%m-%d")] = granules

    if fetch and len(missing) > 0:
        print(f"Missing data for the following days: {list(missing.keys())}")

        # create strings of days to request in batches
        for datestr in missing:
            dateObj = datetime.strptime(datestr, "%Y-%m-%d")

            # TODO make list of days that are consecutive


    return dfs  # TODO theres probably a better way to store this, but you can send this to .h5 for storage


def createMergedRxr(files: list[str]) -> xr.DataArray:
    """
        Creates a merged (mosaic-ed) rasterio dataset from a list of files.

        Parameters:
            files (list[str]): A list of filepaths to open and merge.

        Returns:
            merged (DataArray): A merged DataArray.
    """

    # FIXME sometimes throws "CPLE_AppDefined The definition of geographic CRS EPSG:4035 got from GeoTIFF keys is not
    #   the same as the one from the EPSG registry, which may cause issues during reprojection operations. Set
    #   GTIFF_SRS_SOURCE configuration option to EPSG to use official parameters (overriding the ones from GeoTIFF
    #   keys), or to GEOKEYS to use custom values from GeoTIFF keys and drop the EPSG code."
    tifs = [rxr.open_rasterio(file) for file in files]  # Open all the files as Rioxarray DataArrays

    noLakes = [tif.where(tif != 237, other=0) for tif in tifs]  # replace all the lake values with 0
    noOceans = [tif.where(tif != 239, other=0) for tif in noLakes]  # replace all the ocean values with 0
    noErrors = [tif.where(tif <= 100, other=100) for tif in
                noOceans]  # replace all the other values with 100 (max Snow)
    return merge_arrays(noErrors, nodata=255)  # Merge the arrays


def augmentGeoDF(gdf: gpd.GeoDataFrame,
                 raster: xr.DataArray,
                 threshold: float = 20,  # TODO try 10
                 noData: int = 255,
                 buffer: float = None) -> gpd.GeoDataFrame:
    """
        Augments a GeoDataFrame with a raster's values.

        Parameters:
            gdf (GeoDataFrame): The GeoDataFrame to append the SCA to. Requires geometry to be an area, see buffer param
            raster (DataArray): The raster to augment the GeoDataFrame with.
            threshold (int): The threshold to use to determine if a pixel is snow or not.
            noData (int): The no data value of the raster.
            buffer (float): The buffer to use around the geometry. Set if the geometry is a point.

        Returns:
            gdf (GeoDataFrame): The augmented GeoDataFrame.
    """

    if buffer is not None:
        buffered = gdf.to_crs("3857").buffer(buffer,
                                             cap_style=3)  # Convert CRS to a projected CRS and buffer points into squares
        buffered = buffered.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster
    else:
        buffered = gdf.to_crs(raster.rio.crs)  # Convert the GeoDataFrame to the same CRS as the raster

    stats = rs.zonal_stats(buffered,  # pass the buffered geometry
                           raster.values[0],  # pass the raster values as a numpy array, TODO investigate passing GDAL
                           no_data=noData,
                           affine=raster.rio.transform(),  # required for passing numpy arrays
                           stats=['mean'],  # we only want the mean, others are available if needed
                           geojson_out=False,  # we will add the result back into a GeoDataFrame, so no need for GeoJSON
                           )

    gdf["VIIRS_SCA"] = [stat['mean'] for stat in stats]  # add the mean to the GeoDataFrame
    gdf["hasSnow"] = gdf["VIIRS_SCA"] > threshold  # snow value is above 20%

    return gdf


if __name__ == '__main__':
    NSM = NSM_SCA("/Users/jmac/Documents/Programming/REU/National-Snow-Model", "2019-04-08")

    NSM.Get_Monitoring_Data_Threaded()

    bbox = NSM.getPredictionExtent()

    NSM.initializeGranules(bbox, NSM.SCA_folder)

    NSM.Data_Processing()

    NSM.augmentPredictionDFs()

    NSM.SWE_Predict()
    print("done")
