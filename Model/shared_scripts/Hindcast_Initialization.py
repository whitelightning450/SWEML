#created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
# SWEET supported by the University of Alabama and the Alabama Water Institute
# 10-19-2023
import os
import pandas as pd
import warnings
import pickle
from datetime import date, datetime, timedelta
import geopandas as gpd
from tqdm import tqdm
import contextily as cx #contextily-1.4.0 mercantile-1.2.1
import matplotlib.pyplot as plt
import glob
import contextlib
from PIL import Image
from IPython.display import Image as ImageShow
import boto3
from progressbar import ProgressBar
from botocore import UNSIGNED
from botocore.client import Config
import os
warnings.filterwarnings("ignore")

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
BUCKET_NAME = 'national-snow-model'
BUCKET = S3.Bucket(BUCKET_NAME)

#Function for initializing a hindcast
def Hindcast_Initialization(cwd, datapath, new_year, threshold, Region_list, frequency): 
    print('Creating files for a historical simulation within ', str(Region_list)[1:-1], ' regions for water year ', new_year)
    #set day_delta
    if frequency == 'Weekly':
        day_delta = 7
    if frequency == 'Daily':
        day_delta = 1

    PreProcessedpath = "PaperData/Data/PreProcessed/"
    HoldOutpath = f"PaperData/Data/Predictions/Hold_Out_Year/{frequency}"
    
    #Grab existing files based on water year
    prev_year = '2022'
    prev_date = prev_year + '-09-24'

    #input the new water year of choice
    new_year = str(int(new_year)-1)
    new_date = new_year + '-09-25'

    #threshold
    threshold = threshold

    #Make a submission DF
    try:
        old_preds = pd.read_csv(f"{datapath}/data/PreProcessed/submission_format_2022-09-24.parquet")
    except:
        #Make a submission DF
        key = f"{PreProcessedpath}submission_format_2022-09-24.parquet"
        obj = BUCKET.Object(key)
        body = obj.get()['Body']
        old_preds = pd.read_csv(body)
    
    old_preds['2022-10-01'] = 0
    new_preds_date = new_year+'-10-02'
    old_preds.rename(columns = {'2022-10-02':new_preds_date}, inplace = True)
    old_preds.set_index('cell_id', inplace = True)
    old_preds.to_hdf(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/submission_format.h5", key =f"{new_date}")
    
    
    #define start and end date for list of dates
    start_dt = date(int(new_year), 10, 2)
    end_dt = date(int(new_year)+1, 6, 26)
    
    #create empty list to store dates
    datelist = []

    #append dates to list
    for dt in daterange(start_dt, end_dt, day_delta):
        #print(dt.strftime("%Y-%m-%d"))
        dt=dt.strftime('%Y-%m-%d')
        datelist.append(dt)
        
     #makes sure all prediction locations for testing are included in the simulation   
    #addPredictionLocations(Region_list, datapath, cwd, datelist[0])
    
    print('New simulation start files complete')
    return datelist
    

    
#can be altered to create list every n number of days by changing 7 to desired skip length
def daterange(start_date, end_date, day_delta):
     for n in range(0, int((end_date - start_date).days) + 1, day_delta):
        yield start_date + timedelta(n)
        
        
        
        
def HindCast_DataProcess(datelist,Region_list, cwd, datapath, model, frequency):
    if frequency == 'Weekly':
        day_delta = 7
    if frequency == 'Daily':
        day_delta = 1
       #get held out year observational data
    Test = pd.DataFrame()
    cols = ['Date','y_test','Long', 'Lat', 'elevation_m', 'WYWeek', 'northness', 'VIIRS_SCA', 'hasSnow', 'Region']
    for Region in Region_list:
        T= pd.read_hdf(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/RegionWYTest.h5", Region)
        T['Region'] = Region
        #T['y_pred'] = -9999
        T.rename(columns = {'SWE':'y_test'}, inplace = True)
        T = T[cols]
        Test = pd.concat([Test, T])

        #Load predictions into a DF
    preds = pd.DataFrame()
    prev_SWE = pd.DataFrame()
    pred_sites = pd.DataFrame()
    TestsiteData = pd.DataFrame()
    TestsiteDataPSWE = pd.DataFrame()
    print('Getting prediction files')
    for date in datelist:
        try:
            preds[date] = pd.read_hdf(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/2019_predictions.h5", key = date)
        except:
            print('No prediction file found, retrieving from AWS')
            S3.meta.client.download_file(BUCKET_NAME, f"{model}/Hold_Out_Year/2019_predictions.h5",f"{cwd}/Predictions/Hold_Out_Year/{frequency}/2019_predictions.h5")
            preds[date] = pd.read_hdf(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/2019_predictions.h5", key = date)
            print('File loaded, proceeding...')
        
        #get previous SWE predictions for DF
        startdate = str(datetime.strptime(date, '%Y-%m-%d').date() -timedelta(day_delta))
        if startdate < f"{startdate[:4]}-10-01":
            prev_SWE[startdate] = preds[date]
            prev_SWE[startdate] = 0
            
        else:
            prev_SWE[startdate] = pd.read_hdf(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/2019_predictions.h5", key = startdate)
        
        
        Tdata = Test[Test['Date'] == date]
        TestsiteData = pd.concat([TestsiteData, Tdata])
        
        
        try:
            prev_Tdata = Test[Test['Date'] == startdate]
            prev_Tdata['Date'] = date
            
        except:
            print('No previous observations for ', date)
        #add previous obs to determine prev_SWE error
        TestsiteDataPSWE = pd.concat([TestsiteDataPSWE, prev_Tdata])
        
        sites = Test[Test['Date'] == date].index
        #print(date, len(sites))
        #print(sites)
        
        for site in sites:
            #predictions
            s = pd.DataFrame(preds.loc[site].copy()).T
            s['Date'] = date
            s.rename(columns = {date:'y_pred'}, inplace = True)
            cols =['Date', 'y_pred']
            s = s[cols]
            
            #previous SWE
            pSWE = pd.DataFrame(preds.loc[site].copy()).T
            pSWE['Date'] = date
            pSWE.rename(columns = {startdate:'prev_SWE'}, inplace = True)
            cols =['Date', 'prev_SWE']
            pSWE = pSWE[cols]
            
            #print(s)
            s['prev_SWE'] = pSWE['prev_SWE']
            pred_sites = pd.concat([pred_sites, s])

    print('Site data processing complete, setting up prediction dataframes...')  
    pswecols = ['Date', 'y_test']
    TestsiteDataPSWE = TestsiteDataPSWE[pswecols]
    TestsiteDataPSWE.rename(columns = {'y_test':'y_test_prev'}, inplace = True)
    #display(TestsiteDataPSWE)       
    
    #get predictions for obs locations
    cols = ['Date','y_test', 'y_test_prev', 'y_pred','prev_SWE','Long', 'Lat', 'elevation_m', 'WYWeek', 'northness', 'VIIRS_SCA', 'hasSnow', 'Region']
    #display(TestsiteData)
    TestsiteData = pd.concat([TestsiteData, pred_sites], axis =1)
    
    TestsiteData = TestsiteData.loc[:,~TestsiteData.columns.duplicated()].copy()
    TestsiteDataPSWE.reset_index(inplace = True)
    TestsiteData.reset_index(inplace = True)
    TestsiteData['Date'] = TestsiteData['Date'].dt.strftime('%Y-%m-%d')
    TestsiteData = pd.merge(TestsiteData, TestsiteDataPSWE,  how='left', left_on=['index','Date'], right_on = ['index','Date'])
    TestsiteData.set_index('index', inplace = True)
    TestsiteData.fillna(0, inplace = True)
    TestsiteData = TestsiteData[cols]
    TestsiteData['prev_SWE_error'] = TestsiteData['y_test_prev'] - TestsiteData['prev_SWE']
    #Set up dictionary to match the training data
    EvalTest = {}
    print('Finalizing Evaluation dataframes...')
    for Region in Region_list:
        EvalTest[Region] = TestsiteData[TestsiteData['Region'] == Region]
        EvalTest[Region]['y_pred'] = EvalTest[Region]['y_pred']*2.54
        EvalTest[Region]['y_pred_fSCA'] = EvalTest[Region]['y_pred']
        EvalTest[Region]['y_test'] = EvalTest[Region]['y_test']*2.54
        EvalTest[Region]['y_test_prev'] = EvalTest[Region]['y_test_prev']*2.54
        EvalTest[Region]['prev_SWE'] = EvalTest[Region]['prev_SWE']*2.54
        
    return  EvalTest
        

#function to add prediction locations in training dataset but not in the submission format file        
def addPredictionLocations(Region_list, datapath, cwd, startdate, frequency):
    if frequency == 'Weekly':
        day_delta = 7
    if frequency == 'Daily':
        day_delta = 1
    print('Making sure all testing locations are in prediction dataframe.')
    #get held out year observational data
    Test = pd.DataFrame()
    cols = ['Date','y_test', 'y_pred','Long', 'Lat', 'elevation_m', 'WYWeek', 'northness', 'VIIRS_SCA', 'hasSnow', 'Region']
    for Region in Region_list:
        T= pd.read_hdf(f"{datapath}/data/RegionWYTest.h5", Region)
        T['Region'] = Region
        T['y_pred'] = -9999
        T.rename(columns = {'SWE':'y_test'}, inplace = True)
        T = T[cols]
        Test = pd.concat([Test, T])


    res = []
    [res.append(x) for x in list(Test.index) if x not in res]

    rows_not_pred = []
    for row in res:
        try:
            preds.loc[row]
        except:
            rows_not_pred.append(row)

    regions = pd.read_pickle(f"{datapath}/data/PreProcessed/RegionVal.pkl")
    regionval = pd.DataFrame()
    Region_list2 = ['N_Sierras', 'S_Sierras']
    for region in Region_list2:
        regionval = pd.concat([regionval, regions[region]])
    regionval.set_index('cell_id', inplace = True)

    #get rows in Test that are not in regionVal
    Test2Rval = Test.copy()
    cols = ['Long','Lat','elevation_m', 'northness']
    Test2Rval = Test2Rval[cols]
    Test2Rval.drop_duplicates(inplace = True)
    Test2Rval = Test2Rval.loc[rows_not_pred]

    #add to regionval
    regionval = pd.concat([regionval, Test2Rval])
    regionval['Region'] = 'none'
    regionval.reset_index(inplace=True)
    #need to put back into respective regions and Region
    regionval = Region_id(regionval)
    regionval.rename(columns = {'index':'cell_id'}, inplace = True)
    regionval.set_index('cell_id', inplace = True)
    regionDict = {}
    for Region in Region_list2:
        regionDict[Region] = regionval[regionval['Region'] ==Region]
        regionDict[Region].pop('Region')
        regionDict[Region].reset_index(inplace = True)
     
    #set elevation based S_Sierras REgions
    regionDict['S_Sierras_Low'] = regionDict['S_Sierras'][regionDict['S_Sierras']['elevation_m'] <= 2500]
    regionDict['S_Sierras_High'] = regionDict['S_Sierras'][regionDict['S_Sierras']['elevation_m'] > 2500]

    # write the python object (dict) to pickle file
    path = f"{datapath}/data/PreProcessed/RegionVal2.pkl"
    RVal = open(path, "wb")
    pickle.dump(regionDict, RVal)
    
    #Fix Predictions start DF
    startdate = datetime.strptime(startdate, '%Y-%m-%d').date() -timedelta(day_delta)
    prev_SWE = {}
    for region in Region_list:
        prev_SWE[region] = pd.read_hdf(f"{cwd}/Predictions/Hold_Out_Year/Predictions/predictions{startdate}.h5", key =  region)                       

        pSWEcols = list(prev_SWE[region].columns) 
        rValcols = list(regionDict[region].columns)
        rValcols.remove('cell_id')
        res = [i for i in pSWEcols if i not in rValcols]
        for i in res:
            regionDict[region][i]=0
            
        #Final DF prep    
        regionDict[region]['WYWeek'] = 52  
        regionDict[region].set_index('cell_id', inplace = True)
        #save dictionary   
        regionDict[region].to_hdf(f"{cwd}/Predictions/Hold_Out_Year/Predictions/predictions{startdate}.h5", key = region)
                                              

           # return regionDict
    
    
def Region_id(df):

    # put obervations into the regions
    for i in range(0, len(df)):

        # Sierras
        # Northern Sierras
        if -122.5 <= df['Long'][i] <= -119 and 39 <= df['Lat'][i] <= 42:
            loc = 'N_Sierras'
            df['Region'].iloc[i] = loc

        # Southern Sierras
        if -122.5 <= df['Long'][i] <= -117 and 35 <= df['Lat'][i] <= 39:
            loc = 'S_Sierras'
            df['Region'].iloc[i] = loc
    return df

#function for making a gif/timelapse of the hindcast
def Snowgif(cwd, datelist, Region_list,frequency):
    
    #Load prediction file with geospatial information
    path = f"{cwd}/Predictions/Hold_Out_Year/{frequency}/Prediction_DF_SCA_2018-10-02.pkl"
    geofile =open(path, "rb")
    geofile = pickle.load(geofile)
    cols = ['Long', 'Lat']

    #convet to one dataframe
    geo_df = pd.DataFrame()
    for Region in Region_list:
        geofile[Region] = geofile[Region][cols]
        geo_df = pd.concat([geo_df, geofile[Region]])

    #convert to geodataframe
    geo_df = gpd.GeoDataFrame(
        geo_df, geometry=gpd.points_from_xy(geo_df.Long, geo_df.Lat), crs="EPSG:4326"
    )

    path = f"{cwd}/Predictions/Hold_Out_Year/{frequency}/2019_predictions.h5"
    #get predictions for each timestep
    print('processing predictions into geodataframe')
    for date in tqdm(datelist):
        pred = pd.read_hdf(path, key = date)
        geo_df[date] = pred[date]*2.54 #convert to cm


    #convert to correct crs.
    geo_df = geo_df.to_crs(epsg=3857)

    print('creating figures for each prediction timestep') #This could be threaded/multiprocessed to speed up
    for date in tqdm(datelist):
    #date = "2019-03-26"
        cols = ['geometry', date]
        plotdf = geo_df[cols]
        fig, ax = plt.subplots(figsize=(10, 10))
        #plot only locations with SWE
        plotdf = plotdf[plotdf[date] > 0]
        ax = plotdf.plot(date, 
                     #figsize=(10, 10), 
                     alpha=0.5, 
                     markersize = 10,
                     edgecolor="k", 
                     vmin =1, 
                     vmax =250,
                    legend = True,
                    legend_kwds={"label": "Snow Water Equivalent (cm)", "orientation": "vertical"},
                    ax = ax)
        ax.set_xlim(-1.365e7, -1.31e7)
        ax.set_ylim(4.25e6, 5.25e6)
        cx.add_basemap(ax)
        ax.set_axis_off()
        ax.text(-1.35e7, 5.17e6, f"SWE estimate: {date}", fontsize =14)
        #plt.title(f"SWE estimate: {date}")
        plt.savefig(f"{cwd}/Predictions/Hold_Out_Year/{frequency}/Figures/SWE_{date}.PNG")
        plt.close(fig)
            
    # filepaths
    print('Figures complete, creating .gif image')
    fp_in =f"{cwd}/Predictions/Hold_Out_Year/{frequency}/Figures/SWE_*.PNG"
    fp_out = f"{cwd}/Predictions/Hold_Out_Year/{frequency}/Figures/SWE_2019_predictions.gif"

    # use exit stack to automatically close opened images
    with contextlib.ExitStack() as stack:

        # lazily load images
        imgs = (stack.enter_context(Image.open(f))
                for f in sorted(glob.glob(fp_in)))

        # extract  first image from iterator
        img = next(imgs)

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=200, loop=0)
    #Display gif    
    return ImageShow(fp_out)


def Hindcast_to_AWS(modelname, folderpath, AWSpath, type):
 
    files = []
    for file in os.listdir(folderpath):
        if file.endswith(type):
            files.append(file)
    print(files)

    #Load and push to AWS
    pbar = ProgressBar()
    print('Pushing files to AWS')
    for file in pbar(files):
        filepath = f"{folderpath}{file}"
        S3.meta.client.upload_file(Filename= filepath, Bucket=BUCKET_NAME, Key=f"{modelname}/{AWSpath}{file}")

def AWS_to_Hindcast(modelname, frequency):
    #load access key
    
    files = []
    for objects in BUCKET.objects.filter(Prefix=f"{modelname}/Hold_Out_Year/{frequency}/"):
        files.append(objects.key)
    if f"{modelname}/Hold_Out_Year/{frequency}/" in files:
        files.remove(f"{modelname}/Hold_Out_Year/{frequency}/")
    print('Downloading files from AWS to local')
    for file in tqdm(files):
        filename = file.replace(f"{modelname}/Hold_Out_Year/{frequency}/", '')
        filename = f"{HOME}/SWEML/Model/{modelname}/Predictions/Hold_Out_Year/{frequency}/{filename}"
        #print(filename)
        S3.meta.client.download_file(BUCKET_NAME, file, filename)