#Standardized Snow Water Equivalent Evaluation Tool (SWEET)#created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
# SWEET supported by the University of Alabama and the Alabama Water Institute
# 10-19-2023

import os
from os import listdir
from os.path import isfile, join
import time
import re
import copy
import numpy as np
import pandas as pd
import h5py
import tables
import random
from tqdm import tqdm
import math
import pickle 
from pickle import dump
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings; warnings.filterwarnings("ignore")
import boto3
from botocore import UNSIGNED
from botocore.client import Config
import os

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



# Create .py function to process data to train model. 

def DataProcess(test_year, frequency, modelname, Region_list, fSCA = False):

    print('Processing training dataframes for each region')

    RegionTrain = {}
    for Region in Region_list:
        try:
            RegionTrain[Region] = pd.read_hdf(f"{HOME}/SWEML/data/VIIRS/RegionTrain_SCA.h5", key = Region)
        except:
            S3.meta.client.download_file(BUCKET_NAME, f"data/VIIRS/RegionTrain_SCA.h5",f"{HOME}/SWEML/data/VIIRS/RegionTrain_SCA.h5")
            RegionTrain[Region] = pd.read_hdf(f"{HOME}/SWEML/data/VIIRS/RegionTrain_SCA.h5", key = Region)

    #load RFE optimized features 
    file_key = 'data/Optimal_Features.pkl'
    obj = BUCKET.Object(file_key)
    body = obj.get()['Body']
    Region_optfeatures = pd.read_pickle(body)

    #Split the data the same as original train/test split to make same figures/analysis
    VIIRS_cols = ['Date', 'VIIRS_SCA', 'hasSnow']

    RegionObs_Train = {}
    RegionObs_Test = {}
    RegionTest = {}
    RegionTest_notScaled = {}
    TestingYR = test_year

    #correct testing year for WY
    WY = str(TestingYR-1)

    for Region in Region_list:
        print(Region)

        #Pull a specific Water Year out of the Training Data
        RegionWYTest = RegionTrain[Region][(RegionTrain[Region]['Date'] >= f"10-01-{WY}") & (RegionTrain[Region]['Date'] < f"10-01-{str(TestingYR)}")].copy()
        t_low = RegionTrain[Region][RegionTrain[Region]['Date'] < f"10-01-{WY}"]
        t_high = RegionTrain[Region][RegionTrain[Region]['Date'] >= f"10-01-{str(TestingYR)}"]
        RegionTrain[Region] = pd.concat([t_low, t_high])
        
        #get y data
        y = RegionTrain[Region]['SWE']

        #get max SWE for normalization and prediction
        SWEmax = max(RegionTrain[Region]['SWE'])
        y = y/SWEmax
        
        #make a df copy of specific region
        df = RegionTrain.get(Region).copy()
    

        #get optimal features for each regions (from LGBM RFE), first pop off fSCA
        optfeatures = list(Region_optfeatures[Region])
        if fSCA == True:
            VIIRS_cols.append('HasSnow')
            optfeatures.append('VIIRS_SCA')
            optfeatures.append('HasSnow')
            df['HasSnow'] = 0
            df['HasSnow'][df['hasSnow']==True] = 1
            #df.pop('hasSnow')
            


        dfVIIRS = df[VIIRS_cols].copy()
        dfVIIRS.reset_index(inplace = True)
        
        df = df[optfeatures]

        ### replace special character ':' with '__' 
        #df = df.rename(columns = lambda x:re.sub(':', '__', x))

        #change all na values to prevent scaling issues
        df[df< -9000]= -10    
        df_notscaled = df.copy()
        #normalize training data    
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        #save scaler data here too
        scaled = scaler.fit_transform(df)
        checkpoint_filepath = f"./Model/{Region}/fSCA_{fSCA}/"
        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)
        dump(scaler, open(f"{checkpoint_filepath}{Region}_scaler.pkl", 'wb'))
        
        df = pd.DataFrame(scaled, columns = df.columns)

        #Add Viirs colums
        df = pd.concat([df, dfVIIRS], axis = 1)
        df = df.loc[:,~df.columns.duplicated()].copy()
        df.set_index('index', inplace = True, drop = True)

        #Set the 75/25% train/test split, set a random state to get train/test for future analysis
        X = df
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1234)

        #get non transformed data to support analysis
        X_notscaled = df_notscaled
        X_train_notscaled, X_test_notscaled, y_train_notscaled, y_test_notscaled = train_test_split(X_notscaled, y, test_size=0.25, random_state=1234)

        #Set the RegionTrain and RegionOBS dicts
        RegionTrain[Region] = X_train
        RegionTest[Region] = X_test
        RegionTest_notScaled[Region] = X_test_notscaled
        RegionObs_Train[Region] = pd.DataFrame(y_train, columns = ['SWE'])
        RegionObs_Test[Region] = pd.DataFrame(y_test, columns = ['SWE'])
        RegionWYTest.to_hdf(f"./Predictions/Hold_Out_Year/{frequency}/RegionWYTest.h5", Region)
        SWEmax = np.array(SWEmax)
        print(f"{checkpoint_filepath}{Region}_SWEmax.npy")
        np.save(f"{checkpoint_filepath}{Region}_SWEmax.npy" , SWEmax)
    
    
    return RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, RegionTest_notScaled


