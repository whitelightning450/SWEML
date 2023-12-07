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



# Create .py function to process data to train model. 

def DataProcess(test_year, modelname, Region_list):
    #regions are N_Sierra, S_Sierra_Low, S_Sierra_High
    
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

    #store and load region list
    #Region_list = ['N_Sierras', 'S_Sierras_Low', 'S_Sierras_High']
    #content_object = s3.Object(bucket_name, 'data/Regions.txt')
    #file_content = content_object.get()['Body'].read().decode('utf-8')
    #Region_list = json.loads(file_content)
    

    #set data path
    #training_path = f"{datapath}/data"
    #RegionTrain = {}
    print('Processing training dataframes for each region')
    #for region in tqdm(Region_list):   
        #load the RegionTrain DF, select the key grid cell colums, and add to training DF.
     #   RegionTrain[region] = pd.read_hdf(f"{training_path}/RegionTrain_SCA.h5", region)
    #RegionTrain= pickle.load(open(f"{training_path}/RegionTrain_SCA.pkl", "rb"))

    #load RFE optimized features
    #Region_optfeatures= pickle.load(open(f"{training_path}/Optimal_Features.pkl", "rb"))
    #Load Training Data
    #file_key = 'data/RegionTrain_SCA.pkl'
    #obj = bucket.Object(file_key)
    #body = obj.get()['Body']
    #RegionTrain = pd.read_pickle(body)
    RegionTrain = {}
    for Region in Region_list:
        RegionTrain[Region] = pd.read_hdf(f"{home}/NSM/Snow-Extrapolation/data/VIIRS/RegionTrain_SCA.h5", key = Region)
        

    #load RFE optimized features
    file_key = 'data/Optimal_Features.pkl'
    obj = bucket.Object(file_key)
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

        #get optimal features for each regions (from LGBM RFE), first pop off fSCA
        optfeatures = list(Region_optfeatures[Region])

        #make a df copy of specific region
        df = RegionTrain.get(Region).copy()
        dfVIIRS = df[VIIRS_cols].copy()
        dfVIIRS.reset_index(inplace = True)
        df = df[optfeatures]

        ### replace special character ':' with '__' 
        df = df.rename(columns = lambda x:re.sub(':', '__', x))

        #change all na values to prevent scaling issues
        df[df< -9000]= -10    
        df_notscaled = df.copy()
        #normalize training data    
        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        #save scaler data here too
        scaled = scaler.fit_transform(df)
        dump(scaler, open(f"./Model/{Region}/{Region}_scaler.pkl", 'wb'))
        
        df = pd.DataFrame(scaled, columns = df.columns)

        #Add Viirs colums
        df = pd.concat([df, dfVIIRS], axis = 1)
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
        RegionWYTest.to_hdf("./Predictions/Hold_Out_Year/RegionWYTest.h5", Region)
        SWEmax = np.array(SWEmax)
        np.save(f"./Model/{Region}/{Region}_SWEmax.npy" , SWEmax)
        
        
    #save RegionTest
    #pickle_byte_obj = pickle.dumps(RegionWYTest) 
    #s3.Object(bucket_name,f"data/RegionWYTest.pkl").put(Body=pickle_byte_obj)  
    
    return RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, RegionTest_notScaled


# In[ ]:




