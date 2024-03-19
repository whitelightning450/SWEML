#created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
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
import matplotlib.pyplot as plt
import seaborn as sns
import math
import pickle 
import sklearn
from pickle import dump
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from progressbar import ProgressBar
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
import threading  # this is the threading library
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

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def Model_train(epochs,batchsize, RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, Region_list, fSCA = False):
    
    
    #Get regions
    #Regions = list(RegionTrain.keys())
    
    for Region in Region_list:
        print('Training model for: ', Region)
        #set up train/test dfs
        X_train = RegionTrain[Region].copy()
        X_test = RegionTest[Region].copy()
        y_train = RegionObs_Train[Region].copy()
        y_test = RegionObs_Test[Region].copy()
        
        X_train.pop('Date')
        X_test.pop('Date')

        #remove Date, VIIRS_SCA, and hasSnow from training and testing df
        if fSCA == False:
            colrem = [ 'VIIRS_SCA', 'hasSnow']
            X_train.drop(columns = colrem, axis =1, inplace = True)
            X_test.drop(columns = colrem, axis =1, inplace = True)

        if fSCA == True:
            X_train['HasSnow'] = 0
            X_train['HasSnow'][X_train['hasSnow']==True] = 1
            X_train.pop('hasSnow')
            
            X_test['HasSnow'] = 0
            X_test['HasSnow'][X_test['hasSnow']==True] = 1
            X_test.pop('hasSnow')

        #set up prediction dataframe
        pred_obs = pd.DataFrame(y_test)
        pred_obs = pred_obs.rename(columns = {'SWE':'y_test'})


        #set up model checkpoint to be able to extract best models, ok to save locaally
        checkpointfilename ='ASWE_{val_loss:.8f}.h5'
        checkpoint_filepath = f"./Model/{Region}/fSCA_{fSCA}/"
        if not os.path.exists(checkpoint_filepath):
            os.makedirs(checkpoint_filepath)

        checkpoint_filename = checkpoint_filepath+checkpointfilename
        
        #clear all files from model folder
        try:
            files = os.listdir(checkpoint_filepath)
            for file in files:
                if file.endswith(".h5"):
                    file_path = os.path.join(checkpoint_filepath, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            print("All previous files deleted successfully.")
        except OSError:
            print("Error occurred while deleting files.")

        callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filename,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

        #get the training data shape to form MLP model input layer
        shape = X_train.shape

        #set up layer-nodes for MLP model
        LD1=128
        #seems like only slight increase in runtime with another layer
        LD2=128
        #seems like only slight increase in runtime with another layer
        LD3=64
        #seems like only slight increase in runtime with another layer, sig per improvement 
        LD4=64
        LD5=32
        LD6=16
        LD7=5

        input_1 = layers.Input(shape=(shape[1],))
        x = layers.Dense(LD1, activation="relu")(input_1)
        x = layers.Dense(LD2, activation="relu")(x)
        x = layers.Dense(LD3, activation="relu")(x)
        x = layers.Dense(LD4, activation="relu")(x)
        x = layers.Dense(LD5, activation="relu")(x)
        x = layers.Dense(LD6, activation="relu")(x)
        x = layers.Dense(1)(x)

        model = keras.Model(inputs=input_1,outputs=x)
        model.compile(loss='mse', optimizer=keras.optimizers.Adam(1e-4), metrics=['mse'])# google keras optimizers
       # print(model.summary())

        model.fit(X_train, y_train, epochs=epochs, batch_size=batchsize,
                            validation_data=(X_test,y_test),shuffle=True,callbacks=[callback], verbose=0) #shuffle -> false, can you get repeatable model results this way?

def Model_predict(RegionTest, RegionObs_Test, RegionTest_notScaled, Region_list, fSCA = False):
    
    
    Predictions = {}
    
    #Get regions
    #Regions = list(RegionTest.keys())
    
    for Region in Region_list:
    
     #set up test dfs
        X_test = RegionTest[Region].copy()
        X_test_notscaled = RegionTest_notScaled[Region]
        y_test = RegionObs_Test[Region].copy()
        
     #Set up model to predict 0" SWE if hasSnow = False
        VIIRScols = ['Date', 'VIIRS_SCA' , 'hasSnow']
        y_pred_VIIRS = X_test[VIIRScols].copy(deep = True)
        y_pred_VIIRS['y_pred_fSCA'] = np.nan

        for i in np.arange(0, len(y_pred_VIIRS),1):
            if y_pred_VIIRS['hasSnow'][i] == False:
                y_pred_VIIRS['y_pred_fSCA'][i] = 0  

        #drop VIIRS cols from prediction DF
        if fSCA == False:
            X_test = X_test.drop(VIIRScols, axis = 1)
        if fSCA == True:
            X_test['HasSnow'] = 0
            X_test['HasSnow'][X_test['hasSnow']==True] = 1
            X_test.pop('Date')
            X_test.pop('hasSnow')

        pred_obs = pd.DataFrame(RegionObs_Test[Region])
        pred_obs = pred_obs.rename(columns = {'SWE':'y_test'})

        #set up model checkpoint to be able to extract best models
        #checkpoint_filepath = f"./Model/{Region}/"
        checkpoint_filepath = f"./Model/{Region}/fSCA_{fSCA}/"
        #load the model with highest performance
        try:
            model = keras.models.load_model(f"{checkpoint_filepath}{Region}_model.keras")
        except:
            bestmodel = [f for f in listdir(checkpoint_filepath) if isfile(join(checkpoint_filepath, f))]
            bestmodel.sort(key=natural_keys)
            bestmodel = checkpoint_filepath+bestmodel[0]
            model=load_model(bestmodel)
        
    # print(bestmodel)
            #save this model
            model.save(f"{checkpoint_filepath}{Region}_model.keras")
            #make sure the model loads
            model = keras.models.load_model(f"{checkpoint_filepath}{Region}_model.keras")


         #Load SWEmax
        SWEmax = np.load(f"{checkpoint_filepath}/{Region}_SWEmax.npy")
        #make predictions and rescale, the 10 is bc changed -9999 values to -10
        y_pred = (SWEmax* model.predict(X_test, verbose = 0))

        #negative SWE is impossible, change negative values to 0
        y_pred[y_pred < 0 ] = 0
        y_test = (SWEmax * y_test)
        pred_obs['y_test'] = y_test
        pred_obs['y_pred'] = y_pred 
        pred_obs['Region'] = Region
        
          #Add in predictions from fSCA   
        pred_obs = pd.concat([pred_obs, y_pred_VIIRS], axis=1)
        pred_obs.y_pred_fSCA.fillna(pred_obs.y_pred, inplace=True)

        #combine predictions with model inputs
        df = pd.concat([pred_obs, X_test_notscaled], axis = 1)
        df = df.loc[:,~df.columns.duplicated()].copy()

        Predictions[Region] = df
        df.to_hdf(f"./Predictions/Testing/Predictions.h5", Region)
        
    return Predictions


def Prelim_Eval(Predictions):
    #Get regions
    Regions = list(Predictions.keys())
    Performance = pd.DataFrame()
    
    for Region in Regions:
        print('Preliminary Model Analysis for: ', Region)
        pred_obs = Predictions[Region]
        
        #set up model checkpoint to be able to extract best models
        checkpoint_filepath = f"./Model/{Region}/fSCA_True/"
        SWEmax = np.load(f"{checkpoint_filepath}{Region}_SWEmax.npy")
        #convert to cm
        pred_obs['y_test'] = pred_obs['y_test']*2.54
        pred_obs['y_pred'] = pred_obs['y_pred']*2.54
        pred_obs['y_pred_fSCA'] = pred_obs['y_pred_fSCA']*2.54

        
        #Run model evaluate function
        r2_test = sklearn.metrics.r2_score(pred_obs['y_test'], pred_obs['y_pred'])
        rmse_test = sklearn.metrics.mean_squared_error(pred_obs['y_test'], pred_obs['y_pred'], squared = False)
        r2_fSCA = sklearn.metrics.r2_score(pred_obs['y_test'], pred_obs['y_pred_fSCA'])
        rmse_fSCA = sklearn.metrics.mean_squared_error(pred_obs['y_test'], pred_obs['y_pred_fSCA'], squared = False)


        print(' R2 is ', r2_test)
        print(' RMSE is ', rmse_test)
        print(' R2 fSCA is ', r2_fSCA)
        print(' RMSE fSCA is ', rmse_fSCA)

        #print("MSE: %.4f" % mean_squared_error(y_test, y_pred))

        error_data = np.array([Region, round(r2_test,2),  
                               round(rmse_test,2), 
                               round(r2_fSCA,2),
                               round(rmse_fSCA,2)])

        error = pd.DataFrame(data = error_data.reshape(-1, len(error_data)), 
                             columns = ['Region', 'R2', 'RMSE', 'R2_fSCA', 'RMSE_fSCA'])
        Performance = pd.concat([Performance, error])
        

        #plot graph
        plt.scatter( pred_obs['y_test'],pred_obs['y_pred'], s=5, color="blue", label="Predictions")
        plt.scatter( pred_obs['y_test'],pred_obs['y_pred_fSCA'], s=5, color="red", label="Predictions_wfFSCA")
        plt.plot([0,SWEmax], [0,SWEmax], color = 'red', linestyle = '--')
        plt.xlabel('Observed SWE')
        plt.ylabel('Predicted SWE')

        #plt.plot(x_ax, y_pred, lw=0.8, color="red", label="predicted")
        plt.title(Region)
        plt.legend()
        plt.show()
        
    return Performance


def save_model_AWS(modelname, Region):
   #push model to AWS
    #Get a list of the necessary files
    path = f"./Model/{Region}/"
    files = []
    for file in os.listdir(path):
         # check the files which are end with specific extension
        #if file.endswith(" "):
            # print path name of selected files
        files.append(file)

    #Load and push to AWS
    pbar = ProgressBar()
    print('Pushing files to AWS')
    for file in pbar(files):
        filepath = f"{path}/{file}"
        S3.meta.client.upload_file(Filename= filepath, BUCKET=BUCKET_NAME, Key=f"{modelname}/Model/{Region}/{file}")

#function to load model from AWS if needed
def Load_Model(modelname, Region):
    #Set AWS connection

    #download model
    S3.BUCKET(BUCKET_NAME).download_file(f"{modelname}/Model/{Region}/{Region}_model.keras", f"./Model/{Region}/{Region}_model.keras")
    S3.BUCKET(BUCKET_NAME).download_file(f"{modelname}/Model/{Region}/{Region}_scaler.pkl", f"./Model/{Region}/{Region}_scaler.pkl")
    S3.BUCKET(BUCKET_NAME).download_file(f"{modelname}/Model/{Region}/{Region}_SWEmax.npy", f"./Model/{Region}/{Region}_SWEmax.npy")