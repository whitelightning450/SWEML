# created by Dr. Ryan C. Johnson as part of the Cooperative Institute for Research to Operations in Hydrology (CIROH)
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
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Bidirectional, LSTM
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import threading  # this is the threading library
import warnings; warnings.filterwarnings("ignore")
#from tensorflow.keras.optimizers.legacy import Adam
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Input, Conv1D, MaxPooling1D


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


def Model_train(cwd, epochs, batchsize, RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, Region):
    
    print('Training model for: ', Region)
    #set up train/test dfs
    X_train = RegionTrain[Region].copy()
    X_test = RegionTest[Region].copy()
    y_train = RegionObs_Train[Region].copy()
    y_test = RegionObs_Test[Region].copy()

    #remove Date, VIIRS_SCA, and hasSnow from training and testing df
    colrem = ['Date', 'VIIRS_SCA', 'hasSnow']
    X_train.drop(columns = colrem, axis =1, inplace = True)
    X_test.drop(columns = colrem, axis =1, inplace = True)

    #set up prediction dataframe
    pred_obs = pd.DataFrame(y_test)
    pred_obs = pred_obs.rename(columns = {'SWE':'y_test'})


    #set up model checkpoint to be able to extract best models
    checkpointfilename ='ASWE_{val_loss:.8f}.h5'
    checkpoint_filepath = f"{cwd}/Model/{Region}/"

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

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()

    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    
    print(X_train.shape)

    model = Sequential()

    model.add(Bidirectional(LSTM(600, activation='relu')))

    model.add(Dense(1))

    # Compile Model ==================================================================

    model.compile(optimizer=Adam(learning_rate=0.001, decay=1e-3), loss=tf.keras.losses.MeanAbsoluteError())

    # Train  and save Our Model  =====================================================
    
    checkpointfilename = Region + '_best.h5'
    checkpoint_filepath = f"Model/{Region}/"

    checkpoint_filename = checkpoint_filepath+checkpointfilename

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,  patience=20)

    mc = ModelCheckpoint(checkpoint_filename, monitor='val_loss', mode='min', verbose=0, save_best_only=True)

    model.fit(X_train, y_train, batch_size=batchsize, epochs=epochs, validation_split=0.1,        
                                  verbose=0, shuffle='False', callbacks=[mc])  #es, 
    
    
    bestmodel = checkpoint_filename
    model_temp=load_model(bestmodel)

    #save this model
    model_temp.save(f"{checkpoint_filepath}{Region}_model.keras")

         

def Model_predict(cwd, RegionTest, RegionObs_Test, RegionTest_notScaled, Region):
    
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
    X_test = X_test.drop(VIIRScols, axis = 1)

    pred_obs = pd.DataFrame(RegionObs_Test[Region])
    pred_obs = pred_obs.rename(columns = {'SWE':'y_test'})

    #set up model checkpoint to be able to extract best models
    checkpoint_filepath = f"Model/{Region}/"

    SWEmax = np.load(f"{checkpoint_filepath}{Region}_SWEmax.npy")


    X_test = X_test.to_numpy()
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = keras.models.load_model(f"{checkpoint_filepath}{Region}_model.keras")
    y_pred = (SWEmax* model.predict(X_test))



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
    df = pd.concat([pred_obs.reset_index(drop=True), X_test_notscaled.reset_index(drop=True)], axis = 1)
    df = df.loc[:,~df.columns.duplicated()].copy()

    Predictions = df
    df.to_hdf(f"Predictions/Testing/Predictions.h5", Region)

        
    return Predictions



def Prelim_Eval(cwd,Predictions):
    #Get regions
    Regions = list(Predictions.keys())
    Performance = pd.DataFrame()
    
    for Region in Regions:
        print('Preliminary Model Analysis for: ', Region)
        pred_obs = Predictions[Region]
        
        #set up model checkpoint to be able to extract best models
        checkpoint_filepath = f"Model/{Region}/"
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
        Performance = Performance.append(error, ignore_index = True)

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

