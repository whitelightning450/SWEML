import os
import pandas as pd
import warnings
import hydroeval as he
import random
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from pickle import dump
import pickle 
from tqdm import tqdm
from mpl_toolkits.basemap import Basemap
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import contextily as cx
import matplotlib.patches as mpatches 
import geopandas as gpd
import xyzservices.providers as xyz
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import boto3
from progressbar import ProgressBar
from botocore import UNSIGNED
from botocore.client import Config
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


#plot time series of regionally average obs and preds
def Peak_SWE(datelist, df, RegionList):
    
 
    RegionStats = pd.DataFrame()   
    for region in RegionList:
        RegionDF = pd.DataFrame()
        cols = ['Date', 'y_test', 'y_pred']
        for date in datelist:
            RegionDate = df[region][df[region]['Date'] == date].copy()
            RegionDate = RegionDate[cols]
            RegionDate.set_index('Date', inplace = True, drop = True)
            RegionDF = pd.concat([RegionDF, RegionDate])

        RegionDF.index = pd.to_datetime(RegionDF.index)
        RegionDF = RegionDF.resample('D').mean().dropna()
        if region == 'N_Sierras':
            name = 'Northern Sierra Nevada'
        elif region == 'S_Sierras_High':
            name = 'Southern Sierra Nevada High'
        elif region == 'S_Sierras_Low':
            name = 'Southern Sierra Nevada Low'
        elif region == 'Greater_Yellowstone':
            name = 'Greater Yellowstone'
        elif region == 'N_Co_Rockies':
            name = 'Upper Colorado Rockies'
        elif region == 'SW_Mont':
            name = 'SW Montana'
        elif region == 'SW_Co_Rockies':
            name = 'San Juan Mountains'
        elif region == 'GBasin':
            name = 'Great Basin'
        elif region == 'N_Wasatch':
            name = 'Northern Wasatch'
        elif region == 'N_Cascade':
            name = 'Northern Cascades'
        elif region == 'S_Wasatch':
            name = 'SW Utah'
        elif region == 'SW_Mtns':
            name = 'SW Desert'
        elif region == 'E_WA_N_Id_W_Mont':
            name = 'NW Rockies'
        elif region == 'S_Wyoming':
            name = 'Northern Colorado Rockies'
        elif region == 'SE_Co_Rockies':
            name = 'Sangre de Cristo Mountains'
        elif region == 'Ca_Coast':
            name = 'California Coast Range'
        elif region == 'E_Or':
            name = 'Blue Mountains of Oregon'
        elif region == 'N_Yellowstone':
            name = 'Elkhorn Mountains of Montana '
        elif region == 'S_Cascade':
            name = 'Southern Cascades'
        elif region == 'Wa_Coast':
            name = 'Washington Coast Range '
        elif region == 'Greater_Glacier':
            name = 'Northern Rockies'
        elif region == 'Or_Coast':
            name = 'Oregon Coast Range'
        elif region == 'Sawtooth':
            name = 'Sawtooth'
        
        #add max obs sw
        maxpred = max(RegionDF['y_pred'])
        maxobs = max(RegionDF['y_test'])
        dateobs = RegionDF[RegionDF['y_test'] == maxobs].index[0].strftime("%Y-%m-%d")
        datepred = RegionDF[RegionDF['y_pred'] == maxpred].index[0].strftime("%Y-%m-%d")
        #add metrics to df
        cols = ['Region', 'ObsMax', 'PredMax', 'ObsMaxDate', 'PredMaxDate']
        data = [name, maxobs, maxpred,dateobs, datepred]
        DF = pd.DataFrame(data)
        DF = DF.T
        DF.columns = cols
   
        #display(DF)
        #add to regional stats df
        RegionStats = pd.concat([RegionStats, DF])
    RegionStats.set_index('Region', inplace = True, drop =True)
    return RegionStats



def Dict_2_DF(EvalDF, Region_list):
    #Change the location names to improve the interpretability of the figures
    Model_Results= pd.DataFrame(columns = ['y_test', 'y_pred',  'Region'])

    for region in Region_list:
        if region == 'N_Sierras':
            EvalDF[region]['Region'] = 'Northern Sierra Nevada'
        elif region == 'S_Sierras_High':
            EvalDF[region]['Region'] = 'Southern Sierra Nevada High'
        elif region == 'S_Sierras_Low':
            EvalDF[region]['Region'] = 'Southern Sierra Nevada Low'
        elif region == 'Greater_Yellowstone':
            EvalDF[region]['Region'] = 'Greater Yellowstone'
        elif region == 'N_Co_Rockies':
            EvalDF[region]['Region'] = 'Upper Colorado Rockies'
        elif region == 'SW_Mont':
            EvalDF[region]['Region'] = 'SW Montana'
        elif region == 'SW_Co_Rockies':
            EvalDF[region]['Region'] = 'San Juan Mountains'
        elif region == 'GBasin':
            EvalDF[region]['Region'] = 'Great Basin'
        elif region == 'N_Wasatch':
            EvalDF[region]['Region'] = 'Northern Wasatch'
        elif region == 'N_Cascade':
            EvalDF[region]['Region'] = 'Northern Cascades'
        elif region == 'S_Wasatch':
            EvalDF[region]['Region'] = 'SW Utah'
        elif region == 'SW_Mtns':
            EvalDF[region]['Region'] = 'SW Desert'
        elif region == 'E_WA_N_Id_W_Mont':
            EvalDF[region]['Region'] = 'NW Rockies'
        elif region == 'S_Wyoming':
            EvalDF[region]['Region'] = 'Northern Colorado Rockies'
        elif region == 'SE_Co_Rockies':
            EvalDF[region]['Region'] = 'Sangre de Cristo Mountains'
        elif region == 'Ca_Coast':
            EvalDF[region]['Region'] = 'California Coast Range'
        elif region == 'E_Or':
            EvalDF[region]['Region'] = 'Blue Mountains of Oregon'
        elif region == 'N_Yellowstone':
            EvalDF[region]['Region'] = 'Elkhorn Mountains of Montana '
        elif region == 'S_Cascade':
            EvalDF[region]['Region'] = 'Southern Cascades'
        elif region == 'Wa_Coast':
            EvalDF[region]['Region'] = 'Washington Coast Range '
        elif region == 'Greater_Glacier':
            EvalDF[region]['Region'] = 'Northern Rockies'
        elif region == 'Or_Coast':
            EvalDF[region]['Region'] = 'Oregon Coast Range'

        Model_Results = Model_Results.append(EvalDF[region])
        
    Model_Results['error'] = Model_Results['y_test']-Model_Results['y_pred']
    
    return Model_Results




#Sturm classification of performance
def Sturm_Classified_Performance(Model_Results):
    Maritime_Region = ['Southern Sierra Nevada High','Southern Sierra Nevada Low', 'Northern Sierra Nevada','Southern Cascades',
                      'Northern Cascades', 'California Coast Range', 'Washington Coast Range ', 
                      'Oregon Coast Range']

    Prairie_Region  =  ['Elkhorn Mountains of Montana ','SW Montana', 'Great Basin', 'SW Utah', 'Sawtooth', 'SW Desert']

    Alpine_Region =['Blue Mountains of Oregon', 'Northern Wasatch', 'NW Rockies', 'Greater Yellowstone', 'Upper Colorado Rockies','Northern Colorado Rockies', 'San Juan Mountains',
                         'Northern Rockies', 'Sangre de Cristo Mountains']

    Snow_Class = {'Maritime':Maritime_Region, 
                  'Alpine':Alpine_Region, 
                  # 'Transitional':Transitional_Region, 
                  'Prairie':Prairie_Region}

    for snow in Snow_Class.keys():
        regions = Snow_Class[snow]
        Class = Model_Results[Model_Results['Region'].isin(regions)]
        y_test = Class['y_test']
        y_pred = Class['y_pred']
      #Run model evaluate function
        r2 = sklearn.metrics.r2_score(y_test, y_pred)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
        PBias = he.evaluator(he.pbias, y_pred, y_test)

        print(snow, ' RMSE: ', rmse, ' R2: ', r2, 'pbias:', PBias)
    return Maritime_Region, Prairie_Region, Alpine_Region, Snow_Class




# Figure 3, predicted vs observed for Southern Sierra Nevada, Upper Colorado Rockiesk All regions (subsetted into maritime ,apine, prarie)
def Slurm_Class_parity(Model_Results, Maritime_Region, Prairie_Region, Alpine_Region):

    Model_Results1 = Model_Results[Model_Results['Region'].isin(Maritime_Region)]
    Model_Results2 = Model_Results[Model_Results['Region'].isin(Prairie_Region)]
    Model_Results3 = Model_Results[Model_Results['Region'].isin(Alpine_Region)]

    # fig, (ax1, ax2,ax3) = plt.subplots(3, 1, figsize=(3,9))
    font= 10
    tittle_font = font*1.2

    fig = plt.figure(figsize=(5.5,5.5))

    gs=GridSpec(2,2)
    ax1 = fig.add_subplot(gs[:,1])
    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[1,0])

    plt.subplots_adjust(hspace=0.2, wspace=0.25)

    #all grouping
    groups_maritime = Model_Results1.groupby('Region')
    for name, group in groups_maritime:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='royalblue', label = name, alpha =.2)
    groups_alpine = Model_Results3.groupby('Region')
    for name, group in groups_alpine:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='forestgreen', label = name, alpha =.4)
    groups_prairie = Model_Results2.groupby('Region')
    for name, group in groups_prairie:
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='red', label = name, alpha =.2)  

    # groups = Model_Results1.groupby('Region')
    # for name, group in groups:
    #     ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 1, color='grey', label = name)

    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('red')
    ax1.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    #ax1.set_xlabel('Observed SWE (cm)')
    # ax1.set_ylabel('Predicted SWE (cm)')
    ax1.set_title('All Regions')
    ax1.set_xlim(0,300)
    ax1.set_ylim(0,300)
    ax1.tick_params(axis='y', which='major', pad=1)
    ax1.set_xlabel('Observed SWE (cm)')

    #Sierra Nevada grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Southern Sierra Nevada High") | (Model_Results["Region"]=="Southern Sierra Nevada Low")].groupby('Region')
    for name, group in groups:
        ax2.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='grey', label = name, alpha = .4)

    # ax2.legend(title ='Snow Classification: Prairie', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1, 1), markerscale = 2)
    ax2.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    #ax2.set_xlabel('Observed SWE (cm)')
    # ax2.set_ylabel('Predicted SWE (cm)')
    ax2.set_title('Southern Sierra Nevada')
    ax2.set_xlim(0,300)
    ax2.set_ylim(0,300)
    ax2.xaxis.set_ticklabels([])
    tick_spacing = 100
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax2.tick_params(axis='y', which='major', pad=1)
    ax2.set_ylabel('Predicted SWE (cm)', labelpad=0)

    #Colorado Rockies Grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Upper Colorado Rockies")].groupby('Region')
    for name, group in groups:
        ax3.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='grey', label = name,alpha = .4)

    # ax3.legend(title ='Snow Classification: Alpine', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1., 1.), markerscale = 2)
    ax3.plot([0,Model_Results['y_test'].max()], [0,Model_Results['y_test'].max()], color = 'red', linestyle = '--')
    ax3.set_xlabel('Observed SWE (cm)')
    ax3.set_ylabel('Predicted SWE (cm)', labelpad=0)
    ax3.set_title('Upper Colorado Rockies')
    ax3.set_xlim(0,300)
    ax3.set_ylim(0,300)
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax3.tick_params(axis='y', which='major', pad=1)

    #save figure
    plt.savefig('./Predictions/Hold_Out_Year/Paper_Figures/Parity_Plot_All4.png', dpi =600, bbox_inches='tight')
    plt.savefig('./Predictions/Hold_Out_Year/Paper_Figures/Parity_Plot_All4.pdf', dpi =600, bbox_inches='tight')
    
    
#Evaluate by Elevation cleaned up
def EvalPlots3(Model_Results, Maritime_Region, Prairie_Region, Alpine_Region, x, y, xlabel, ylabel, plotname, mark_size):
    # fig, (ax1, ax2,ax3) = plt.subplots(2, 1, 2, figsize=(3,9))
    
    Model_Results1 = Model_Results[Model_Results['Region'].isin(Maritime_Region)]
    Model_Results2 = Model_Results[Model_Results['Region'].isin(Prairie_Region)]
    Model_Results3 = Model_Results[Model_Results['Region'].isin(Alpine_Region)]
    
    font= 10
    tittle_font = font*1.2
    
    fig = plt.figure(figsize=(5,6))
    
    gs=GridSpec(2,2)
    ax1 = fig.add_subplot(gs[1,:])
    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[0,1])

    plt.subplots_adjust(hspace=0.3, wspace=0.1)

    #all grouping
    groups_maritime = Model_Results1.groupby('Region')
    for name, group in groups_maritime:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='royalblue', label = name, alpha =.2)
    groups_alpine = Model_Results3.groupby('Region')
    for name, group in groups_alpine:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='forestgreen', label = name, alpha =.4)
    groups_prairie = Model_Results2.groupby('Region')
    for name, group in groups_prairie:
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='red', label = name, alpha =.2)  
    
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('red')
    ax1.set_title('All Regions')
    ax1.hlines(y=0,xmin = xmin, xmax=xmax, color = 'black', linestyle = '--')
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_ylim(-150,150)
    ax1.set_ylabel(ylabel, labelpad=-10)

    #Sierra Nevada grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Southern Sierra Nevada High") | (Model_Results["Region"]=="Southern Sierra Nevada Low")].groupby('Region')
    for name, group in groups:
        ax2.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='grey', label = name, alpha = .4)
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    # ax2.legend(title ='Snow Classification: Prairie', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1, 1), markerscale = 2)
    ax2.set_title('Southern Sierra Nevada')
    ax2.hlines(y=0,xmin = xmin, xmax=xmax,  color = 'black', linestyle = '--')
    #ax2.set_xlabel('Observed SWE (in)')
    ax2.set_ylabel(ylabel, labelpad=-10)
    ax2.set_ylim(-150,150)



    #Colorado Rockies Grouping
    groups = Model_Results.loc[(Model_Results["Region"]=="Upper Colorado Rockies")].groupby('Region')
    for name, group in groups:
        ax3.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='grey', label = name, alpha = .4)
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    # ax3.legend(title ='Snow Classification: Alpine', fontsize=font, title_fontsize=tittle_font, ncol = 1, bbox_to_anchor=(1., 1.), markerscale = 2)
    ax3.set_title('Upper Colorado Rockies') 
    ax3.hlines(y=0,xmin = xmin, xmax=xmax,  color = 'black', linestyle = '--')
    ax3.yaxis.set_ticklabels([])
    # ax3.set_xlabel(xlabel)
    # ax3.set_ylabel(ylabel)
    ax3.set_ylim(-150,150)



    # save figure
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}3.png", dpi =600, bbox_inches='tight')
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}3.pdf", dpi =600, bbox_inches='tight')
    
    
    
#plot time series of regionally average obs and preds
def SWE_TS_plot(datelist, df, regions, plotname):
    
    RegionDict = {}
    for region in regions:
        RegionDF = pd.DataFrame()
        cols = ['Date', 'y_test', 'y_pred']
        for date in datelist:
            RegionDate = df[region][df[region]['Date'] == date].copy()
            RegionDate = RegionDate[cols]
            RegionDate.set_index('Date', inplace = True, drop = True)
            RegionDF = pd.concat([RegionDF, RegionDate])

        RegionDF.index = pd.to_datetime(RegionDF.index)
        RegionDF = RegionDF.resample('D').mean().dropna()
        if region == 'N_Sierras':
            name = 'Northern Sierra Nevada'
        elif region == 'S_Sierras_High':
            name = 'Southern Sierra Nevada High'
        elif region == 'S_Sierras_Low':
            name = 'Southern Sierra Nevada Low'
        elif region == 'Greater_Yellowstone':
            name = 'Greater Yellowstone'
        elif region == 'N_Co_Rockies':
            name = 'Upper Colorado Rockies'
        elif region == 'SW_Mont':
            name = 'SW Montana'
        elif region == 'SW_Co_Rockies':
            name = 'San Juan Mountains'
        elif region == 'GBasin':
            name = 'Great Basin'
        elif region == 'N_Wasatch':
            name = 'Northern Wasatch'
        elif region == 'N_Cascade':
            name = 'Northern Cascades'
        elif region == 'S_Wasatch':
            name = 'SW Utah'
        elif region == 'SW_Mtns':
            name = 'SW Desert'
        elif region == 'E_WA_N_Id_W_Mont':
            name = 'NW Rockies'
        elif region == 'S_Wyoming':
            name = 'Northern Colorado Rockies'
        elif region == 'SE_Co_Rockies':
            name = 'Sangre de Cristo Mountains'
        elif region == 'Ca_Coast':
            name = 'California Coast Range'
        elif region == 'E_Or':
            name = 'Blue Mountains of Oregon'
        elif region == 'N_Yellowstone':
            name = 'Elkhorn Mountains of Montana '
        elif region == 'S_Cascade':
            name = 'Southern Cascades'
        elif region == 'Wa_Coast':
            name = 'Washington Coast Range '
        elif region == 'Greater_Glacier':
            name = 'Northern Rockies'
        elif region == 'Or_Coast':
            name = 'Oregon Coast Range'
        elif region == 'Sawtooth':
            name = 'Sawtooth'
        RegionDict[name] = RegionDF
    
    #Get keys from dictionary
    keys = list(RegionDict.keys())
    print(keys)

    #make figure
    fig, ax = plt.subplots(2,2, figsize=(7, 7))
    
    ax = ax.ravel()

    for i in range(len(ax.ravel())):
        key = keys[i]
        RegionDF = RegionDict[key]
        
        #fig.patch.set_facecolor('white')
        ax[i].plot(RegionDF.index, RegionDF.y_test, color = 'blue')
        ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'orange')
        
        if i == 0:
            ax[i].set_ylabel('Snow Water Equivalent (cm)', fontsize = 12)
            ax[i].set_xticklabels([])
        if i == 1:
            ax[i].set_xticklabels([])
            
        if i == 2:
            ax[i].set_ylabel('Snow Water Equivalent (cm)', fontsize = 12)
            ax[i].set_xlabel('Date', fontsize = 12)
            ax[i].tick_params(axis='x', rotation=45)
            
            
        if i == 3:
            ax[i].set_xlabel('Date', fontsize = 12)
            ax[i].tick_params(axis='x', rotation=45)
            ax[i].plot(RegionDF.index, RegionDF.y_test, color = 'blue', label = 'Average in situ observations')
            ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'orange', label = 'Average regional estimates')
            
        #ax[0,0].set_xlabel('Date', fontsize = 12)
        ax[i].set_title(key, fontsize = 14)

    plt.legend( loc = 'lower center', bbox_to_anchor = (0, -0.1, 1, 1),  bbox_transform = plt.gcf().transFigure, ncol = 2)
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}.png", dpi = 600, box_inches = 'tight')
    plt.show()
    
    
#plot time series of regionally average obs and preds
def SWE_TS_plot_classes(datelist, df, regions1, regions2, regions3, plotname, fontsize, opacity):
    
    RegionAll = regions1+regions2+regions3

 
    RegionDict = {}
    for region in RegionAll:
        RegionDF = pd.DataFrame()
        cols = ['Date', 'y_test', 'y_pred']
        for date in datelist:
            RegionDate = df[region][df[region]['Date'] == date].copy()
            RegionDate = RegionDate[cols]
            RegionDate.set_index('Date', inplace = True, drop = True)
            RegionDF = pd.concat([RegionDF, RegionDate])

        RegionDF.index = pd.to_datetime(RegionDF.index)
        #RegionDF = RegionDF.resample('D').mean().dropna()
        RegionDF = RegionDF.resample('D', base=0).agg(['min','max','mean']).round(1).dropna()
        if region == 'N_Sierras':
            name = 'Northern Sierra Nevada'
        elif region == 'S_Sierras_High':
            name = 'Southern Sierra Nevada High'
        elif region == 'S_Sierras_Low':
            name = 'Southern Sierra Nevada Low'
        elif region == 'Greater_Yellowstone':
            name = 'Greater Yellowstone'
        elif region == 'N_Co_Rockies':
            name = 'Upper Colorado Rockies'
        elif region == 'SW_Mont':
            name = 'SW Montana'
        elif region == 'SW_Co_Rockies':
            name = 'San Juan Mountains'
        elif region == 'GBasin':
            name = 'Great Basin'
        elif region == 'N_Wasatch':
            name = 'Northern Wasatch'
        elif region == 'N_Cascade':
            name = 'Northern Cascades'
        elif region == 'S_Wasatch':
            name = 'SW Utah'
        elif region == 'SW_Mtns':
            name = 'SW Desert'
        elif region == 'E_WA_N_Id_W_Mont':
            name = 'NW Rockies'
        elif region == 'S_Wyoming':
            name = 'Northern Colorado Rockies'
        elif region == 'SE_Co_Rockies':
            name = 'Sangre de Cristo Mountains'
        elif region == 'Ca_Coast':
            name = 'California Coast Range'
        elif region == 'E_Or':
            name = 'Blue Mountains of Oregon'
        elif region == 'N_Yellowstone':
            name = 'Elkhorn Mountains of Montana '
        elif region == 'S_Cascade':
            name = 'Southern Cascades'
        elif region == 'Wa_Coast':
            name = 'Washington Coast Range '
        elif region == 'Greater_Glacier':
            name = 'Northern Rockies'
        elif region == 'Or_Coast':
            name = 'Oregon Coast Range'
        elif region == 'Sawtooth':
            name = 'Sawtooth'
        RegionDict[name] = RegionDF
    
    #Get keys from dictionary
    keys = list(RegionDict.keys())
    print(keys)

    #make figure
    fig, ax = plt.subplots(6,2, figsize=(4, 8))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    ax = ax.ravel()

    for i in range(len(ax.ravel())):
        key = keys[i]
        RegionDF = RegionDict[key]
        
        ax[i].plot(RegionDF.index, RegionDF['y_test']['mean'], color = 'black')
        ax[i].fill_between(RegionDF.index,RegionDF['y_test']['mean'],RegionDF['y_test']['min'],interpolate=True,color='black', alpha=opacity)
        ax[i].fill_between(RegionDF.index,RegionDF['y_test']['mean'],RegionDF['y_test']['max'],interpolate=True,color='black', alpha=opacity)
        ax[i].tick_params(axis='y', labelsize=fontsize)
        
        if i<4:
            ax[i].plot(RegionDF.index, RegionDF['y_pred']['mean'],  color = 'royalblue')
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['min'],interpolate=True,color='darkblue', alpha=opacity*2)
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['max'],interpolate=True,color='darkblue', alpha=opacity*2)
        if 4<=i<8:
            ax[i].plot(RegionDF.index, RegionDF['y_pred']['mean'],  color = 'red')
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['min'],interpolate=True,color='red', alpha=opacity*2)
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['max'],interpolate=True,color='red', alpha=opacity*2)
        if i>=8:
            ax[i].plot(RegionDF.index, RegionDF['y_pred']['mean'],  color = 'forestgreen')
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['min'],interpolate=True,color='forestgreen', alpha=opacity*2)
            ax[i].fill_between(RegionDF.index,RegionDF['y_pred']['mean'],RegionDF['y_pred']['max'],interpolate=True,color='forestgreen', alpha=opacity*2)
        
        
        if i<10:
        
            if i%2 == 0:
                ax[i].set_xticklabels([])
            if i%2 >=0:
                ax[i].set_xticklabels([])
        else:

            if i == 10:
                ax[i].tick_params(axis='x', rotation=45, labelsize=fontsize)


            if i == 11:
                ax[i].tick_params(axis='x', rotation=45, labelsize=fontsize)
                ax[i].plot(RegionDF.index, RegionDF['y_test']['mean'], color = 'black')
                ax[i].plot(RegionDF.index, RegionDF['y_pred']['mean'],  color = 'forestgreen')

      
        ax[i].set_title(key, fontsize = fontsize*1.2)
        # Creating legend with color box 
    maritime = mpatches.Patch(color='royalblue', label='Average Regional Maritime Estimates') 
    prairie = mpatches.Patch(color='red', label='Average Regional Prairie Estimates') 
    alpine = mpatches.Patch(color='forestgreen', label='Average Regional Alpine Estimates')
    obs = mpatches.Patch(color='black', label='Average Observations')

    fig.text(0.06, 0.5, 'Snow Water Equivalent (cm)', ha='center', va='center', rotation='vertical', fontsize= fontsize*1.2)
    fig.text(0.5, 0.06, 'Datetime', ha='center', va='center', fontsize= fontsize*1.2)
    plt.legend( handles=[maritime,prairie, alpine, obs], loc = 'lower center', bbox_to_anchor = (0, 0, 1, 1),  bbox_transform = plt.gcf().transFigure, ncol = 2, fontsize = fontsize)
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}.png", dpi = 600, box_inches = 'tight')
    return RegionDict, RegionAll
    plt.show()

def SSM_Fig(datelist, Region_list,variant):
    
    #Load prediction file with geospatial information
    path = f"./Predictions/Hold_Out_Year/Prediction_DF_SCA_2018-10-02.pkl"
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

    path = f"./Predictions/Hold_Out_Year/2019_predictions.h5"
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
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        ax = plotdf.plot(date, 
                     figsize=(3, 5), 
                     alpha=0.5, 
                     markersize = 6,
                     edgecolor="k", 
                     vmin =1, 
                     vmax =250,
                    legend = True,
                    legend_kwds={"label": "Snow Water Equivalent (cm)", "orientation": "vertical"},
                    ax = ax,
                    cax= cax)
        ax.set_xlim(-1.365e7, -1.31e7)
        #ax.set_ylim(4.25e6, 5.25e6)
        ax.set_ylim(4.32e6, 4.62e6)
        cx.add_basemap(ax, source="https://server.arcgisonline.com/ArcGIS/rest/services/"+variant+"/MapServer/tile/{z}/{y}/{x}")   #cx.providers.OpenStreetMap.Mapnik)
        ax.set_axis_off()
        ax.text(-1.345e7, 4.64e6, f"SWE estimate: {date}", fontsize =14)
        #plt.title( f"SWE estimate: {date}", fontsize =14)
        #plt.title(f"SWE estimate: {date}")
        plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/SWE_{date}.png", dpi =600, bbox_inches='tight')
        plt.show()
        plt.close(fig)

def slurmNSE(df, slurm_class):
    regionsNSE = []
    for region in slurm_class:
        sites = df[region].index.unique()

        NSE = []
        cols = ['y_test', 'y_pred']
        for site in sites:
            sitedf = df[region][df[region].index ==site][cols]
            #0's are causing issues
            sitedf['y_pred'][sitedf['y_pred']<0.1]=0.1
            sitedf['y_test'][sitedf['y_test']<0.1]=0.1
            #display(sitedf.head(40))
            sitense = he.nse(sitedf['y_pred'].values,sitedf['y_test'].values)
            NSE.append(sitense)
            #change values less than 0 to 0
            NSE = [0 if b < 0 else b for b in NSE]
        regionsNSE = regionsNSE + NSE
        regionsNSE.sort()
        
    return np.array(regionsNSE)

def getSNODAS_AWS(modelname):
    file = 'SNODAS/SNODAS_WY2018.pkl'
    filename = f"{HOME}/SWEML/Model/{modelname}/Predictions/Hold_Out_Year/{file}"
    S3.meta.client.download_file(BUCKET_NAME, file, filename)

def SNODASslurmNSE(EvalDF, SNODAS, slurm_class):
    regionsNSE = []
    for region in slurm_class:
        sites = EvalDF[region].index.unique()

        #Get evaluataion sites in SNODAS
        SNODAS['region'] = SNODAS[region].T[sites].T

        NSE = []
        cols = ['Date', 'y_test']
        for site in np.arange(0,len(sites),1):
            #transform SNODAS df, process each site, process each Eval sit3
            SNODASsite = pd.DataFrame(SNODAS[region].loc[sites[site]])
            SNODASsite.reset_index(inplace = True)
            SNODASsite.rename(columns = {sites[site]: 'y_pred', 'index':'Date'}, inplace = True)
            SNODASsite.set_index('Date', inplace=True)

            Evalsite = EvalDF[region].loc[sites[site]][cols]
            Evalsite.set_index('Date', inplace = True)

            #Merge SNODAS and Eval df site
            Evalsite = pd.concat([Evalsite, SNODASsite], axis=1).dropna()

            #0's are causing issues
            Evalsite['y_pred'][Evalsite['y_pred']<0.1]=0.1
            Evalsite['y_test'][Evalsite['y_test']<0.1]=0.1

            #convert snodas from m to cm
            Evalsite['y_pred'] = Evalsite['y_pred']*100

            sitense = he.nse(Evalsite['y_pred'].values,Evalsite['y_test'].values)
            NSE.append(sitense)
            #change values less than 0 to 0
            NSE = [0 if b < 0 else b for b in NSE]
        regionsNSE = regionsNSE + NSE
        regionsNSE.sort()
        
    return np.array(regionsNSE)

def regionCDF(MaritimeNSE, PrarieeNSE, AlpineNSE, SNODAS_MaritimeNSE, SNODAS_PrarieeNSE, SNODAS_AlpineNSE, SNODAS, plt_save):
    # calculate the proportional values of samples
    Mp = 1. * np.arange(len(MaritimeNSE)) / (len(MaritimeNSE) - 1)
    Pp = 1. * np.arange(len(PrarieeNSE)) / (len(PrarieeNSE) - 1)
    Ap = 1. * np.arange(len(AlpineNSE)) / (len(AlpineNSE) - 1)

    SMp = 1. * np.arange(len(SNODAS_MaritimeNSE)) / (len(SNODAS_MaritimeNSE) - 1)
    SPp = 1. * np.arange(len(SNODAS_PrarieeNSE)) / (len(SNODAS_PrarieeNSE) - 1)
    SAp = 1. * np.arange(len(SNODAS_AlpineNSE)) / (len(SNODAS_AlpineNSE) - 1)

    # plot the sorted data:
    fig = plt.figure(figsize=(10,5))
    ax2 = fig.add_subplot(122)
    #SWEML
    ax2.plot(MaritimeNSE, Mp, color = 'royalblue', label = 'Maritime')
    ax2.plot(PrarieeNSE, Pp, color = 'red', label = 'Prairie')
    ax2.plot(AlpineNSE, Ap, color = 'forestgreen', label = 'Alpine')

    if SNODAS == True:
    #SNODAS
        ax2.plot(SNODAS_MaritimeNSE, SMp, color = 'royalblue', label = 'SNODAS Maritime', linestyle ='--')
        ax2.plot(SNODAS_PrarieeNSE, SPp, color = 'red', label = 'SNODAS Prairie', linestyle ='--')
        ax2.plot(SNODAS_AlpineNSE, SAp, color = 'forestgreen', label = 'SNODAS Alpine', linestyle ='--')


    ax2.set_xlabel('$NSE$')
    ax2.set_ylabel('$p$')
    ax2.legend()
    if plt_save == True:
        plt.savefig(f"Predictions/Hold_Out_Year/Paper_Figures/CDF.png", dpi =600, bbox_inches='tight')
    plt.show()