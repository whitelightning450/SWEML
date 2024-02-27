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


warnings.filterwarnings("ignore")


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
        ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 2, color='gold', label = name, alpha =.2)  

    # groups = Model_Results1.groupby('Region')
    # for name, group in groups:
    #     ax1.plot( group['y_test'],group['y_pred'], marker = 'o', linestyle = ' ', markersize = 1, color='grey', label = name)

    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('gold')
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
        ax1.plot( group[x],group[y], marker = 'o', linestyle = ' ', markersize = mark_size, color='gold', label = name, alpha =.2)  
    
    xmin = min(Model_Results[x])
    xmax = max(Model_Results[x])
    ax1.legend(['Maritime', 'Alpine','Prairie'], markerscale=2, handletextpad=0.1, frameon=False)
    leg1=ax1.get_legend()
    for lh in leg1.legendHandles: 
        lh.set_alpha(1)
    leg1.legendHandles[0].set_color('royalblue')
    leg1.legendHandles[1].set_color('forestgreen')
    leg1.legendHandles[2].set_color('gold')
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
def SWE_TS_plot_classes(datelist, df, regions1, regions2, regions3, plotname, fontsize):
    
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
    fig, ax = plt.subplots(6,2, figsize=(6, 12))
    
    ax = ax.ravel()

    for i in range(len(ax.ravel())):
        key = keys[i]
        RegionDF = RegionDict[key]
        
        ax[i].plot(RegionDF.index, RegionDF.y_test, color = 'blue')
        
        if i<4:
            ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'orange')
        if 4<=i<8:
            ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'black')
        if i>=8:
            ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'red')
        
        
        if i<10:
        
            if i%2 == 0:
                ax[i].set_ylabel('Snow Water \n Equivalent (cm)', fontsize = fontsize)
                ax[i].set_xticklabels([])
            if i%2 >=0:
                ax[i].set_xticklabels([])
        else:

            if i == 10:
                ax[i].set_ylabel('Snow Water \n Equivalent (cm)', fontsize = fontsize)
                ax[i].set_xlabel('Date', fontsize = fontsize)
                ax[i].tick_params(axis='x', rotation=45)


            if i == 11:
                ax[i].set_xlabel('Date', fontsize = fontsize)
                ax[i].tick_params(axis='x', rotation=45)
                ax[i].plot(RegionDF.index, RegionDF.y_test, color = 'blue')
                ax[i].plot(RegionDF.index, RegionDF.y_pred,  color = 'red')

        #ax[0,0].set_xlabel('Date', fontsize = fontsize)
        ax[i].set_title(key, fontsize = fontsize*1.2)
        # Creating legend with color box 
    maritime = mpatches.Patch(color='orange', label='Average Regional Maritime Estimates') 
    prairie = mpatches.Patch(color='black', label='Average Regional Prairie Estimates') 
    alpine = mpatches.Patch(color='red', label='Average Regional Alpine Estimates')
    obs = mpatches.Patch(color='blue', label='Average Observations')
    #plt.legend(handles=[pop_a,pop_b]) 
    plt.legend( handles=[maritime,prairie, alpine, obs], loc = 'lower center', bbox_to_anchor = (0, 0, 1, 1),  bbox_transform = plt.gcf().transFigure, ncol = 2, fontsize = fontsize)
    plt.savefig(f"./Predictions/Hold_Out_Year/Paper_Figures/{plotname}.png", dpi = 600, box_inches = 'tight')
    return RegionDict, RegionAll
    plt.show()