import os
from tqdm import tqdm
import MLP_Model
import sys

sys.path.insert(0, '..')
from shared_scripts import DataProcess, Hindcast_Initialization, NSM_SCA
import warnings
import datetime

warnings.filterwarnings('ignore')

# Set working directories
cwd = os.getcwd()
datapath = f"{os.path.expanduser('~')}/SWEML"


def sweml_hindcast(new_year, threshold, Region_list, fSCA, frequency, NewSim, single_day):
    model = 'Neural_Network'

    retries = 0
    if single_day:
        prev_data_flag = True
        days = 4
        while prev_data_flag and retries < 60:
            dt = datetime.datetime.now() - datetime.timedelta(days=days)
            prev_dt = dt - datetime.timedelta(days=1)
            datelist = [dt.strftime('%Y-%m-%d')]
            if os.path.isfile(f"Predictions/Hold_Out_Year/Daily/fSCA_True/Prediction_DF_SCA_{prev_dt.strftime('%Y-%m-%d')}.pkl"):
                prev_data_flag = False
            else:
                days += 1
                retries += 1

        if datelist[0][-5:] == '10-01':
            Hindcast_Initialization.Hindcast_Initialization(cwd, datapath, new_year, threshold, Region_list,
                                                            frequency, fSCA=fSCA)
    else:
        datelist = Hindcast_Initialization.Hindcast_Initialization(cwd, datapath, new_year, threshold, Region_list,
                                                                   frequency, fSCA=fSCA)

    # Run data processing script to partition key regional dataframes
    # note, need to load RegionTrain_SCA.h5,
    if datelist[0][-5:] == '10-01':
        RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, RegionTest_notScaled = DataProcess.DataProcess(
            new_year, frequency, model, Region_list, fSCA=fSCA)

        """
        # model training, each participants model will be different but should follow the prescribed input feature 
        template epochs= 30 
        batchsize = 60 
        MLP_Model.Model_train(epochs, batchsize, RegionTrain, RegionTest, RegionObs_Train, RegionObs_Test, Region_list, 
                              fSCA = fSCA)
        """

        # Need to create Predictions folder if running for the first time
        Predictions = MLP_Model.Model_predict(RegionTest, RegionObs_Test, RegionTest_notScaled, Region_list, fSCA=fSCA)

    for day in tqdm(datelist):
        # connect interactive script to Wasatch Snow module
        snow = NSM_SCA.NSM_SCA(day, threshold=threshold, Regions=Region_list, modelname=model, frequency=frequency,
                               fSCA=fSCA, NewSim=NewSim)

        # Go get SNOTEL observations - all data currently loaded, set to True to download
        snow.Get_Monitoring_Data_Threaded(getdata=True)

        # Initialize/Download the granules, all data preprocessed for the SSM activRegion_listhange to True to use
        # the functions.
        snow.initializeGranules(getdata=True)

        # Process observations into Model prediction ready format,
        snow.Data_Processing(SCA=True)

        # Agument with SCA
        snow.augmentPredictionDFs()

        # Make predictions, set NewSim to False Look to multiprocess, each region can do a prediction to speed things
        # up. set NewSim to true for New simulation, turn to false once all data has been proces and saved.
        snow.SWE_Predict(NewSim=NewSim, Corrections=False, fSCA=fSCA)
        snow.netCDF_compressed(plot=False)
        snow.Geo_df()
    
    modelname = 'Neural_Network'
    folderpath = 'Predictions/Hold_Out_Year/Daily/fSCA_True/'
    AWSpath = f"Hold_Out_Year/Daily/"
    file_types = ['.h5', '.pkl']
    for file_type in file_types:
        Hindcast_Initialization.Hindcast_to_AWS(modelname, folderpath, AWSpath, file_type)

    if retries > 0:
        sweml_hindcast(new_year, threshold, Region_list, fSCA, frequency, NewSim, single_day)


if __name__ == "__main__":
    new_year = 2023
    threshold = 10
    Region_list = ['N_Sierras', 'S_Sierras_High', 'S_Sierras_Low', 'Greater_Yellowstone',
                   'N_Co_Rockies', 'SW_Mont', 'SW_Co_Rockies', 'GBasin', 'N_Wasatch', 'N_Cascade',
                   'S_Wasatch', 'SW_Mtns', 'E_WA_N_Id_W_Mont', 'S_Wyoming', 'SE_Co_Rockies',
                   'Sawtooth', 'Ca_Coast', 'E_Or', 'N_Yellowstone', 'S_Cascade', 'Wa_Coast',
                   'Greater_Glacier', 'Or_Coast'
                   ]
    fSCA = True
    frequency = 'Daily'
    NewSim = True
    single_day = True

    sweml_hindcast(new_year, threshold, Region_list, fSCA, frequency, NewSim, single_day)
