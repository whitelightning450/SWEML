![mountain](https://user-images.githubusercontent.com/33735397/155626251-72859ce7-e05a-4282-89be-6871b19ab234.PNG)

# Geosmart Use Case Jupyter Book

[![Deploy](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/use_case_template)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geo-smart/use_case_template/HEAD?urlpath=lab)
[![GeoSMART Use Case](./book/img/use_case_badge.svg)](https://geo-smart.github.io/usecases)
![GitHub](https://img.shields.io/github/license/whitelightning450/National-ML-Snow-Prediction-Mod?logo=GitHub&style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Jupyter&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Github&style=flat-square)

### Deep Learning national scale 1 km resolution SWE prediction model
Snow-derived water is a critical hydrological component for characterizing the quantity of water available for domestic, recreation, agriculture, and power generation in the western United States.
Advancing the efficiency and optimization of these aspects of water resources management requires an enhanced characterization of the snow state variable, particularly the essential global inputs of snow-water-equivalent (SWE), peak SWE, and snowmelt onset for hydrological models.
While physically-based models that characterize the feedbacks and interactions between influencing factors predict SWE values well in homogeneous settings, these models exhibit limitations for CONUS-scale deployment due to challenges attributed to spatial resolution, landscape heterogeneity, and computational intensity. 
Leveraging a collaborative partnership between the Alabama Water Institute (AWI) at the University of Alabama (UA) and the University of Utah (UU), we address these limitations through the National Snow Model (NSM) as a full stack data-driven machine learning (ML) platform with a modular structure to account for the heterogeneity of climate and topographical influences on SWE across the western United States.
This model consists of twenty-three regionally specific sub-models tailored to the unique topography and hydroclimate phenomena in the Western U.S., exhibiting an RMSE less than 8 cm and a coefficient of determination approaching 0.99 on predictions spanning the 2013-2017 training period.
The NSM pipeline assimilates nearly 700 snow telemetry (SNOTEL) and California Data Exchange Center (CDEC) sites and combines with processed lidar-derived terrain features for the prediction of a 1 km x 1 km SWE inference in critical snowsheds under 20 minutes on personal computer.
We complete the full stack product by leveraging the Tethys interface (TBA), supporting the interactive use of the results (i.e., individual to HUC-scale SWE estimates).
With preliminary regional testing performance ranging between 2.5 cm to 8 cm (i.e., RMSE), this model has the potential to advance the snow state variable in hydrological models such as the National Water Model to improve the estimates of peak flow for flood management and low-flows for supply operations. 
 This readme describes the necessary Python dependencies, training sources, and instructions to get the ML model running for near-real-time SWE inference.


## Model Running Instructions: Making a Model Inference

The Wasatch Snow-ML model requires several steps prior to making an inference. 
Below is a high-level overview of the necessary steps, which we go into more detail later.
1. Observations for the prior and current week (i.e. Snotel). For example, to make predictions for January 20th, we need observations for both January 20th and January 13th. 
2. Initial Conditions. The initial conditions model (SWE_Initial_Conditions_Prediction.ipynb) initiates each location’s modeled SWE value, creating a prediction for all locations in the submission file. 
3. Model Spin-up. Model predictions for after the initial conditions (SWE_Prediction.ipynb) up to the current set of observations.
4. Inference. Same as the previous step but for to-date observations.

![InitialRun](https://user-images.githubusercontent.com/33735397/182939128-07ee36a9-ba66-4399-9b2e-b5e574324d66.PNG)

Figure 1. The initial conditions model requires the date to be January 20th, 2022, and the previous date to be January 13th, 2022. This step makes predictions for all locations to begin model spin up. 


## Model Spin-up.
After the completion of the initial conditions predictions, the Wasatch Snow-ML uses the SWE_Prediction.ipynb script which continues to leverage the current and previous week’s ground measures. 
This model continues to require all DataDriven groundmeasures.csv files to be named according to their latest release (e.g., ground_measures_features_02_03_2022.csv for February 3rd, 2022). 
This ensures that the existing script pulls the most to-date observations and processes accordingly. Run this model up to the current period of observation.
A complete run provides a visualization of each region’s SWE, mapped and plotted against elevation as exhibited in Figures 2 and 3. 
The model matches all predictions to the submission_format.csv, and saves all predictions and associated data into the Predictions folder for use as features in future model runs.

![S_Sierras_SWE_elev](https://user-images.githubusercontent.com/33735397/155618058-b0a3fd32-fd46-4815-9dc4-badad48cd892.PNG)

Figure 2. Model spin-up illustrates each region’s predictions. For example, the high elevation sites in the Southern Sierras region demonstrate the greatest SWE from 2,500 m to 3,100 m.

![S_Sierras_SWE_Map](https://user-images.githubusercontent.com/33735397/155616067-30779c28-3f4a-4b09-a54d-7cbf04e91269.PNG)

Figure 3. The Wasatch Snow-ML model illustrates each week’s model prediction over the region of interest.


## Inference. 
The model inference is performed the same as model spin, but with the to-date observations loaded in the SWE_Prediction.ipynb script. 
This model continues to require all DataDriven groundmeasures.csv files to be named according to their latest release (e.g., ground_measures_features_02_10_2022.csv for February 10th, 2022). 
This ensures that the existing script pulls the most to-date observations and processes accordingly.
This step ensures all of the previous week’s observations form inputs in the current week’s inference. 
For example, if making predictions for February 10th, 2022, the date should be  “02_10_2022” and the previous date “02_03_2022”. 
See Figure 4 for an example. This script loads the to-date ground features data (when saved in the appropriate date format), processes the data into model input feature, and makes predictions. 
Model predictions are illustrated similarly to Figures 2 and 3.
The model matches all predictions to the submission_format.csv, and saves all predictions and associated data into the Predictions folder for use as features in the next week’s model run.

![ModelSpinUp](https://user-images.githubusercontent.com/33735397/182938274-81a14af4-7de6-4a65-bb24-d165bfd9c3c4.PNG)

Figure 4. For a prediction run for February 10th, 2022, the current and previous dates should be entered as illustrated.


## Dependencies (versions, environments)
Python: Version 3.8 or later

### Required packages

| os           | ulmo       | pandas             |
|:-----------: | :--------: | :----------------: | 
| io           | shapely    | datetime           |
| re           | rasterio   | matplot.pyplot     |
| copy         | lightgbm   |  numpy             |
| time         | tensorflow |  pystac_client     |
| tables       | platfrom   | planetray_computer |
| xarray       | tqdm       | random             |
| rioxarray    | geopandas  | requests           |
| pyproj       | richdem    | cartopy            |
| h5py         | elevation  | cmocean            |
| mpl_toolkits | hdfdict    | warning            |
| math         | pickle     |  contextily        |
|folium        | branca     |  earthpy           | 
|netCDF4       | osgeo      | requests           |
| warnings     | geojson    | fiona              |
|fiona.crs     |webbrowser  |                    |

## Data Sources (training, inference, where/how used)
### Model Training Data:
Training data for the model was obtained through the drivendata.org online Development Stage data download portal: 
https://www.drivendata.org/competitions/86/competition-reclamation-snow-water-dev/data/

Ground measurements for training were obtained from the provided SNOTEL and CDEC measurement file: ground_measure_features_template.csv

Latitude, Longitude, and Elevation for all measurement locations were obtained from the metadata file: ground_measures_metadata.csv

GeoJSON data for the submission format grid cell delineation were obtained through the grid_cells.geoJSON file. 

SWE training measurements for the submission format grid cells were obtained through the train_labels.csv

Using the above data, a training dataset was produced for the timespan measured in train_labels.csv. 
The submission grid cell ids were identified by latitude and longitude into one of the twenty-three sub-regions. SNOTEL and CDEC measurements were also identified by coordinates and grouped by sub-region. 
Previous SWE and Delta SWE values were derived for each grid cell, and for each ground measurement site, as the previous measured or estimated SWE value at that location, and as the current measure or estimated SWE value - previous measure or estimated SWE value, respectively. 
Aspect and slope angle from the geoJSON data for each gridcell was converted to northness on a scale of -1 to 1. 
The training data is compiled in /Data_Processing_Assimilation/Geoprocessing_and_Training/Data_Training.ipynb  into a dictionary format and saved as a .h5 file (/Data/Model_Calibraition_Data/RegionTrain_Final.h5).

## Model instructions: Training
The Wasatch Snow-ML model calibration scripts are located in the following directory:

Model->Model_Calibration. 

We perform feature selection using recursive feature elimination (RFE) in a tree-based model (light gradient boost model) for both initial conditions(LGBM_Intial_Conditions_Training.ipynb) and thereafter (LGBM_SWE_Training.ipynb) for each of the twenty-three regions, see Figure 5. 
The identified features demonstrated the greatest prediction accuracy in the deep learning model (multi-layered perceptron, MLP). 
The identified features for each region, and for initial and post-initial conditions are saved in opt_features_intial.pkl, and in opt_features_final.pkl, respectively.

![Regions](https://user-images.githubusercontent.com/33735397/155618808-ea8f9cc0-180e-4621-a4c5-142a2adf8621.PNG)

Figure 5. The Wasatch Snow-ML model consists of twenty-three subregions (Southern Sierras consist of lower and high elevations) to create regionally-specific model features.


Each region’s and prediction conditions (initial or thereafter) deep learning model (MLP) uses the same nine layer-node architecture as illustrated in Table 1, with the exception of layer one (Input) which is based on the total number of regionally-specific input features.


| Layer | Layer Number  | Node                |
|:----: | :-----------: |:------------------: |
| Input | 1             | # of input features |
|Hidden | 2             |  128                |
|Hidden | 3             |  128                |
|Hidden | 4             |   64                |
|Hidden | 5             |   64                |
|Hidden | 6             |   32                |
|Hidden | 7             |  16                 |
|Hidden | 8             |  5                  |
| Output| 9             |  1                  |

Table 1.  The initial conditions and after Wasatch Snow-ML models deep learning structure consists of an input layer determined by the number of ideal region-specific features, and the same layer-node for all hidden layers and output the layer.


The model calibration of initial and thereafter conditions uses all of the provided  2013-2017 ground observations (SNOTEL, CDEC) and in-situ observations (1 km lat/long) processed into the “Region_Train_Final.h5” file. 
This file is the result of assimilating Copernicus 90 m data with the provided “Train Features - Ground Measure”, “Train Label”, and associated metadata from the Data_Training.ipynb file in the Data_Processing_Assimilation-> Geoprocessing_and_Training directory. 
Running the respective scripts, MLP_Intitial_Conditions_Training.ipynb or MLP_SWE_Training.ipynb, loads the processed training data, performs a 75-25% training-testing split, loads the ideal features, and saves the scaled feature and target values while running 3,000 epochs of batch size 100 and using an adam Optimizer (1e-4). 
The best model prediction files are saved in their respective folder for later use in prediction. 

We validate model performance on the remaining 25% split using RMSE and the coefficient of determination (R2). While running the calibration script, once each regional model is trained, each regional model makes a prediction on the data not used in training. 
The prediction includes a model summary and a parity plot along with the respective model’s RMSE and R2, see Figure 6 for an illustration. 
Upon calibration completion, the training script produces grouped parity plot and barplot to investigate predictive performance, see Figures 7 and 8, respectively. The calibration model saves the best model for each region. 


## Model Weights

Model weights for the trained initial conditions MLP model, and for the post-initial conditions MLP model can be found in the following files,  

/Model/Model_Calibration/Initial_MLP/Model_Weights_initial.pkl

/Model/Model_Calibration/Prev_MLP/Model_Weights_final.pkl

The model weights are stored in a .pkl file that contains a dictionary of dictionaries, with the two key structures of, Region, model layer. 
For example, the key “N_Sierras”, contains 7 keys ( integers 1 through 7) that each corresponds to a numpy array of the model weights for the respective model layer.


![ModelTraining](https://user-images.githubusercontent.com/33735397/155620067-f0221bf7-6f9f-4e4e-981c-b571027bebf3.PNG)

Figure 6. The calibration model provides a summary of each region’s model and respective model performance.



![PerformanceBAr](https://user-images.githubusercontent.com/33735397/155620107-cdbf3cc2-5d8a-405d-9395-2afcc1060a7c.PNG)

Figure 7.  The barplot illustrates each model’s predictive error over the unseen testing data.


![Performanceparity](https://user-images.githubusercontent.com/33735397/155620143-70445535-9f70-47fb-a4bf-d312b4dd89cc.PNG)

Figure 8.  A parity plot informs on outliers and regional predictive performance. 


# Near-Real-Time Snow-Water-Equivalent Estimation
The current iteration of the NSM makes 20,000 1 km x 1km SWE inferences for select locations throughout the Western U.S.
There is a heavy focus on SWE inferences in the Sierra Nevada mountains, Colorado Rockies, and Wind River Range in Wyoming. 
Once the user inititates Model Run up (i.e., SWE_Initial_Conditions_Prediction), the NSM_SWE_Prediction script makes predictions for the remainder of the water year. 
This script includes a Data_Assimilation() function that retrieves all SWE observations from SNOTEL and CDEC snow monitoring locations for the date of interest (currently set up to make predictions each Thursday).
The Data_Processing() function creates a model-friendly dataframe for each regions to drive each regional ML model (i.e., SWE_Predict()).
Two new functions support further use of the model results, the netCDF() and plot_interactive() functions.
The netCDF() saves the model results in the common netcdf file format for use in other water resources applications.
The plot_interactive() function creates an html file to support the interactive exploration of SWE accross the western U.S. 
This function allows the user to input coordinates of interest to start the map at this location.


![SWEinteractive](https://user-images.githubusercontent.com/33735397/182941234-c56d2a17-03eb-4e28-bddf-476485bf3074.PNG)
Figure 9. The NSM supports an interactive SWE inference interface to explore how SWE changes with location accross the western U.S. 

## Project support through the University of Alabama and the University of Utah
![UU](https://user-images.githubusercontent.com/33735397/155627859-a34f7856-22a5-4376-89ca-a59b70f3692e.PNG)
![UA](https://user-images.githubusercontent.com/33735397/155628209-c4742053-eb29-4d1e-b8e2-c4e354188ca9.PNG)
