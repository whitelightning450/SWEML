## National-ML-Snow-Prediction-Model
Deep Learning national scale 1 km resolution SWE prediction model


Seasonal snow-derived water is a critical component of the water supply in the mountains and connected downstream regions. 
The need for an accurate understanding and characterization of available water in the form of snow-water-equivalent (SWE), peak SWE, and snowmelt onset are essential global inputs for water management efforts. 
Traditional SWE estimation leverages physically-based models that characterize the feedbacks and interactions between influencing factors (i.e., incoming solar radiation, wind speed and direction, precipitation type and quantity, elevation and aspect, etc.). 
While robust physically-based models predict snow depth and corresponding SWE values well in homogeneous settings, these models exhibit limitations in operational conditions in response to spatial resolution, landscape heterogeneity, and the quality of input data. 
In critical watersheds, on-ground snow sampling remains the most accurate methodology for extrapolating key snowpack information. 
However, the resource intensity and the optimization of sampling networks challenge the large-scale application of these methods. 
The most recent advancements in snow modeling align with improvements in machine learning (ML) and artificial intelligence (Ai) to optimally characterize snowpack across various scales and in near real-time. 
Leveraging a collaborative partnership between the Alabama Water Institute (AWI) at the University of Alabama (UA) and the University of Utah (UU), we introduce a hierarchical deep learning model and data assimilation pipeline to address the need for a high-spatial national-scale machine learning SWE prediction model. 
This model consists of twenty-three regionally specific sub-models tailored to the unique topography and hydroclimate phenomena in the Western U.S., exhibiting an RMSE less than 2” and a coefficient of determination approaching 0.99 on predictions spanning the 2013-2017 training period.
 Competing in the U.S. Bureau of Reclamation Snowcast Showdown presents an opportunity to evaluate near-real-time operational modeling performance, establishing the foundation for Snow-ML programs to advance snow-related products within the NOAA-OWP, such as the National Water Model (NWM), to enhance streamflow prediction and related water resources planning and management resources to a broad range of stakeholders. 
 This readme describes the necessary Python dependencies, training sources, and instructions to get the ML model running for near-real-time SWE inference.


## Model Running Instructions: Making a Model Inference

The Wasatch Snow-ML model requires several steps prior to making an inference. 
Below is a high-level overview of the necessary steps, which we go into more detail later.
1. Observations for the prior and current week (i.e. Snotel). For example, to make predictions for January 20th, we need observations for both January 20th and January 13th. 
2. Initial Conditions. The initial conditions model (SWE_Initial_Conditions_Prediction.ipynb) initiates each location’s modeled SWE value, creating a prediction for all locations in the submission file. 
3. Model Spin-up. Model predictions for after the initial conditions (SWE_Prediction.ipynb) up to the current set of observations.
4. Inference. Same as the previous step but for to-date observations.



(https://user-images.githubusercontent.com/33735397/155603114-d0f56d80-7b1a-4899-a567-ca76527c3787.PNG)

Figure 1. The initial conditions model requires the date to be January 20th, 2022, and the previous date to be January 13th, 2022. This step makes predictions for all locations to begin model spin up. 


## Model Spin-up.
After the completion of the initial conditions predictions, the Wasatch Snow-ML uses the SWE_Prediction.ipynb script which continues to leverage the current and previous week’s ground measures. 
This model continues to require all DataDriven groundmeasures.csv files to be named according to their latest release (e.g., ground_measures_features_02_03_2022.csv for February 3rd, 2022). 
This ensures that the existing script pulls the most to-date observations and processes accordingly. Run this model up to the current period of observation.
A complete run provides a visualization of each region’s SWE, mapped and plotted against elevation as exhibited in Figures 2 and 3. 
The model matches all predictions to the submission_format.csv, and saves all predictions and associated data into the Predictions folder for use as features in future model runs.


[S_Sierras_SWE_elev](https://user-images.githubusercontent.com/33735397/155616021-4ead87e0-3b25-4dd6-987f-ee4fbac85ed1.PNG)

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

![PredictionRun](https://user-images.githubusercontent.com/33735397/155616234-f7cec34d-7166-43f7-bc9e-08c7bbe99595.PNG)

Figure 4. For a prediction run for February 10th, 2022, the current and previous dates should be entered as illustrated.


## Dependencies (versions, environments)
Python: Version 3.8 or later

### Required packages

| os           | contextily | pandas             |
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
| math         | pickle     |                    |
