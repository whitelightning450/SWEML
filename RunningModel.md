![NSM_Cover](./Images/ML_SWE.jpg)

# Snow Water Equivalent Machine Learning (SWEML): Using Machine Learning to Advance Snow State Modeling

[![Deploy](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/use_case_template)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geo-smart/use_case_template/HEAD?urlpath=lab)
[![GeoSMART Use Case](./book/img/use_case_badge.svg)](https://geo-smart.github.io/usecases)
![GitHub](https://img.shields.io/github/license/whitelightning450/National-ML-Snow-Prediction-Mod?logo=GitHub&style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Jupyter&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Github&style=flat-square)


## Model Running Instructions: Making a Model Inference
SWEML supports a flexible ML framework that allows the use and exploration of many ML algorithms. 
The [Model](https://github.com/whitelightning450/SWEML/tree/main/Model) folder exemplifies the model agnostic structure and variety of ML algorithms explored during the development of the model.
We recommend using the [Neural Network](https://github.com/whitelightning450/SWEML/tree/main/Model/Neural_Network) model as it has consistently proven to be the best performing model. 
After completing the [Getting Started](https://github.com/whitelightning450/SWEML/blob/main/Getting%20Started.md) steps to set up the correct packages and versioning, one can begin to explore the model.
Most files are linked to the CIROH AWS S3 folder but can also be made using the files within each directory.
The hindcast simulation is set to the 2019 water year [here](https://github.com/whitelightning450/SWEML/blob/main/Model/Neural_Network/SSM_Hindcast_2019.ipynb), as we pre-compiled all of the necessary information to run and evaluate the model for this year.
The model framework fully supports the use of other years in the 2013-2018 period but will require the user to turn on the Get_Monitoring_Data_Threaded(), Data_Processing(), and augmentPredictionDFs() functions.
The 2019 simulation at a weekly temporal resolution takes approximately 90 seconds on a quality laptop and can quickly exceed 1 hour when running for a different year due to the data acquisition and processing.
