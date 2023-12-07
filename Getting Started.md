![NSM_Cover](./Images/ML_SWE.jpg)

# Snow Water Equivalent Machine Learning (SWEML): Using Machine Learning to Advance Snow State Modeling

[![Deploy](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml/badge.svg)](https://github.com/geo-smart/use_case_template/actions/workflows/deploy.yaml)
[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](https://geo-smart.github.io/use_case_template)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/geo-smart/use_case_template/HEAD?urlpath=lab)
[![GeoSMART Use Case](./book/img/use_case_badge.svg)](https://geo-smart.github.io/usecases)
![GitHub](https://img.shields.io/github/license/whitelightning450/National-ML-Snow-Prediction-Mod?logo=GitHub&style=flat-square)
![GitHub top language](https://img.shields.io/github/languages/top/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Jupyter&style=flat-square)
![GitHub repo size](https://img.shields.io/github/repo-size/whitelightning450/National-ML-Snow-Prediction-Mod?logo=Github&style=flat-square)





# Getting Started: 
The first step is to identify a folder location where you would like to work in a development environment.
We suggest a location that will be able to easily access streamflow predictions to make for easy evaluation of your model.
Using the command prompt, change your working directory to this folder and git clone [Snow-Extrapolation](https://github.com/geo-smart/Snow-Extrapolation)

    git clone https://github.com/geo-smart/Snow-Extrapolation


## Virtual Environment
It is a best practice to create a virtual environment when starting a new project, as a virtual environment essentially creates an isolated working copy of Python for a particular project. 
I.e., each environment can have its own dependencies or even its own Python versions.
Creating a Python virtual environment is useful if you need different versions of Python or packages for different projects.
Lastly, a virtual environment keeps things tidy, makes sure your main Python installation stays healthy and supports reproducible and open science.

## Creating Stable CONDA Environment
Go to home directory
```
cd ~
```
Create a envs directory
```
mkdir envs
```
Create .condarc file and link it to a text file
```
touch .condarc

ln -s .condarc condarc.txt
```
Add the below lines to the condarc.txt file
```
# .condarc
envs_dirs:
 - ~/envs
```
Restart your server

### Creating your NSM_env Python Virtual Environment
Since we will be using Jupyter Notebooks for this exercise, we will use the Anaconda command prompt to create our virtual environment. 
In the command line type: 

    conda create -n NSM_env python=3.9.12

For this example, we will be using Python version 3.9.12, specify this version when setting up your new virtual environment.
After Anaconda finishes setting up your NSM_env , activate it using the activate function.

    conda activate NSM_env 

You should now be working in your new NSM_env within the command prompt. 
However, we will want to work in this environment within our Jupyter Notebook and need to create a kernel to connect them.
We begin by installing the **ipykernel** python package:

    pip install --user ipykernel

With the package installed, we can connect the NSM_env to our Python Notebook

    python -m ipykernel install --user --name=NSM_env 

Under contributors, there is a start to finish example to get participants up to speed on the modeling workflow.
To double check you have the correct working environment, open the  up the [Methods](./contributors/NSM_Example/methods.ipynb) file, click the kernel tab on the top toolbar, and select the NSM_env. 
The NSM_env should show up on the top right of the Jupyter Notebook.

![Notebook_env](./contributors/NSM_Example/Images/NSM-Kernel.JPG)


### Loading other Python dependencies
We will now be installing the packages needed to use NSM_env, as well as other tools to accomplish data science tasks.
Enter the following code block in your Anaconda Command Prompt to get the required dependencies with the appropriate versions, note, you must be in the correct working directory:

    pip install -r requirements.txt

### Explore the model through an example

The objective of the project is to optimize the NSM, or SSM in this case.
To do, the next step is to explore the [NSM Example](./contributors/NSM_Example/methods.ipynb).
