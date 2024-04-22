# Envionment Setup and Installation

## Install Mamba

We recommend using [mamba](https://mamba.readthedocs.io/en/latest/installation.html) as a drop in replacement for the `conda` package manager. Mamba performs operations in parallel, which is really fast.

## Create a conda/mamba envionment

In your mamba shell, run the following commands to create a new conda/mamba environment (based on an OSX-arm64 env):
```bash
mamba create -n NSW python=3.9.12
mamba activate NSW
```

```bash
mamba install -c anaconda suds-jurko
```

```bash
mamba install geopandas pandas scikit-learn numpy=1.22 pyshp=2.1 matplotlib=3.5 seaborn tensorflow pillow shapely fonttools mamba install rioxarray rasterio earthpy h5py joblib pyproj kiwisolver pytables netCDF4 jupyter tqdm geojson ulmo pystac-client planetary-computer richdem elevation xgboost lightgbm python-graphviz vincent hvplot holoviews bokeh earthengine-api python-dotenv geemap h5netcdf rasterstats 
```

```bash
pip install tables basemap hdfdict hydroeval geetools 
```

> If you ever get any errors, try running `mamba install --force-reinstall <package>` to install the package.

## Notes

- Skipped Feature reduction grid search
- Skipped Model training
- Both of them took forever on my Macbook Pro M1 Pro 16GB