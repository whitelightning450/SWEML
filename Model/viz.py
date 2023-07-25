import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.animation as animation
from IPython.display import HTML

def plot_single_date(cwd, area, date):
    """
    Create a single matplotlib plot of a single date model inference
        Args: 
            cwd (str): current working diectory. This should be the "Model" directory.
            area (str): Name of the region to model.
            date (str): 'YYYY-MM-DD' - Date of model infrence to plot.
    """
    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'\\'+area+'\Predictions\submission_format_'+area+'_'+date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'\\'+area+'\\'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'\\'+area+'\\'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")

    date=date
    fig, ax = plt.subplots(figsize=(6, 6))
    preds_gdf.plot(ax=ax,column=date, marker='s',markersize =1, legend=True)
    gdf_shapefile.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)

    # Set plot labels and title
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(date+' Estimated SWE [in]')

    # Display the plot
    plt.show()


def plot_multiple_dates(cwd, area, end_date, num_subplots):
    """
    Create a matplotlib subplot grid of multiple date model inference
        Args: 
            cwd (str): current working diectory. This should be the "Model" directory.
            area (str): Name of the region to model.
            end_date (str): 'YYYY-MM-DD' - Date of final model infrence to plot.
            num_subplots (int): Number of model inferences to plot. 
    """
     
    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'\\'+area+'\Predictions\submission_format_'+area+'_'+end_date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'\\'+area+'\\'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'\\'+area+'\\'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")

    # Calculate the number of rows and columns for the subplot layout
    num_rows = int(np.sqrt(num_subplots))
    num_cols = int(np.ceil(num_subplots / num_rows))

    # Create a figure and axes for the subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18,24))

    # Create a normalization instance for color mapping across subplots
    cmap = plt.cm.get_cmap('BuPu')  # Choose a colormap #viridis
    norm = colors.Normalize(vmin=0, vmax=round(preds_gdf[preds_gdf.columns[4:-1]].max(axis=1).max()+5.1, -1))

    # Iterate over each column and create a subplot for each. To begin on a later date change the starting column slice. 
    for i, column in enumerate(preds_gdf.columns[4:-1]):
        # Calculate the subplot position in the layout
        row = i // num_cols
        col = i % num_cols

        # Select the column data
        column_data = preds_gdf[column]
        
        # Plot the column data in the corresponding subplot
        preds_gdf.plot(ax=axes[row, col],column=column_data, marker='s', markersize=.5, cmap=cmap, norm=norm)
        gdf_shapefile.plot(ax=axes[row, col], facecolor='none', edgecolor='black', linewidth=1)
        
        # Set the subplot title to the column name
        axes[row, col].set_title(column)

        # Add x and y-axis labels for the top-left subplot
        axes[row, col].xaxis.set_tick_params(labelbottom=False)
        axes[row, col].yaxis.set_tick_params(labelleft=False)
        if row == 0 and col == 0:
            axes[row, col].xaxis.set_tick_params(labelbottom=True, labelrotation=45)
            axes[row, col].yaxis.set_tick_params(labelleft=True)

    #remove empty plot at end
    axes.flat[-2].set_visible(False)
    axes.flat[-1].set_visible(False)

    # fig = axes.get_figure()
    cax = fig.add_axes([0.001, 0.3, 0.007, 0.4])  # Adjust the position and size of the colorbar axis
    # Create a colorbar for the entire plot
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label('SWE (in)')
    cbar.ax.set_title('SWE (in)', fontsize=10)

    # sm._A = []
    # fig.colorbar(sm, cax=cax)

    # Adjust the spacing between subplots
    plt.tight_layout()
    # plt.savefig(r'C:\Users\Dane Liljestrand\Box\Snow_Modeling_Group\NSM_App_Deployment\National-Snow-Model\Predictions\UCRB\UCRB_SWE_2021.png')

    plt.show()

@staticmethod
def animate_SWE(cwd, area, end_date):
    """
    Create a matplotlib animation of multiple date model inferences.
        Args: 
            cwd (str): current working diectory. This should be the "Model" directory.
            area (str): Name of the region to model.
            end_date (str): 'YYYY-MM-DD' - Date of final model infrence to plot.
    """

    #load SWE predictions up to the end_date
    preds=pd.read_csv(cwd+'\\'+area+'\Predictions\submission_format_'+area+'_'+end_date+'.csv', index_col = [0])
    preds.index.names=['cell_id']

    #load feature dataframe & area shapefile
    Geo_df = pd.read_csv(cwd+'\\'+area+'\\'+area+'_Geo_df.csv', index_col='cell_id')
    shapefile_path = cwd+'\\'+area+'\\'+area+'.shp'
    gdf_shapefile = gpd.read_file(shapefile_path)

    # Make Geodataframe for plotting
    preds=Geo_df.join(preds)
    preds_gdf = gpd.GeoDataFrame(preds, geometry=gpd.points_from_xy(preds.Long, preds.Lat), crs="EPSG:4326")
    
    # Set up the figure and subplots
    fig, ax = plt.subplots(figsize=(8,12))

    # Get the column names with the dates. To begin on a later date change the starting column slice.
    date_columns = preds_gdf.columns[4:-1].tolist()

    # Define the update function for each frame
    def update(frame):
        ax.clear()
        # Get the column name for the current frame
        date_column = '{}'.format(frame)
        # Filter the data for the current frame
        # current_data = preds_gdf[[date_column, 'geometry']]
        # Plot the current frame
        preds_gdf.plot(ax=ax,column=date_column, marker='s', markersize=1, cmap=cmap_ani, norm=norm)
        gdf_shapefile.plot(ax=ax, facecolor='none', edgecolor='black', linewidth=1)
        # current_data.plot(ax=ax)
        ax.set_title('{}'.format(date_column))

    # Create a normalization instance for color mapping
    cmap_ani = plt.cm.get_cmap('BuPu')  # Choose a colormap #'Blues' 'BuPu' 'jet'
    norm = colors.Normalize(vmin=0, vmax=round(preds_gdf[preds_gdf.columns[4:-1]].max(axis=1).max()+5.1, -1))
    cax = fig.add_axes([0.935, 0.3, 0.01, 0.4])  # Adjust the position and size of the colorbar axis
    # Create a colorbar for the entire plot
    sm = plt.cm.ScalarMappable(cmap=cmap_ani, norm=norm)
    cbar = plt.colorbar(sm, cax=cax)
    # cbar.set_label('SWE (in)')
    cbar.ax.set_title('Est. SWE (in)', fontsize=10)

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=date_columns, interval=1000)
    plt.tight_layout()

    # save animation - may need to conda install ffmpeg, pip install might not work
    # f = 'out_path'
    # writervideo = animation.FFMpegWriter(fps=10) 
    # ani.save(f, writer=writervideo)

    # Show the animation
    # fig.show()
    plt.close()
    out = HTML(ani.to_jshtml())

    return out


