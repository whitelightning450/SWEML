import math
import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import CRS, Transformer
import planetary_computer
from pystac_client import Client
import planetary_computer
import gdal
from gdal import gdalconst
from shapely import Point, Polygon
from shapely.geometry import box
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_grids(bounding_box):
    """
    Generates 100m resolution grid cells for low sierras regions uisng bounding box coordinates. Do not run this code if you have the file downloaded
    this might take longer to run this.
    """
    #gdf_shapefile = gpd.read_file(shapefile_path)
    ## Get bounding box coordinates
    #minx, miny, maxx, maxy = gdf_shapefile.total_bounds
    
    minx, miny, maxx, maxy = *bounding_box[0], *bounding_box[1]
    bbox_polygon = box(minx, miny, maxx, maxy)
    
    #buffer the bounds
    minx = minx-1
    maxx = maxx+1
    miny = miny-1
    maxy = maxy+1
    
    # Define the source and target coordinate reference systems
    src_crs = CRS('EPSG:4326')  # WGS84
    
    if -126 < minx < -120:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32610', always_xy=True)
        target_crs = CRS('EPSG:32610') #UTM zone 10
    elif -120 < minx < -114:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32611', always_xy=True)
        target_crs = CRS('EPSG:32611') #UTM zone 11
    elif -114 < minx < -108:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32612', always_xy=True)
        target_crs = CRS('EPSG:32612') #UTM zone 12
    elif -108 < minx < -102:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32613', always_xy=True)
        target_crs = CRS('EPSG:32613') #UTM zone 13
    else:
        # transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        target_crs = CRS('EPSG:3857') #Web Mercator
        
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    
    # Convert the bounding box coordinates to Web Mercator
    minx, miny = transformer.transform(minx, miny)
    maxx, maxy = transformer.transform(maxx, maxy)
    
    # set the grid cell size in meters
    cell_size = 100
    
    # Calculate the number of cells in x and y directions
    num_cells_x = int((maxx-minx)/cell_size)
    num_cells_y = int((maxy-miny)/cell_size)
    
    # Calculate the total grid width and height
    grid_width = num_cells_x*cell_size
    grid_height = num_cells_y*cell_size
    
    # Calculate the offset to center the grid
    offset_x = ((maxx-minx)-grid_width)/2
    offset_y = ((maxy-miny)-grid_height)/2
    
    # Generate latitude and longitude ranges
    lon_range = np.linspace(minx + offset_x, maxx - offset_x, num=num_cells_x)
    lat_range = np.linspace(miny + offset_y, maxy - offset_y, num=num_cells_y)
    
    # Create a grid of points
    points = []
    for lon in lon_range:
        for lat in lat_range:
            points.append((lon, lat))
            
    # Convert the coordinate pairs back to WGS84
    back_transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
    target_coordinates = [back_transformer.transform(lon, lat) for lon, lat in points]
    
    # Create a list of Shapely Point geometries
    coords = [Point(lon, lat) for lon, lat in target_coordinates]
    
    # Create a GeoDataFrame from the points
    gdf_points = gpd.GeoDataFrame(geometry=coords)
    
    # set CRS to WGS84
    gdf_points=gdf_points.set_crs('epsg:4326')
    # Clip the points to the shapefile boundary
    gdf_clipped_points = gpd.clip(gdf_points, bbox_polygon)
    # Specify the output points shapefile path
    output_shapefile = r'/home/vgindi/Provided_Data/low_sierras_points.shp'#######
    # Export the clipped points to a shapefile
    gdf_clipped_points.to_file(output_shapefile)
    print("Regional Grid Created")
    
    #Create Submission format .csv for SWE predictions
    gdf_clipped_points.index.names = ['cell_id']
    Geospatial_df = pd.DataFrame()
    Geospatial_df['lon']= gdf_clipped_points['geometry'].x
    Geospatial_df['lat']= gdf_clipped_points['geometry'].y
    
    ### Begin process to import geospatial features into DF
    min_lon = min(Geospatial_df['lon'])
    min_lat = min(Geospatial_df['lat'])
    
    # Define the source and target coordinate reference systems
    src_crs = CRS('EPSG:4326')  # WGS84
    if -126 < min_lon < -120:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32610', always_xy=True)
        target_crs = CRS('EPSG:32610') #UTM zone 10
    elif -120 < min_lon < -114:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32611', always_xy=True)
        target_crs = CRS('EPSG:32611') #UTM zone 11
    elif -114 < min_lon < -108:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32612', always_xy=True)
        target_crs = CRS('EPSG:32612') #UTM zone 12
    elif -108 < min_lon < -102:
        # transformer = Transformer.from_crs(src_crs, 'epsg:32613', always_xy=True)
        target_crs = CRS('EPSG:32613') #UTM zone 13
    else:
        # transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
        target_crs = CRS('EPSG:3857') #Web Mercator
        
    transformer = Transformer.from_crs(src_crs, target_crs, always_xy=True)
    
    # Convert the bounding box coordinates to Web Mercator
    Geospatial_df['lon_m'], Geospatial_df['lat_m'] = transformer.transform(Geospatial_df['lon'].to_numpy(), Geospatial_df['lat'].to_numpy())
    geocols=['BR_Coord_Long', 'BR_Coord_Lat', 'UR_Coord_Long', 'UR_Coord_Lat',
        'UL_Coord_Long', 'UL_Coord_Lat', 'BL_Coord_Long', 'BL_Coord_Lat']
    
    Geospatial_df = Geospatial_df.reindex(columns=[*Geospatial_df.columns.tolist(), *geocols], fill_value=0)
    Geospatial_df.reset_index(drop=True, inplace=True)
    Geospatial_df = Geospatial_df.assign(BR_Coord_Long=lambda x: x.lon_m + 50,
                        BR_Coord_Lat=lambda x: x.lat_m - 50,
                        UR_Coord_Long=lambda x: x.lon_m + 50,
                        UR_Coord_Lat=lambda x: x.lat_m + 50,
                        UL_Coord_Long=lambda x: x.lon_m - 50,
                        UL_Coord_Lat=lambda x: x.lat_m + 50,
                        BL_Coord_Long=lambda x: x.lon_m - 50,
                        BL_Coord_Lat=lambda x: x.lat_m - 50,)
    
    transformer = Transformer.from_crs(target_crs, src_crs, always_xy=True)
    #Geospatial_df['lon_m'], Geospatial_df['lat_m'] = transformer.transform(Geospatial_df['lon'].to_numpy(), Geospatial_df['lat'].to_numpy())
    Geospatial_df['BR_Coord_Long'], Geospatial_df['BR_Coord_Lat']=transformer.transform(Geospatial_df['BR_Coord_Long'].to_numpy(), Geospatial_df['BR_Coord_Lat'].to_numpy())
    Geospatial_df['UR_Coord_Long'], Geospatial_df['UR_Coord_Lat']=transformer.transform(Geospatial_df['UR_Coord_Long'].to_numpy(), Geospatial_df['UR_Coord_Lat'].to_numpy()) 
    Geospatial_df['UL_Coord_Long'], Geospatial_df['UL_Coord_Lat']=transformer.transform(Geospatial_df['UL_Coord_Long'].to_numpy(), Geospatial_df['UL_Coord_Lat'].to_numpy()) 
    Geospatial_df['BL_Coord_Long'], Geospatial_df['BL_Coord_Lat']=transformer.transform(Geospatial_df['BL_Coord_Long'].to_numpy(), Geospatial_df['BL_Coord_Lat'].to_numpy()) 
    print(Geospatial_df.columns)
    Geospatial_df['cell_id'] = Geospatial_df.apply(lambda x: f"11N_cell_{x['lon']}_{x['lat']}", axis=1)

    client = Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        ignore_conformance=True,
    )

    search = client.search(
                collections=["cop-dem-glo-90"],
                intersects = {
                        "type": "Polygon",
                        "coordinates": [[
                        [min_x, min_y],
                        [max_x, min_y],
                        [max_x, max_y],
                        [min_x, max_y],
                        [min_x, min_y]  
                    ]]})

    tiles = list(search.items())
    regions = pd.DataFrame([(i, tile.id) for i, tile in enumerate(tiles)], columns=['sliceID', 'tileID']).set_index('tileID')

    Geospatial_df['tile_id'] = Geospatial_df.apply(lambda x: 'Copernicus_DSM_COG_30_N' + str(math.floor(x['lat'])) + '_00_W' + str(math.ceil(abs(x['lon']))) + '_00_DEM', axis=1)
    
    regions_dict = regions['sliceID'].to_dict()
    Geospatial_df['index_id'] = Geospatial_df['tile_id'].map(regions_dict)
    Geospatial_df = Geospatial_df.drop(columns = ['tile_id'], axis = 1)
    
    return Geospatial_df[['cell_id', 'lon', 'lat', 'BR_Coord_Long', 'BR_Coord_Lat', 'UR_Coord_Long', 'UR_Coord_Lat',
        'UL_Coord_Long', 'UL_Coord_Lat', 'BL_Coord_Long', 'BL_Coord_Lat', 'index_id']]

if __name__ = "__main__":
    bounding_box = ((-120.3763448720203, 36.29256774541929), (-118.292253412863, 38.994985247736324)) 
    grid_cells_meta = generate_grids(bounding_box)
    grid_cells_meta.to_csv(r'/home/vgindi/Provided_Data/grid_cells_meta.csv')