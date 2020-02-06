import os
import numpy as np
import xarray as xr
import geopandas as gpd

import hydrate as hyd
from hydrate.models import Vic


tnv = gpd.read_file('./tnvalley_subbasin.geojson')

session = hyd.authorization.getSession(
    '../../credentials/ee-sandbox-credentials.json')

model = Vic(name='mekong', startTime='2015-01-01', endTime='2015-12-31',
            resolution=0.1, region=tnv, verbose=True, outputPath="/home/vic_test/")


# get precipitation time series for the basin
chirps = hyd.Forcing(session, 'earthengine-public',
                    'UCSB-CHG/CHIRPS/DAILY', bands=['precipitation'])
precip = chirps.getModelDomain(model, verbose=True, maxWorkers=2).rename({
    'precipitation': 'PREC'})

# get landcover dataset for the basin
lc = hyd.Parameter(session, "earthengine-public",
                  "MODIS/006/MCD12Q1", bands=['LC_Type1'])
basinLc = lc.getModelDomain(model, scaleFactor=10).rename({
    'LC_Type1': 'landcover'})


soils = hyd.Parameter(session, "earthengine-public",
                     "OpenLandMap/SOL/SOL_TEXTURE-CLASS_USDA-TT_M/v02",bands=['b0','b30','b100'])
basinSoil = soils.getModelDomain(model)

dem = hyd.Parameter(session, "earthengine-public", "WWF/HydroSHEDS/03CONDEM")
elv = dem.getModelDomain(model, scaleFactor=10).rename({"b1": "elevation"})


# specify meteorological dataset bands to fetch
metBands = ['maximum_2m_air_temperature',
            'minimum_2m_air_temperature',
            'u_component_of_wind_10m',
            'v_component_of_wind_10m']
newMetBands = ['TMAX', 'TMIN', 'uWIND', 'vWIND']  # new bands to rename
# get the meteorological dataset for the basin
era5 = hyd.Forcing(session, 'earthengine-public',
                  'ECMWF/ERA5/DAILY', bands=metBands)
met = era5.getModelDomain(model, verbose=True, maxWorkers=2).rename(
    {metBands[i]: newMetBands[i] for i in range(len(metBands))})

# merge forcing data together and compute additional variables/correct units
forcings = xr.merge([precip, met])
forcings["WIND"] = xr.ufuncs.sqrt((forcings["uWIND"] + forcings["vWIND"])**2)
forcings["TMAX"] = forcings["TMAX"] - 273.15 # K to Celcius
forcings["TMIN"] = forcings["TMIN"] - 273.15 # K to Celcius


lai = hyd.Parameter(session, "earthengine-legacy",
                   "users/kelmarkert/public/lai_climatology_2002_2019")
basinLai = lai.getModelDomain(model, scaleFactor=10)

alb = hyd.Parameter(session, "earthengine-legacy",
                   "users/kelmarkert/public/alb_climatology_2002_2019")
basinAlb = alb.getModelDomain(model, scaleFactor=10)

# wrtie the model soil parameter file
model.writeSoilParam(basinSoil,elv,precip,b_val=0.2,Ws_val=0.85,Ds_val=0.00125,s2=0.2,s3=0.7,c=2)
# write model snow parameter file
model.writeSnowParam(elv,interval=500)
# write model vegetation parameter file
model.writeVegParam(basinLc.isel(time=-1),lai=basinLai,alb=basinAlb)
# write model vegetation library
model.writeVegLib(basinLc.isel(time=-1),basinLai,basinAlb)
# write model forcing files to directory
model.writeForcing(forcings,["PREC","TMAX","TMIN","WIND"],maxWorkers=8)
# write model global parameter file
model.writeGlobalParam()
# run the model after setting up all of the parameters
model.execute()
