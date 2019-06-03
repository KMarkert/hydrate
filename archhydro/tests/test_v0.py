import archhydro as ah
import xarray as xr
import geopandas as gpd
from archhydro.models import Hymod


tnv = gpd.read_file('/Users/kmarkert/archhydro/archhydro/tests/tnvalley_subbasin.geojson')


http = ah.authorization.authorize('ee-sandbox@ee-sandbox.iam.gserviceaccount.com',
                                  '/users/kmarkert/credentials/ee-sandbox.json')

# bounds = (-88.7, 34.1, -81.25, 37.25)
bounds = (98,6,108,33)
# rout = ah.models.routing.Routing(bounds,flowMethod='d8')
rout = ah.models.routing.Routing(bounds,gauges=[(105.386,11.905),(105.948,13.538),(105.519,15.323),(104.717,16.531),(104.794,17.398),(102.613,17.928),(100.1499,20.247)]
            ,flowMethod='d8')

elv = rout.get_elv(http,scaleFactor=3)
# sheds = rout.elv_to_sheds(elv,streamThreshold=10000,minLength=5000)
sheds = rout.elv_to_sheds(elv)

model = Hymod(name='tnv',startTime='2010-01-01',endTime='2010-09-01',
              resolution=0.05,region=sheds,outputPath='/users/kmarkert/bvtest')

# tempBands = ['Maximum_temperature_height_above_ground_6_Hour_Interval',
#              'Minimum_temperature_height_above_ground_6_Hour_Interval']
#
# chirps = ah.Forcing('projects/earthengine-public/assets/UCSB-CHG/CHIRPS/DAILY','precipitation')
# cfs = ah.Forcing('projects/earthengine-public/assets/NOAA/CFSV2/FOR6H',tempBands,timestep='1D',reductions=['max','min'])
#
# precip = chirps.getTimeSeries(http,model).rename({'precipitation':'precip'})
# temp = cfs.getTimeSeries(http,model).rename({tempBands[0]:'maxtemp',tempBands[1]:'mintemp'})
# eto = ah.pet.hargreaves(temp.mintemp,temp.maxtemp)
#
# forcings = xr.merge([precip,temp,eto])
#
# forcings = xr.open_dataset('/Users/kmarkert/archhydro/archhydro/tests/mekong_forcings.nc')
#
# model.setLumpedForcing(forcings,keys=['precip','eto'])
# pars = model.get_random_params()
