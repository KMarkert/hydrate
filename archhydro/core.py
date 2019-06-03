from __future__ import absolute_import

import os
import time
import json
import pickle
import datetime
import random,string
from io import BytesIO
from urllib.parse import urlencode

import geojson
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely import geometry,wkt
from rasterstats import zonal_stats
from affine import Affine


class model(object):
    def __init__(self,name=None,startTime='2000-01-01',endTime='2001-12-31',timestep='1D',resolution=0,region=None,outputPath=None):
        '''

        '''
        if name != None:
            self.name = name
        else:
            self.name = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(6)])

        if os.path.exists(outputPath) != True:
            os.mkdir(outputPath)

        if outputPath:
            self.path = outputPath
        else:
            raise ValueError('outputPath keyword needs to be explicitly set')

        if resolution <= 0:
            raise ValueError('resolution keyword needs to defined as a positive value')
        else:
            self.res = resolution

        if type(region) != gpd.geodataframe.GeoDataFrame:
            try:
                region_gdf = gpd.read_file(region)
                self.basins = region_gdf
            except exception as e:
                raise ValueError('region keyword parameter should be a geopandas geodataframe or file read in by geopandas',e)
        else:
            self.basins = region

        self.step = timestep

        self.start = startTime
        self.end = endTime

        self.extent = self.roundOut(region.total_bounds)
        self.box = gpd.GeoDataFrame(pd.DataFrame({'index':[0],'name':name,'coordinates':[geometry.box(*self.extent)]}),
                                                      geometry='coordinates')
        minx,miny,maxx,maxy = self.extent

        self.initPt = (minx,maxy)

        self.xx, self.yy = self.build_grid(self.extent,resolution)
        self.dims = self.xx.shape

        self.gt = (minx,resolution,0,maxy,0,-resolution)

        return

    # TODO: create a pretty printout
    # def __repr__(self):
    #     return


    def build_boundingBoxes(self):
        outDf = self.region.copy()

        for i in range(len(outDf)):
            bbgeom = geometry.box(*outDf.iloc[i].geometry.bounds)
            outDf.iloc[i]['geometry'] = bbgeom

        return outDf

    def roundOut(self,bb):
        minx = bb[0] - (bb[0] % self.res)
        miny = bb[1] - (bb[1] % self.res)
        maxx = bb[2] + (self.res - (bb[2] % self.res))
        maxy = bb[3] + (self.res - (bb[3] % self.res))
        return minx,miny,maxx,maxy

    def timeSlice(self,time,columns=None):
        return getTimeSlice(self.basins,time,columns)


    # why is this a static method???
    # is `build_grid` used anywhere???
    @staticmethod
    def build_grid(bb,resolution):
        minx,miny,maxx,maxy = bb
        yCoords = np.arange(miny,maxy+resolution,resolution)[::-1]
        xCoords = np.arange(minx,maxx+resolution,resolution)

        return np.meshgrid(xCoords,yCoords)


    def setLumpedForcing(self,timeSeries,keys=None,how='space',reducer='mean'):
        """
        resets the `basins` property with a same geopandas dataframe but with
        forcing time series for each of the keys provided
        """
        if keys == None:
            keys = list(timeSeries.data_vars.keys())

        args = [[i,j] for i in keys for j in range(timeSeries.dims['t'])]

        zs = list(map(lambda x:
                list(map(lambda t:
                    zonal_stats(self.basins,
                        timeSeries[x].isel(t=t).values,
                        affine=Affine.from_gdal(*(self.gt)),
                        stats=[reducer],
                        nodata=-9999.),
                    range(timeSeries.dims['t']))
                ),
                keys)
              )

        if how == 'space':
            out = self.basins.copy()
            for i,key in enumerate(keys):
                full= []
                for j in range(len(self.basins)):
                    series = []
                    for t in zs[i]:
                        series.append(t[j][reducer])
                    full.append({key:pd.Series(np.array(series),index=timeSeries.coords['t'].values)})
                out = out.join(pd.DataFrame(full))

        elif how == 'time':
            raise NotImplementedError('the selected aggregation in not implemented')
        else:
            raise NotImplementedError('the selected aggregation in not implemented')

        self.basins = out

        return

class distributed(object):
    def __init__(self):
        return

class lumped(object):
    def __init__(self):
        return


def listImages(http,project,collection,startTime=None,endTime=None,region=None):
    name = '{}/assets/{}'.format(project, collection)

    t1 = datetime.datetime.strptime(startTime,'%Y-%m-%d')
    t2 = datetime.datetime.strptime(endTime,'%Y-%m-%d')

    dt = t2-t1
    if dt.days > 100:
        iters = int(np.ceil(dt.days/100))
        ts = [t1+datetime.timedelta(100*i) for i in range(iters)]
        if (t2-ts[-1]).days > 0:
            ts[-1] = t2
    else:
        ts = [t1,t2]
        iters = 2

    results = []

    for i in range(1,iters):
        tStrStart = ts[i-1].strftime('%Y-%m-%d')
        tStrEnd = ts[i].strftime('%Y-%m-%d')
        url = 'https://earthengine.googleapis.com/v1/{}:listImages?{}'.format(name, urlencode({
            'startTime': '{}T00:00:00.000Z'.format(tStrStart),
            'endTime': '{}T00:00:00.000Z'.format(tStrEnd),
            'region': json.dumps(region),
        }))
        (response, content) = http.request(url)

        if response['status'] == '200':
            assetList = json.loads(content)['assets']
            results = results + assetList
        else:
            raise ValueError('Server returned a bad status of {}'.format(response['status']))

    return results


def getMetadata(http,project,id):
    name = '{}/assets/{}'.format(project, id)
    url = 'https://earthengine.googleapis.com/v1/{}'.format(name)
    (response, content) = http.request(url)
    if response['status'] != '200':
        raise ValueError('Server returned a bad status of {}'.format(response['status']))
    asset = json.loads(content)
    return asset

def getBands(http,project,asset):
    asset = getMetadata(http,project,asset)
    return  [str(band['id']) for band in asset['bands']]


def getPixels(http,project,asset,resolution=0.05,bands=None,initPt=[-180,85],dims=[256,256]):
    name = '{}/assets/{}'.format(project, asset)
    if bands == None:
        bands = getBands(http,project,asset)

    url = 'https://earthengine.googleapis.com/v1/{}:getPixels'.format(name)
    body = json.dumps({
        'fileFormat': 'NPY',
        'bandIds': bands,
        'grid': {
            "crsCode": "EPSG:4326",
            'affineTransform': {
                'scaleX': resolution,
                'scaleY': -resolution,
                'translateX': initPt[0],
                'translateY': initPt[1],
            },
            'dimensions': {'width': dims[1], 'height': dims[0]},
        },
    })
    (response, content) = http.request(url, method='POST', body=body)

    if response['status'] != '200':
        raise ValueError('Server returned a bad status of {}'.format(response['status']))

    result = np.load(BytesIO(content))

    return result

def getTimeSeries(http,project,collection,resolution=0.05,startTime=None,endTime=None,
                  bands=None,initPt=[-180,85],dims=[256,256]):

    xmin,ymin = initPt[0], initPt[1] + dims[0]*-resolution
    xmax,ymax = initPt[0] + dims[1]*resolution, initPt[1]
    box = [(xmin,ymin),(xmin,ymax),(xmax,ymax),(xmax,ymin),(xmin,ymin)]
    region = geojson.Polygon([box])

    images = listImages(http,project,collection,startTime,endTime,region)

    if bands == None:
        bands = getBands(http,project,images[0]['id'])

    bandMetadata = getMetadata(http,project,images[0]['id'])

    dates,series = [],[]

    referencet = pd.to_datetime('1970-01-01T00:00:00Z')

    for i,img in enumerate(images):
        t = (pd.to_datetime(img['startTime'])-referencet).total_seconds()
        dates.append(t)
        series.append(getPixels(http,project,img['id'],bands=bands,resolution=resolution,
                                 initPt=initPt,dims=dims))

    dims = list(series[0][bands[0]].shape)
    dims.append(len(series))

    xx = np.arange(xmin,xmax,resolution)
    yy = np.arange(ymin,ymax,resolution)

    # TODO: set better metadata/attributes on the output dataset
    # include geo2d attributes
    df = {'time':{'dims':('time'),'data':dates,
            'attrs':{'unit':'seconds since 1970-01-01','calendar':"proleptic_gregorian"}},
          'lon':{'dims':('lon'),'data':xx,
            'attrs':{'long_name':"longitude",'units':"degrees_east"}},
          'lat':{'dims':('lat'),'data':yy[::-1],
            'attrs':{'long_name':"latitude",'units':"degrees_north"}}
         }

    for i in range(len(series)):
        for j in bands:
            if i == 0:
                df[j] = {'dims':('lat','lon','time'),'data':np.zeros(dims)}
            df[j]['data'][:,:,i] = series[i][j][:,:]


    outDs = xr.Dataset.from_dict(df)

    attrs = getMetadata(http,project,collection)
    props = attrs.pop('properties')
    attrs.update(props)
    outDs.attrs = attrs

    return outDs


def getTimeSlice(gdf,time,columns=None):
    if type(columns) == list:
        outDf = gdf.copy()
        for i in range(len(columns)):
            col = []
            for j in range(len(outDf)):
                col.append(outDf.iloc[j][columns[i]].loc[time])
            outDf[columns[i]] = col
    else:
        raise ValueError('Columns argument requires list')

    return outDf


def save_model(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
    return


def open_model(filename):
    with open(filname,'rb') as input:
        model = pickle.load(input)
    return model
