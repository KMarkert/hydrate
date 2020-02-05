from __future__ import absolute_import

import os
import time
import json
import pickle
import datetime
import random
import string
from tqdm import tqdm
from io import BytesIO
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor

import geojson
import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage
import geopandas as gpd
from shapely import geometry, wkt
from rasterstats import zonal_stats
from affine import Affine
from rasterio import features
from pysheds.grid import Grid
from pyproj import Proj

from archhydro import utils

ERROR_CODES = {400: "Bad request", 403: "Forbidden", 404: "Not found",
               429: "Too many requests", 500: "Internal server error", }


class Model(object):
    def __init__(self, name=None,
                 startTime='2000-01-01',
                 endTime='2001-12-31',
                 timestep='1D',
                 resolution=0,
                 region=None,
                 outputPath=None,
                 verbose=False,
                 maxWorkers=1):
        '''

        '''
        if name is not None:
            self.name = name
        else:
            self.name = ''.join(
                [random.choice(string.ascii_letters + string.digits) for n in range(6)])

        if outputPath is not None and os.path.exists(outputPath) is False:
            os.mkdir(outputPath)

        if outputPath is not None:
            self.path = outputPath
        else:
            raise ValueError('outputPath keyword needs to be explicitly set')

        if resolution <= 0:
            raise ValueError(
                'resolution keyword needs to defined as a positive value')
        else:
            self.res = resolution

        if type(region) != gpd.geodataframe.GeoDataFrame:
            try:
                region_gdf = gpd.read_file(region)
                self.basins = region_gdf
            except exception as e:
                raise ValueError(
                    'region keyword parameter should be a geopandas geodataframe or file read in by geopandas', e)
        else:
            self.basins = region

        if "1D" == timestep or "H" in timestep:
            self.step = timestep
        else:
            raise ValueError(
                f"Model timestep has to be at most 1D...{timestep} was provided")

        self.start = startTime
        self.end = endTime

        self.extent = self.roundOut(region.total_bounds)
        self.box = gpd.GeoDataFrame(pd.DataFrame({'index': [0], 'name': name, 'coordinates': [geometry.box(*self.extent)]}),
                                    geometry='coordinates')
        minx, miny, maxx, maxy = self.extent

        self.initPt = (minx, maxy)

        self.xx, self.yy = self.build_grid(self.extent, resolution)
        self.dims = self.xx.shape[::-1]  # domain dimensions X,Y

        self.gt = Affine.from_gdal(
            *(minx, resolution, 0, maxy, 0, -resolution))

        self.grid = features.geometry_mask(
            self.basins.geometry, self.dims[::-1], transform=self.gt, all_touched=True, invert=True)

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

    def roundOut(self, bb):
        minx = bb[0] - (bb[0] % self.res)
        miny = bb[1] - (bb[1] % self.res)
        maxx = bb[2] + (self.res - (bb[2] % self.res))
        maxy = bb[3] + (self.res - (bb[3] % self.res))
        return minx, miny, maxx, maxy

    def timeSlice(self, time, columns=None):
        return getTimeSlice(self.basins, time, columns)

    def _terrain(self, elv, variable="elevation",):
        ratio = elvDim = [elv.dims['lon'], elv.dims["lat"]]
        # get ratio of high resoltion to low resolution
        ratio = [elvDim[i] / self.dims[i] for i in range(2)]
        xres, yres = [self.res / r for r in ratio]
        elvVals = np.squeeze(elv[variable].values)
        compute_crs = Proj("EPSG:6933")
        grid = Grid()
        grid.add_gridded_data(data=elvVals,
                              data_name='dem',
                              affine=Affine.from_gdal(
                                  *(self.initPt[0], xres, 0, self.initPt[1], 0, -yres)),
                              crs=Proj("EPSG:4326"),
                              nodata=-9999)
        grid.mask = ~np.isnan(elvVals)
        grid.fill_depressions(data='dem', out_name='flooded_dem')
        grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
        grid.flowdir(data='inflated_dem', out_name='dir', routing='d8')
        grid.cell_dh(fdir='dir', dem="inflated_dem")

        y, x = utils.dd2meters((self.yy, self.xx), scale=yres)

        x = ndimage.zoom(x, zoom=ratio[0], order=0)
        y = ndimage.zoom(y, zoom=ratio[1], order=0)

        dist = utils.gridDistance(x, y, np.array(grid.dir))

        slope = ((np.array(grid.dh) / dist) * 100)
        slope[np.isnan(elvVals)] = np.nan
        elv["slope"] = (('time', 'lat', 'lon'), slope[np.newaxis, :, :])

        return elv

    # why is this a static method???
    # is `build_grid` used anywhere???
    @staticmethod
    def build_grid(bb, resolution):
        minx, miny, maxx, maxy = bb
        yCoords = np.arange(miny, maxy + resolution, resolution)[::-1]
        xCoords = np.arange(minx, maxx + resolution, resolution)

        return np.meshgrid(xCoords, yCoords)


class Distributed(Model):
    def __init__(self, *args, **kwargs):
        super(Distributed, self).__init__(*args, **kwargs)

        return


class Lumped(Model):
    def __init__(self, *args, **kwargs):
        super(Lumped, self).__init__(*args, **kwargs)
        return

    def setLumpedForcing(self, timeSeries, keys=None, how='space', reducer='mean'):
        """
        resets the `basins` property with a same geopandas dataframe but with
        forcing time series for each of the keys provided
        """
        if keys == None:
            keys = list(timeSeries.data_vars.keys())

        args = [[i, j] for i in keys for j in range(timeSeries.dims['t'])]

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
            for i, key in enumerate(keys):
                full = []
                for j in range(len(self.basins)):
                    series = []
                    for t in zs[i]:
                        series.append(t[j][reducer])
                    full.append(
                        {key: pd.Series(np.array(series), index=timeSeries.coords['t'].values)})
                out = out.join(pd.DataFrame(full))

        elif how == 'time':
            raise NotImplementedError(
                'the selected aggregation in not implemented')
        else:
            raise NotImplementedError(
                'the selected aggregation in not implemented')

        self.basins = out

        return


class Dataset(object):
    """
    Basic level of dataset that interfaces with the EE API
    Methods:
        -
    """

    def __init__(self, session, project, collection,):
        self._PROJECT = project
        self._COLLECTION = collection
        self._NAME = f"projects/{project}/assets/{collection}"
        self._BASEURL = f"https://earthengine.googleapis.com/v1alpha/{self._NAME}"
        self._SESSION = session
        return

    def listImages(self, startTime=None, endTime=None, region=None, maxWorkers=8):
        t1 = datetime.datetime.strptime(startTime, '%Y-%m-%d')
        t2 = datetime.datetime.strptime(
            endTime, '%Y-%m-%d') + datetime.timedelta(1)

        dt = t2 - t1
        if dt.days > 50:
            iters = int(np.ceil(dt.days / 50))
            ts = [t1 + datetime.timedelta(50 * i) for i in range(iters)]
            if (t2 - ts[-1]).days > 0:
                ts[-1] = t2
        else:
            ts = [t1, t2]
            iters = 2

        results, payloads = [], []
        url = f'{self._BASEURL}:listImages'

        for i in range(1, iters):
            tStrStart = ts[i - 1].strftime('%Y-%m-%d')
            tStrEnd = ts[i].strftime('%Y-%m-%d')
            payloads.append({
                'startTime': f"{tStrStart}T00:00:00.000Z",
                'endTime': f"{tStrEnd}T00:00:00.000Z",
                'region': json.dumps(region),
            })

        with ThreadPoolExecutor(maxWorkers) as executor:
            responses = list(executor.map(
                lambda x: self._SESSION.get(url, params=x), payloads))

        for response in responses:

            if response.status_code is 200:
                content = response.json()
                if "images" in content.keys():
                    assetList = content['images']
                    results = results + assetList
            else:
                code = response.status_code
                raise ValueError(
                    f"Server returned a bad status of {code}: {ERROR_CODES[code]}")

        return results

    def getMetadata(self, id=None):
        if id is not None:
            imgId = id.split('/')[-1]
            response = self._SESSION.get(f'{self._BASEURL}/{imgId}')
        else:
            response = self._SESSION.get(self._BASEURL)
        if response.status_code is not 200:
            code = response.status_code
            raise ValueError(
                f"Server returned a bad status of {code}:{ERROR_CODES[code]}")
        asset = response.json()
        return asset

    def getBands(self, id=None):
        asset = self.getMetadata(id)
        return [str(band['id']) for band in asset['bands']]

    def getPixels(self, id=None, resolution=0.05, bands=None, initPt=[-180, 85], dims=[256, 256], maxRetries=5):
        if bands == None:
            bands = self.getBands(id)

        if id is not None:
            imgId = id.split('/')[-1]
            url = f"{self._BASEURL}/{imgId}:getPixels"
        else:
            url = f"{self._BASEURL}:getPixels"

        payload = json.dumps({
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
                'dimensions': {'width': dims[0], 'height': dims[1]},
            },
        })
        retries = 1
        while retries <= maxRetries:
            response = self._SESSION.post(url, data=payload)

            if response.status_code is not 200:
                t = utils.exponentialBackoff(retries)
                time.sleep(t)
                retries += 1
            else:
                break

        if response.status_code is not 200:
            code = response.status_code
            raise ValueError(
                f"Server returned a bad status of {code}: {ERROR_CODES[code]}")

        result = np.load(BytesIO(response.content))

        return result

    def getSeries(self, resolution=0.05, startTime=None, endTime=None,
                  bands=None, initPt=[-180, 85], dims=[256, 256], maxWorkers=8,
                  verbose=False):
        """

        arguments:
            resolution: spatial resolution in decimal degrees
            startTime:
            endTime:
            bands: bands to request (default == all bands from dataset)
            initPt: upper left beginning point of dataset
            dims: number of pixels for dataset [X,Y]

        """

        if maxWorkers <= 0:
            maxWorkers = 1

        gt = Affine.from_gdal(
            *[initPt[0], resolution, 0., initPt[1], 0., -resolution])

        halfRes = resolution / 2

        xmin, ymax= [np.around(v,decimals=6) for v in gt * (0, 0)]
        xmax, ymin = [np.around(v,decimals=6) for v in gt * dims]


        box = [(xmin, ymin), (xmin, ymax), (xmax, ymax),
               (xmax, ymin), (xmin, ymin)]
        region = geojson.Polygon([box])

        try:
            images = self.listImages(startTime, endTime, region)
            if bands == None:
                bands = self.getBands(id=images[0]['id'])

            bandMetadata = self.getMetadata(id=images[0]['id'])

        except:
            bandMetadata = self.getMetadata()
            images = [bandMetadata]
            if bands == None:
                bands = self.getBands()

        dates, series, ids = [], [], []

        referencet = pd.to_datetime('1970-01-01T00:00:00Z')

        ids = []
        for i, img in enumerate(images):
            dates.append(pd.to_datetime(img['startTime'][:-1]))
            imgId = None if img['id'] == self._COLLECTION else img['id']
            ids.append(imgId)

        seriesFunc = lambda x: self.getPixels(id=x, bands=bands, resolution=resolution,
                                                    initPt=initPt, dims=dims)

        if len(ids) < maxWorkers:
            series = list(map(seriesFunc, ids))

        else:
            with ThreadPoolExecutor(maxWorkers) as executor:
                gen = executor.map(seriesFunc,ids)
                if verbose:
                    series = list(tqdm(gen, total=len(ids),
                                       desc=f"{self._COLLECTION} progress"))
                else:
                    series = list(gen)

        xrDims = list(series[0][bands[0]].shape)
        xrDims.append(len(series))

        xx = np.linspace(xmin+halfRes, xmax-halfRes, dims[0])
        yy = np.linspace(ymin+halfRes, ymax-halfRes, dims[1])

        # TODO: set better metadata/attributes on the output dataset
        # include geo2d attributes

        df = {'time': {'dims': ('time'), 'data': dates,
                      'attrs': {'unit': 'day', 'calendar': "gregorian"}},
              'lon': {'dims': ('lon'), 'data': xx,
                      'attrs': {'long_name': "longitude", 'units': "degrees_east"}},
              'lat': {'dims': ('lat'), 'data': yy[::-1],
                      'attrs': {'long_name': "latitude", 'units': "degrees_north"}}
              }

        for i in range(len(series)):
            for j in bands:
                if i == 0:
                    df[j] = {'dims': ('lat', 'lon', 'time'),
                             'data': np.zeros(xrDims)}
                df[j]['data'][:, :, i] = series[i][j][:, :]

        outDs = xr.Dataset.from_dict(df)

        attrs = self.getMetadata()
        props = attrs.pop('properties')
        attrs.update(props)
        outDs.attrs = attrs

        return outDs


def getTimeSlice(gdf, time, columns=None):
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
    with open(filname, 'rb') as input:
        model = pickle.load(input)
    return model
