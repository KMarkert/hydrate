from numba import jit
import numpy as np
from pysheds.grid import Grid
import xarray as xr
from affine import Affine
from pyproj import Proj
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.features import shapes


from .. import core


class Routing(object):
    def __init__(self,extent,flowMethod='d8',gauges=None):
        self.flowMethod = flowMethod
        self.extent = extent
        self.gauges = gauges
        return

    def cascading(model,outflowSeries=None):
        raise NotImplementedError()
        return

    def get_elv(self,http,scaleFactor=1):
        res = 0.002083333333333333 * scaleFactor

        assetId = 'projects/earthengine-public/assets/USGS/GMTED2010'

        minx,miny,maxx,maxy = self.extent
        yy,xx = np.arange(miny,maxy,res),np.arange(minx,maxx,res)
        hiDims = (yy.size,xx.size)
        pt = (self.extent[0],self.extent[-1])

        srtm = core.getPixels(http,assetId,'be75',res,pt,hiDims)

        df = {'x':{'dims':('x'),'data':xx},
              'y':{'dims':('y'),'data':yy[::-1]},
              'dem':{'dims':('y','x'),'data':srtm['be75']}
              }

        attrs = {'gt':(xx.min(),res,0,yy.max(),0,-res),
                 'srs': '+init=epsg:4326 '
                }

        out = xr.Dataset.from_dict(df)
        out.attrs = attrs

        return out


    def streamOrder(self,network,maxDepth=10,maxDownstream=10):
        # set all of the segments in the stream network to an order of 1
        network['streamOrder'] = [1 for i in range(network.shape[0])]

        # loop through each segment and get start and end coordinates
        starts,ends = [],[]
        for i in range(network.shape[0]):
            feature = network.iloc[i] # get feature
            coords = list(zip(*feature['geometry'].xy)) # make coords as list((x,y))
            starts.append(coords[-1]) # get the first coord
            ends.append(coords[0]) # get last coord

        # set start and end node columns with the values
        network['start'],network['end'] = starts, ends

        # loop through the different stream orders
        for i in range(2,maxDepth):
            # get the previous order
            iniMask = network['streamOrder'] == i-1
            # find the unique node coords and frequency of each coord
            node, count = np.unique(network['end'].loc[iniMask],return_counts=True)
            # candidate values for next order are where the nodes come together
            candidates = node[np.where(count>1)] # nodes with freq > 1
            mask = network['start'].isin(candidates) # where the start of segment equals confluence node

            # find where the next order segments drain too
            conMask = (network['start'].isin(network['end'].loc[mask]))
            # propogate next order downstream if it is connected
            for j in range(i*maxDownstream):
                 conMask = (conMask | network['start'].isin(network['end'].loc[conMask]))
            # set next order values where there is a confluence from last order
            # and where the next order drains too
            finalMask = (mask | conMask)
            network.loc[finalMask,'streamOrder'] = i

        return network

    # @classmethod

    def elv_to_streams(self,elv,streamThreshold=10000,minLength=1000,**kwargs):
        dims = elv.dem.shape
        grid = Grid()
        grid.add_gridded_data(data=elv.dem.values, data_name='dem',
                              affine=Affine.from_gdal(*(elv.attrs['gt'])),
                              crs=Proj(elv.attrs['srs']),
                              nodata=-9999)
        grid.fill_depressions(data='dem', out_name='flooded_dem')
        grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
        grid.flowdir(data='inflated_dem', out_name='dir',routing=self.flowMethod)
        grid.accumulation(data='dir', out_name='acc',routing=self.flowMethod)
        row,col= np.unravel_index(grid.acc.argmax(),dims)
        grid.catchment(data=grid.dir, x=col, y=row, out_name='catch',
                       recursionlimit=15000, xytype='index',routing=self.flowMethod)
        grid.mask = np.array(grid.catch > 0)

        row,col= np.unravel_index(grid.acc.argmax(),grid.shape)
        grid.flow_distance(x=col,y=row,data='catch',out_name='dist', xytype='index')


        branches = grid.extract_river_network('catch','acc',threshold=streamThreshold,routing='d8')
        network = gpd.GeoDataFrame.from_features(branches['features'])
        network.crs = {'init' :'epsg:4326'}
        network['length'] = network.to_crs({'init':'epsg:3857'}).length

        zs = zonal_stats(network,grid.dist,
                        affine=grid.affine,
                        stats=['max'],
                        nodata=0)

        maxdist,order = [i['max'] for i in zs],[i for i in range(len(zs))]
        network['maxDist'] = maxdist
        ordered = network.sort_values(by=['maxDist']).reset_index()
        ordered['distOrder'] = order

        orderedStreams = self.streamOrder(ordered,**kwargs)
        # sort the df by stream order and distance from drainage
        ordered = orderedStreams.sort_values(by=['streamOrder','distOrder'],ascending=[True,False]).reset_index()
        # set final order based sorting
        ordered['finalOrder'] = [i for i in range(ordered.shape[0])]

        self.grid = grid

        return ordered


    def elv_to_sheds(self,elv,minArea=25000000,**kwargs):
        base = np.zeros(elv.dem.shape,dtype=np.int16)
        subbasins = []
        ordered = self.elv_to_streams(elv,**kwargs)

        if self.gauges == None:
            n = ordered.shape[0]
        else:
            n = len(self.gauges)

        for i in range(n):
            flip = (n-1) - i
            if self.gauges == None:
                feature = ordered.iloc[flip] # get feature
                coords = list(zip(*feature['geometry'].xy))[0] # make coords as list((x,y))
            else:
                coords = self.gauges[i]

            startx,starty = coords
            # print(row,col)/
            self.grid.catchment(data=self.grid.dir, x=startx, y=starty, out_name='catch',
                           recursionlimit=15000,xytype='label',routing=self.flowMethod)
            if np.any(self.grid.catch>0):
                subbasins.append(np.array(self.grid.catch>0).astype(np.uint8))

        base = np.dstack(subbasins).sum(axis=2).astype(np.int16)

        results = (
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v)
        in enumerate(
            shapes(base, mask=base>0, transform=self.grid.affine)))

        subbasins = gpd.GeoDataFrame.from_features(list(results))
        subbasins.crs = {'init' :'epsg:4326'}
        subbasins['area'] = subbasins.to_crs({'init':'epsg:3857'}).area
        subbasins = subbasins.loc[subbasins['area']>minArea]

        subbasins['shedOrder'] = subbasins['raster_val'].max() - subbasins['raster_val']
        sortedBasins = subbasins.sort_values(by='shedOrder').reset_index()

        return sortedBasins


@jit
def linearReservoir(x_slow,inflow,Rs):
    # Linear reservoir
    x_slow = (1 - Rs) * x_slow + (1 - Rs) * inflow
    outflow = (Rs / (1 - Rs)) * x_slow
    return x_slow,outflow

@jit
def muskingum(inflow):
    raise NotImplementedError()
    return
