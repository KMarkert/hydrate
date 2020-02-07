from numba import jit
import numpy as np
import xarray as xr
from affine import Affine
from pyproj import Proj
import geopandas as gpd
from rasterstats import zonal_stats
from rasterio.features import shapes


from hydrate import core


class Routing(object):
    def __init__(self,elv,flowMethod='d8',gauges=None):
        self.flowMethod = flowMethod
        self.gauges = gauges
        self.grid = self._set_grid(elv)
        return


    def _set_grid(self,elv,variable='elevation'):
        dims = elv[variable].shape
        grid = Grid()
        xres = float((elv.lon[1]-elv.lon[0]).values)
        yres = float((elv.lat[1]-elv.lat[0]).values)
        xinit = float((elv.lon[0]).values)
        yinit = float((elv.lat[0]).values)
        grid.add_gridded_data(data=np.squeeze(elv[variable].values), data_name='dem',
                              affine=Affine.from_gdal(*(xinit,xres,0,yinit,0,yres)),
                              crs=Proj("EPSG:4326"),
                              nodata=-9999)
        grid.fill_depressions(data='dem', out_name='flooded_dem')
        grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
        grid.flowdir(data='inflated_dem', out_name='dir',routing=self.flowMethod)

        return grid


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


    def elv_to_streams(self,streamThreshold=10000,minLength=1000,**kwargs):
        grid = self.grid
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

        return ordered


    def getWatersheds(self,variable='elevation',minArea=25000000,**kwargs):
        base = np.zeros(self.grid.shape,dtype=np.int16)
        subbasins = []

        if self.gauges == None:
            n = ordered.shape[0]
        else:
            n = len(self.gauges)

        for i in range(n):
            flip = (n-1) - i
            if self.gauges == None:
                ordered = self.elv_to_streams(elv,**kwargs)
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

        return base


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

@jit
def lohmann(uh,runoff,baseflow,gaugeFrac,velocity=1.5,diffusion=800):
    def make_irf(xmask,diff,velo):
        """
        Function for creating the impulse response function (IRF) for a grid cell
        Arguments:
            xmask -- longest path for any location within the grid cell to reach
                     the stream channel [m]
            diff -- diffusion value
            velo -- overland flow velocity value
        Returns:
            irf -- normalized impulse response function for a grid cell
        Keywords:
            diff -- the diffusivity parameter (default 800) [m^2/s]
            velo -- the flow velocity parameter (default 1.5) [m/s]
        """
        step = np.arange(0,48) #number of time steps
        ir = np.zeros(step.size) # blank array for impulse response
        t = 0.0 #time initialization in seconds

        for i in step:
            t = t + 3600. #time step of one day (in seconds)
            pot = (velo * t - xmask)**2 / (4.0 * diff * t) # function name???
            if pot > 69:
                h = 0.0
            else:
                #impulse response function
                h = 1.0/(2.0*np.sqrt(np.pi*diff)) * (xmask / (t**1.5)) * np.exp(-pot)

            ir[i] = h # pass data to array element

        irf = ir / ir.sum() #normalize the output IRF

        return irf

    def make_uh(xmask_val,uh_box,diff,velo):
        """
        Function for creating the unit hydrograph for a grid cell
        Arguments:
            xmask_val -- longest path for any location within the grid cell to reach
                         the stream channel [m]
            uh_box -- the monthly hydrograph (values 0-1)
            diff -- diffusion value
            velo -- overland flow velocity value
        Keywords:
            none
        Returns:
            uh -- UH hydrograph for the grid cell based on the IRF
        """
        irf = make_irf(xmask_val,diff,velo) #get the IRF of the grid cell

        #defining constants and setting up
        le = 48
        ke = 12
        uh_day = 96
        tmax = uh_day * 24
        uh_s = np.zeros(ke+uh_day)
        uh_daily = np.zeros(uh_day)
        fr = np.zeros([tmax,2])
        if uh_box.sum() != 1.:
            #normalize hydrograph if not already
            uh_box = uh_box.astype(np.float) / uh_box.sum()

        for i in range(24):
            fr[i,0] = 1. / 24.

        for t in range(tmax):
            for l in range(le):
                if t-l > 0:
                    fr[t,1] = fr[t,1] + fr[t-l,0] * irf[l]

            tt = int((t+23) / 24.)
            uh_daily[tt-1] = uh_daily[tt-1]+fr[t,1]

        # for each time period find the UH based on the IRF
        for k in range(ke):
            for u in range(uh_day):
                uh_s[k+u-1] = uh_s[k+u-1] + uh_box[k] * uh_daily[u]

        uh = uh_s / uh_s.sum() #normalize the ouput unit hydrograph

        return uh

    lfactor = 1E3 #convert mm to m
    afactor = 1E6 #convert km^2 to m^2
    tfactor = 86400. #convert day to second

    # create blank arrays for ...
    basinUH = np.zeros((idx[0].size,days)) # ...basin wide UH
    gridUH = np.zeros((days+108,108)) # grid based UH

    return
