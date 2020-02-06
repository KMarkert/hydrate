import os
import json
import datetime
import subprocess
import numpy as np
import pandas as pd
import xarray as xr
from io import StringIO
from concurrent.futures import ThreadPoolExecutor

from hydrate import core
from hydrate.models import routing
from hydrate.lookups import *


class Vic(core.Distributed):
    def __init__(self, *args, parallelExecution=False, **kwargs):
        super(Vic, self).__init__(*args, **kwargs)

        globFile = os.path.join(self.path, 'global.param')
        self.globPath = globFile

        soilFile = os.path.join(self.path, 'soil.param')
        self.soilPath = soilFile

        snowFile = os.path.join(self.path, 'snow.param')
        self.snowPath = snowFile

        vegfile = os.path.join(self.path, 'veg.param')
        self.vegPath = vegfile

        veglib = os.path.join(self.path, 'veg.lib')
        self.vegLib = veglib

        forcingPath = os.path.join(self.path, "forcings/")
        self.forcingPath = forcingPath
        if not os.path.exists(forcingPath):
            os.mkdir(forcingPath)

        fluxPath = os.path.join(self.path, "fluxes/")
        self.fluxPath = fluxPath
        if not os.path.exists(fluxPath):
            os.mkdir(fluxPath)

        self.vegParamLai = False
        self.vegParamAlb = False

        self.parallel = parallelExecution


        return

    def writeGlobalParam(self, moistFract=False, fullEnergy=False, frozenSoil=False, maxSnowTemp=1.5, minSnowTemp=-0.5,
                         outputVars=["OUT_PREC","OUT_EVAP","OUT_RUNOFF","OUT_BASEFLOW","OUT_SWE","OUT_SOIL_MOIST"]):

        modStep = 24 if self.step == '1D' else int(self.step[:-1])

        self.outputVars = outputVars

        startDt = datetime.datetime.strptime(self.start, "%Y-%m-%d")
        endDt = datetime.datetime.strptime(self.end, "%Y-%m-%d")

        laiSrc = 'FROM_VEGPARAM' if self.vegParamLai else 'FROM_VEGLIB'
        albSrc = 'FROM_VEGPARAM' if self.vegParamAlb else 'FROM_VEGLIB'

        text = []
        text.append(f'# VIC Model Parameters - 4.2')
        text.append(f'# Simulation Parameters')
        text.append(f'NLAYER\t{self.nLayers}')
        text.append(f'NODES\t10')
        text.append(f'TIME_STEP\t{modStep}')
        text.append(f'SNOW_STEP\t{modStep}')
        text.append(f'STARTYEAR\t{startDt.year}')
        text.append(f'STARTMONTH\t{startDt.month:02d}')
        text.append(f'STARTDAY\t{startDt.day:02d}')
        text.append(f'ENDYEAR\t{endDt.year}')
        text.append(f'ENDMONTH\t{endDt.month:02d}')
        text.append(f'ENDDAY\t{endDt.day:02d}')
        text.append(f'FULL_ENERGY\t{fullEnergy}'.upper())
        text.append(f'FROZEN_SOIL\t{frozenSoil}'.upper())
        text.append(f'# Precip (Rain and Snow) Parameters')
        text.append(f'MAX_SNOW_TEMP\t{maxSnowTemp}')
        text.append(f'MIN_RAIN_TEMP\t{minSnowTemp}')
        text.append(f'# Forcing Files and Parameters')
        text.append(f'FORCING1\t{self.forcingPath+"forcing_"}')
        text.append(f'FORCE_FORMAT\tASCII')
        text.append(f'FORCE_ENDIAN\tLITTLE')
        text.append(f'N_TYPES\t{self.nMetVars}')
        for v in self.metVars:
            text.append(f'FORCE_TYPE\t{v}')
        text.append(f'FORCE_DT\t{modStep}')
        text.append(f'FORCEYEAR\t{startDt.year}')
        text.append(f'FORCEMONTH\t{startDt.month:02d}')
        text.append(f'FORCEDAY\t{startDt.day:02d}')
        text.append(f'FORCEHOUR\t00')
        text.append(f'GRID_DECIMAL\t4')
        text.append(f'WIND_H\t10.0')
        text.append(f'MEASURE_H\t2.0')
        text.append(f'ALMA_INPUT\tFALSE')
        text.append(f'# Land Surface Files and Parameters')
        text.append(f'SOIL\t{self.soilPath}')
        text.append(f'BASEFLOW\tARNO')
        text.append(f'JULY_TAVG_SUPPLIED\tFALSE')
        text.append(f'ORGANIC_FRACT\tFALSE')
        text.append(f'VEGLIB\t{self.vegLib}')
        text.append(f'VEGPARAM\t{self.vegPath}')
        text.append(f'ROOT_ZONES\t3')
        # TRUE = veg lib file contains 12 monthly values of partial vegcover fraction for each veg class, between the LAI and albedo values
        text.append(f'VEGLIB_VEGCOVER\tFALSE')
        # TRUE = veg param file contains LAI information; FALSE = veg param file does NOT contain LAI information
        text.append(f'VEGPARAM_LAI\t{self.vegParamLai}'.upper())
        # TRUE = veg param file contains albedo information; FALSE = veg param file does NOT contain albedo information
        text.append(f'VEGPARAM_ALB\t{self.vegParamAlb}'.upper())
        # TRUE = veg param file contains veg_cover information; FALSE = veg param file does NOT contain veg_cover information
        text.append(f'VEGPARAM_VEGCOVER\tFALSE')
        # FROM_VEGPARAM = read LAI from veg param file; FROM_VEGLIB = read LAI from veg library file
        text.append(f'LAI_SRC\t{laiSrc}')
        # FROM_VEGPARAM = read albedo from veg param file; FROM_VEGLIB = read albedo from veg library file
        text.append(f'ALB_SRC\t{albSrc}')
        # FROM_VEGPARAM = read veg_cover from veg param file; FROM_VEGLIB = read veg_cover from veg library file
        text.append(f'VEGCOVER_SRC\tFROM_VEGLIB')
        # Number of snow bands; if number of snow bands > 1, you must insert the snow band path/file after the number of bands (e.g. SNOW_BAND 5 my_path/my_snow_band_file)
        text.append(f'SNOW_BAND\t{self.maxBands}\t{self.snowPath}')
        text.append(f'# Output Files and Parameters')
        text.append(f'RESULT_DIR\t{self.fluxPath}')
        # Output interval (hours); if 0, OUT_STEP = TIME_STEP
        text.append(f'OUT_STEP\t0')
        text.append(f'SKIPYEAR\t0')   # Number of years of output to omit from the output files
        text.append(f'COMPRESS\tFALSE')   # TRUE = compress input and output files when done
        text.append(f'BINARY_OUTPUT\tFALSE')   # TRUE = binary output files
        text.append(f'ALMA_OUTPUT\tFALSE')   # TRUE = ALMA-format output files; FALSE = standard VIC units
        # TRUE = output soil moisture as volumetric fraction; FALSE = standard VIC units
        text.append(f'MOISTFRACT\t{moistFract}'.upper())
        # TRUE = insert a header at the beginning of each output file; FALSE = no header
        text.append(f'PRT_HEADER\tTRUE')
        # TRUE = write a "snowband" output file, containing band-specific values of snow variables; NOTE: this is ignored if N_OUTFILES is specified below.
        text.append(f'PRT_SNOW_BAND\tFALSE')
        text.append(f'Format:')
        text.append(f'\tN_OUTFILES\t1')
        text.append(f'\tOUTFILE\tflux_\t{len(outputVars)}')
        for v in outputVars:
            text.append(f'\tOUTVAR\t{v}')

        with open(self.globPath, 'w') as f:
            f.write('\n'.join(text))

        return

    def writeSoilParam(self, soils, elv, precip, soilVariables=None, elvVariable="elevation", precipVariable="precip",
                       b_val=None, Ws_val=None, Ds_val=None, s2=None, s3=None, c=None):
        soilLookup = SOILPARAMS["USDA"]

        cnt = 0  # counter

        elvDim = [elv.dims['lon'], elv.dims["lat"]]
        # get ratio of high resoltion to low resolution
        ratio = [elvDim[i] / self.dims[i] for i in range(2)]

        terrain = self._terrain(elv)
        lowResTerrain = terrain.coarsen(
            lon=int(ratio[0])).mean().coarsen(lat=int(ratio[1])).mean()

        slope = np.squeeze(lowResTerrain["slope"].values)
        elev = np.squeeze(lowResTerrain[elvVariable].values)
        elev[elev < 0] = 0

        # prep input data
        annual_precip = np.squeeze(precip.groupby(
            'time.year').sum().mean(dim="year").to_array()).values

        gridLats = precip.lat.values
        gridLons = precip.lon.values

        soilLayers = np.squeeze(soils.to_array()).values
        soilLayers[np.isnan(soilLayers)] = 255
        nLayers = soilLayers.shape[0]
        if nLayers > 3:
            raise NotImplementedError(
                f"Number of soil layers input ({nLayers}) currently not implemented, max layers = 3")
        elif nLayers <= 1:
            # need to add warning and stack at the first layer on itself to at least len==2
            pass

        depths = [0.1, s2, s3]

        # open soil parameter file for writing
        with open(self.soilPath, 'w') as f:
            cnt = 1
            # loop over each pixel in the model grid
            for i in range(self.dims[1]):
                for j in range(self.dims[0]):
                    if self.grid[i, j] == True:
                        line = []
                        # start passing information to simple variables
                        line.append("1")
                        line.append(f"{cnt}")  # grid cell id
                        line.append(f"{gridLats[i]:.4f}")  # latitude
                        line.append(f"{gridLons[j]:.4f}")  # longitude
                        # variable infiltration curve parameter
                        line.append(f"{b_val:.4f}")
                        line.append(f"{Ds_val:.4f}")  # Ds value
                        Dsmax = (slope[i, j] / 100.) * (
                            float(soilLookup[soilLayers[-1, i, j]]['SatHydraulicCapacity']) * 240)
                        line.append(f"{Dsmax:.4f}")  # Dsmax value
                        line.append(f"{Ws_val:.4f}")  # Ws value
                        line.append(f"{c:.4f}")  # exponent in baseflow curve
                        for l in range(nLayers):
                            # exponent value per layer
                            line.append(
                                f"{3+(2*float(soilLookup[soilLayers[l,i,j]]['SlopeRCurve'])):.4f}")
                        for l in range(nLayers):
                            # layer Ksat value
                            line.append(
                                f"{float(soilLookup[soilLayers[l,i,j]]['SatHydraulicCapacity'])*240:.4f}")
                        for l in range(nLayers):
                            line.append(f"{-999}")  # fill value
                        for l in range(nLayers):
                            soil_den = 2650. if l == 0 else 2685.
                            bulk_den = soilLookup[soilLayers[l,
                                                             i, j]]['BulkDensity']
                            depth = depths[l]
                            # layer inital moisture conditions
                            line.append(
                                f"{(bulk_den / soil_den) * depth *1000:.4f}")
                        # average elevation of gridcell
                        line.append(f"{elev[i,j]:.4f}")
                        for l in range(nLayers):
                            line.append(f"{depths[l]}")  # layer soil depth
                        line.append(f"{27}")  # average temperature of soil
                        # depth that soil temp does not change
                        line.append(f"{4}")
                        for l in range(nLayers):
                            # layer bubbling pressure
                            line.append(
                                f"{float(soilLookup[soilLayers[l,i,j]]['BubblingPressure']):.4f}")
                        for l in range(nLayers):
                            # layer percent quartz
                            line.append(
                                f"{float(soilLookup[soilLayers[l,i,j]]['Quartz']):.4f}")
                        for l in range(nLayers):
                            # layer bulk density
                            line.append(
                                f"{float(soilLookup[soilLayers[l,i,j]]['BulkDensity']):.4f}")
                        for l in range(nLayers):
                            soil_den = 2650. if l == 0 else 2685.
                            # layer soil density
                            line.append(f"{soil_den:.4f}")
                        # time zone offset from GMT
                        line.append(f"{self.xx[i,j] * 24 / 360.:.4f}")
                        for l in range(nLayers):
                            wrc_frac = (float(soilLookup[soilLayers[l, i, j]]['FieldCapacity']) /
                                        float(soilLookup[soilLayers[l, i, j]]['Porosity']))  # layer critical point
                            line.append(f"{wrc_frac}")
                        for l in range(nLayers):
                            wpwp_frac = (float(soilLookup[soilLayers[l, i, j]]['WiltingPoint']) /
                                         float(soilLookup[soilLayers[l, i, j]]['Porosity']))  # layer wilting point
                            line.append(f"{wpwp_frac}")
                        # bare soil roughness coefficient
                        line.append(f"{0.01}")
                        line.append(f"{0.001}")  # snow roughness coefficient
                        # climotological average precipitation
                        line.append(f"{annual_precip[i,j]:.4f}")
                        for l in range(nLayers):
                            # layer residual moisture
                            line.append(
                                f"{soilLookup[soilLayers[l,i,j]]['Residual']:.4f}")
                        # boolean value to run frozen soil algorithm
                        line.append(f"{1}")

                        text = '\t'.join(line) + "\n"
                        f.write(text)
                        cnt += 1

        self.nLayers = nLayers

        return

    def writeSnowParam(self, elev, variable="elevation", interval=250):
        # Filter warnings due to invalid values
        np.warnings.filterwarnings(action='ignore', message='Invalid value encountered',
                                   category=RuntimeWarning)
        np.warnings.filterwarnings(action='ignore', message='Overflow encountered',
                                   category=RuntimeWarning)

        interval = int(interval) if interval > 0 else 1000

        # mask elevation values less than 0
        elvVals = elev[variable].values
        elvVals[np.where(elvVals < 0)] = np.nan

        elvDim = [elev.dims['lon'], elev.dims["lat"]]
        # get ratio of high resoltion to low resolution
        ratio = [elvDim[i] / self.dims[i] for i in range(2)]

        nbands = []  # blank list

        with open(self.snowPath, 'w') as f:

            cnt = 1  # set grid cell id counter

            # pass counter
            pass_counter = range(2)

            # perform two passes on the raster data
            # 1) to grab the maximum number of bands for a given pixel
            # 2) to calculate the snow band parameters and write to output file
            for pass_cnt in pass_counter:

                # loop over each pixel in the template raster
                for i in range(self.dims[1]):
                    y1 = int(i * ratio[1])
                    y2 = int(y1 + ratio[1])
                    for j in range(self.dims[0]):
                        x1 = int(j * ratio[0])
                        x2 = int(x1 + ratio[0])

                        if self.grid[i, j] == True:

                            # get all hi res pixels in a template pixel
                            tmp = elvVals[y1:y2, x1:x2]

                            # find min and max values for interval
                            minelv = np.nanmin(tmp.astype(int)) - \
                                (np.nanmin(tmp.astype(int)) % interval)
                            maxelv = np.nanmax(tmp.astype(int)) + \
                                (np.nanmax(tmp.astype(int)) % interval)

                            # print(np.min(tmp).mask)
                            # create an array of band limits
                            bands = np.arange(
                                minelv, maxelv + interval, interval)

                            bcls = np.zeros_like(tmp)
                            bcls[:, :] = -1

                            # get the number of bands per pixel
                            for b in range(bands.size - 1):
                                # band counter
                                bcls[np.where((tmp >= bands[b]) & (
                                    tmp < bands[b + 1]))] = b

                            uniqcnt = np.unique(bcls[np.where(tmp > 0)])

                            # if it's the first pass get number of bands for each pixel
                            if pass_cnt == 0:
                                    # save to a list for second pass
                                nbands.append(uniqcnt.size)

                            if pass_cnt == 1:
                                # write grid cell id
                                f.write('{0}\t'.format(cnt))

                                # find frational area for each band and write to file
                                fracs, elvs = [], []
                                for c in range(maxbands):
                                    try:
                                        idx = np.where(bcls == uniqcnt[c])
                                        num = np.float(
                                            bcls[np.where(bcls >= 0)].size)
                                        if num == 0:
                                            num = np.float(idx[0].size)
                                        frac = np.float(idx[0].size) / num
                                        muelv = np.nanmean(tmp[idx])
                                    except IndexError:
                                        frac = 0
                                        muelv = 0

                                    fracs.append(frac)
                                    elvs.append(muelv)

                                frcString = '\t'.join(
                                    ['{:.4f}'.format(f) for f in fracs]) + '\t'
                                elvString = '\t'.join(
                                    ['{:.4f}'.format(e) for e in elvs]) + '\t'

                                f.write(frcString + elvString +
                                        frcString + '\n')

                                if pass_cnt == 1:
                                    cnt += 1  # plus one to the grid cell id counter

                if pass_cnt == 0:
                    # maximum number of bands for a pixel
                    maxbands = max(nbands)

        # print the number of bands for user to input into global parameter file
        print('Number of maximum bands: {0}'.format(maxbands))
        self.maxBands = maxbands

        return

    def writeVegParam(self, landcover, variable="landcover", scheme='IGBP', lai=None, alb=None):
        vegLookup = VEGPARAMS[scheme]

        lcDim = [landcover.dims['lon'], landcover.dims["lat"]]
        # get ratio of high resoltion to low resolution
        ratio = [lcDim[i] / self.dims[i] for i in range(2)]

        lcVals = landcover[variable].values
        lcVals[np.isnan(lcVals)] = 999

        if lai is not None:
            laiArr = np.squeeze(xr.where(np.isnan(lai),0,lai).to_array()).values
            laiScale = float(lai.attrs['scale'])
            laiArr = laiArr * laiScale
            self.vegParamLai = True

        if alb is not None:
            albArr = np.squeeze(alb.to_array()).values
            albScale = float(alb.attrs['scale'])
            albArr = albArr * albScale
            self.vegParamAlb = True

        # open output file for writing
        with open(self.vegPath, 'w') as f:
            cnt = 1  # grid cell id counter

            # loop over each pixel in the template raster
            for i in range(self.dims[1]):
                y1 = int(i * ratio[1])
                y2 = int(y1 + ratio[1])
                for j in range(self.dims[0]):
                    x1 = int(j * ratio[0])
                    x2 = int(x1 + ratio[0])

                    # get land cover data within template raster pixel
                    tmp = lcVals[y1:y2, x1:x2].astype(np.int32)

                    # if not a mask value...
                    if self.grid[i, j] == True:
                        ngidx = tmp <= len(vegLookup.keys()) - 1

                        # number of unique values
                        uniqcnt = np.unique(tmp[ngidx])
                        clscnt = np.bincount(tmp[ngidx].ravel())

                        # check if there is only 1 value in unique value array
                        if type(uniqcnt).__name__ == 'int':
                            Nveg = 1  # if so, only 1 value
                        else:
                            # if not, n veg equal to unique values
                            Nveg = len(uniqcnt)

                        # write the grid cell id and number of classes
                        f.write('{0} {1}\n'.format(cnt, Nveg))

                        # if there is vegetation data...
                        if Nveg != 0:
                            # ...loop over each class...
                            for t in range(uniqcnt.size):
                                # ...and grab the parameter attributes
                                vegcls = int(uniqcnt[t])  # class value
                                # percent coverage
                                Cv = np.float(
                                    clscnt[uniqcnt[t]]) / np.float(clscnt.sum())
                                # intermediate variale
                                attributes = vegLookup[vegcls]
                                # rooting depth layer 1
                                rdepth1 = str(attributes['rootd1'])
                                # rooting fraction layer 1
                                rfrac1 = str(attributes['rootfr1'])
                                # rooting depth layer 2
                                rdepth2 = str(attributes['rootd2'])
                                # rooting fraction layer 2
                                rfrac2 = str(attributes['rootfr2'])
                                # rooting depth layer 3
                                rdepth3 = str(attributes['rootd3'])
                                # rooting fraction layer 3
                                rfrac3 = str(attributes['rootfr3'])

                                # write the veg class information
                                f.write('\t{0} {1:.4f} {2} {3} {4} {5} {6} {7}\n'.format(
                                    vegcls, Cv, rdepth1, rfrac1, rdepth2, rfrac2, rdepth3, rfrac3))

                                if lai is not None or alb is not None:
                                    clsIdx = np.where(tmp==vegcls)
                                    monLai,monAlb = [],[]
                                    # loop over each month
                                    for m in range(12):
                                        # grab the month time slice
                                        if lai is not None:
                                            laiStep = laiArr[m, y1:y2, x1:x2]
                                            clsLai = np.nanmean(laiStep[tmp==vegcls])
                                            meanLai = np.nanmean(laiStep) # repeated computation ... can it be computed outside of loop?
                                            laiVal = meanLai if np.isnan(clsLai) else clsLai
                                            monLai.append(f"{laiVal:.4f}")

                                        if alb is not None:
                                            albStep = albArr[m, y1:y2, x1:x2]
                                            clsAlb = np.nanmean(albStep[tmp==vegcls])
                                            meanAlb = np.nanmean(albStep) # repeated computation ... can it be computed outside of loop?
                                            albVal = meanAlb if np.isnan(clsAlb) else clsAlb
                                            monAlb.append(f"{albVal:.4f}")

                                if lai is not None:
                                    f.write('\t'+'\t'.join(monLai)+'\n')
                                if alb is not None:
                                    f.write('\t'+'\t'.join(monAlb)+'\n')



                        cnt += 1  # plus one to grid cell id counter

        return

    def writeVegLib(self, landcover, lai, alb, variable='landcover',scheme='IGBP'):
        vegLookup = VEGPARAMS[scheme]

        lcDim = [landcover.dims['lon'], landcover.dims["lat"]]
        # get ratio of high resoltion to low resolution
        ratio = [lcDim[i] / self.dims[i] for i in range(2)]

        lcVals = np.squeeze(landcover[variable].values)
        lcVals[np.isnan(lcVals)] = 999

        laiArr = np.squeeze(xr.where(np.isnan(lai),0,lai).to_array()).values
        laiScale = float(lai.attrs['scale'])

        albArr = np.squeeze(alb.to_array()).values
        albScale = float(alb.attrs['scale'])

        classes = list(vegLookup.keys())

        # open output file for writing
        with open(self.vegLib, 'w') as f:
            f.write('#Class\tOvrStry\tRarc\tRmin\tJAN-LAI\tFEB-LAI\tMAR-LAI\tAPR-LAI\tMAY-LAI\tJUN-LAI\tJUL-LAI\tAUG-LAI\tSEP-LAI\tOCT-LAI\tNOV-LAI\tDEC-LAI\tJAN-ALB\tFEB_ALB\tMAR-ALB\tAPR-ALB\tMAY-ALB\tJUN-ALB\tJUL-ALB\tAUG-ALB\tSEP-ALB\tOCT-ALB\tNOV-ALB\tDEC-ALB\tJAN-ROU\tFEB-ROU\tMAR-ROU\tAPR-ROU\tMAY-ROU\tJUN-ROU\tJUL-ROU\tAUG-ROU\tSEP-ROU\tOCT-ROU\tNOV-ROU\tDEC-ROU\tJAN-DIS\tFEB-DIS\tMAR-DIS\tAPR-DIS\tMAY-DIS\tJUN-DIS\tJUL-DIS\tAUG-DIS\tSEP-DIS\tOCT-DIS\tNOV-DIS\tDEC-                       DIS\tWIND_H\tRGL\trad_atten\twind_atten\ttruck_ratio\tCOMMENT\n')
            # loop over each class
            for cls in classes:
                line = []

                # get attributes
                attributes = vegLookup[cls]

                clsMask = lcVals==cls
                overstory = int(attributes['overstory'])
                vegHeight = float(attributes['h'])  # veg height
                # adjust wind height value if overstory is true
                wind_h = vegHeight + 10. if overstory == 1 else vegHeight + 2.

                # blank lists to append monthly values
                monLai, monAlb,rough, displace= [], [], [], []

                # loop over each month
                for j in range(12):
                    # grab the month time slice
                    laiStep = laiArr[j, :, :]
                    albStep = albArr[j, :, :]

                    defaultLai = np.nanmean(laiStep) * laiScale
                    defaultAlb = np.nanmean(albStep) * albScale

                    clsLaiMean = np.nanmean(laiStep[clsMask]) * laiScale
                    clsAlbMean = np.nanmean(albStep[clsMask]) * albScale

                    meanLai = defaultLai if np.isnan(clsLaiMean) else clsLaiMean
                    meanAlb = defaultAlb if np.isnan(clsAlbMean) else clsAlbMean

                    # grab the data at location
                    monLai.append(f"{meanLai:.4f}")
                    monAlb.append(f"{meanAlb:.4f}")
                    rough.append(f"{0.123 * vegHeight:.4f}")
                    displace.append(f"{0.67 * vegHeight:.4f}")

                line.append(f"{cls}") # vegetation class
                line.append(f"{overstory}") # overstory value
                line.append(f"{attributes['rarc']}")  # veg architectural resistance
                line.append(f"{attributes['rmin']}") # veg minimum stomatal resistance
                line.append('\t'.join(monLai))
                line.append('\t'.join(monAlb))
                line.append('\t'.join(rough))
                line.append('\t'.join(displace))
                line.append(f"{wind_h:.2f}")
                line.append(f"{attributes['rgl']}") # Minimum incoming shortwave radiation for ET
                line.append(f"{attributes['rad_atn']}") # radiation attenuation factor
                line.append(f"{attributes['wnd_atn']}") # wind speed attenuation
                line.append(f"{attributes['trnk_r']}")
                line.append(f"{attributes['classname']}\n")

                    # write the land surface parameterization data
                f.write('\t'.join(line))

        return

    def writeForcing(self, forcings, variables, maxWorkers=8):

        def writeSeries(pt):
            series = forcings.isel(lat=pt[1], lon=pt[0])
            lat, lon = series.lat.values, series.lon.values

            metFile = self.forcingPath + f"forcing_{lat:.4f}_{lon:.4f}"
            with open(metFile, "w") as f:
                for t in range(series.dims["time"]):
                    line = []
                    for i in variables:
                        line.append(f"{series[i].values[t]:.4f}")
                    text = '\t'.join(line) + "\n"
                    f.write(text)
            return 1

        self.nMetVars = len(variables)
        self.metVars = variables

        pts = [[j, i] for i in range(self.dims[1]) for j in range(
            self.dims[0]) if self.grid[i, j] == True]
        with ThreadPoolExecutor(maxWorkers) as executor:
            gen = executor.map(writeSeries, pts)
            results = list(gen)

        return

    def execute(self,vicExe='vicNl',writeToLog=True):
        cmd = f"which {vicExe}"
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, err = proc.communicate()

        if len(out) == 0:
            raise OSError("could not find the model executable...please explicitly set the exe name or install the model in a system path")

        cmd = f"{vicExe} -g {self.globPath}"

        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, err = proc.communicate()

        if writeToLog and not self.parallel:
            logFile = f"vic_{datetime.datetime.now().strftime('%Y%m%dT%H%M%s')}.log"
            logPath = os.path.join(self.path,logFile)
            with open(logPath,'w') as f:
                f.write(out)

        elif writeToLog and self.parallel:
            # WARNING: need to warn about logging when running in parallel
            pass

        return

    def fluxToDataset(self,nodataVal=-999.):
        fluxFiles = sorted([os.path.join(self.fluxPath,f) for f in os.listdir(self.fluxPath)])

        lats = np.array(sorted(list(set([float(f.split('_')[-2]) for f in fluxFiles]))))
        lons = np.array(sorted(list(set([float(f.split('_')[-1]) for f in fluxFiles]))))

        # check to make sure the order or lat/lon is correct
        for i,file in enumerate(fluxFiles):
            with open(file,'r') as f:
                contents = f.readlines()
                header = contents[:6]
                header = [info.replace('#','').replace('\n','').strip() for info in header]
                data = np.loadtxt(StringIO('\n'.join(contents[6:])))
                data = data[:,3:]

            thisLat = float(file.split('_')[-2])
            thisLon = float(file.split('_')[-1])

            if i == 0:
                # get the date information from header
                startTime = 'T'.join(header[2].split(' ')[-2:])
                timeDelta = header[1].split(' ')[-1]+'H'
                periods = int(header[0].split(' ')[-1])
                dates = pd.date_range(startTime,periods=periods,freq=timeDelta)

                variables = [v.strip() for v in header[-1].split('\t')[3:]]

                df = {'time': {'dims': ('time'), 'data': dates,
                              'attrs': {'unit': 'day'}},
                      'lon': {'dims': ('lon'), 'data': lons,
                              'attrs': {'long_name': "longitude", 'units': "degrees_east"}},
                      'lat': {'dims': ('lat'), 'data': lats,
                              'attrs': {'long_name': "latitude", 'units': "degrees_north"}}
                      }

                for v in variables:
                    template = np.zeros((len(lats),len(lons),len(dates)))
                    template[:,:,:] = nodataVal
                    df[v] = {'dims': ('lat', 'lon', 'time'),
                             'data': template}

            latIdx = np.where(thisLat == lats)
            lonIdx = np.where(thisLon == lons)
            # print(file)
            # print((thisLat,thisLon),(latIdx,lonIdx))

            for j,v in enumerate(variables):
                df[v]['data'][latIdx, lonIdx, :] = data[:,j]

        outDs = xr.Dataset.from_dict(df)
        attrs = {'title': "VIC hydrologic model outputs",
                 'source': 'VIC Version 4.2.d 2015-June-20',
                 'date_created': f'{datetime.datetime.now()}',
                 'nodata':nodataVal}

        return outDs.where(outDs!=nodataVal)


    def setupFromConfig(self):

        return
