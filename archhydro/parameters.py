import warnings
import numpy as np
from scipy import ndimage
import xarray as xr
import json

from archhydro import core


class Parameter(core.Dataset):
    def __init__(self, *args, bands=None, **kwargs,):
        super(Parameter, self).__init__(*args, **kwargs)
        self._BANDS = bands

        return

    @property
    def bands(self):
        return self._BANDS

    def getModelDomain(self, model, scaleFactor=1,maxWorkers=2,verbose=False):
        """
        model - model class from archhydro core
        scaleFactor - factor to adjust the resolution of the dataset, scaleFactor of 2 increases the resolution by 2
        """

        parRes = model.res / scaleFactor
        parDims = tuple(dim * scaleFactor for dim in model.dims)
        mask = ndimage.zoom(model.grid, zoom=scaleFactor, order=0)

        data = self.getSeries(bands=self._BANDS,
                              resolution=parRes, startTime=model.start, endTime=model.end,
                              dims=parDims, initPt=model.initPt,maxWorkers=maxWorkers,verbose=verbose)

        tempMask = xr.DataArray(
            mask, coords=[data.lat, data.lon], dims=['lat', 'lon'])

        return data.where(tempMask)
