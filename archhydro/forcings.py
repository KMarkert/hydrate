import warnings
import numpy as np
import xarray as xr
from archhydro import core


class Forcing(core.Dataset):
    def __init__(self, *args, bands=None, timestep='1D', reductions=None, **kwargs,):
        super(Forcing, self).__init__(*args, **kwargs)
        self._BANDS = bands

        self._STEP = timestep

        if reductions != None:
            if type(reductions) != list:
                reductions = [reductions]

            if (len(bands) == len(reductions)):
                opts = {'mean': np.mean, 'min': np.min,
                        'max': np.max, 'std': np.std, 'sum': np.sum}
                self._REDUCERS = [opts[i] for i in reductions]
            else:
                raise ValueError(
                    'the length of reductions must equal the length of bands')

        else:
            self._REDUCERS = None

        return

    @property
    def bands(self):
        return self._BANDS

    @property
    def timestep(self):
        return self._STEP

    def getModelDomain(self, model,maxWorkers=1,verbose=False):
        data = self.getSeries(bands=self._BANDS,
                              resolution=model.res, startTime=model.start, endTime=model.end, dims=model.dims,
                              initPt=model.initPt,maxWorkers=maxWorkers,verbose=verbose)

        if self._STEP != model.step:
            warnings.warn("selected forcing timestep is not equal to model timestep "
                          "continuing to process but there may be time misalignment")

        if self._REDUCERS is not None:
            # out = data.copy()
            for i, b in enumerate(self._BANDS):
                if i == 0:
                    out = data.resample(time=self._STEP).reduce(
                        self._REDUCERS[i], dim='t').transpose('lat', 'lon', 'time')
                else:
                    out[b][:, :, :] = data[b].resample(time=self._STEP).reduce(
                        self._REDUCERS[i], dim='t').values
        else:
            out = data

        tempMask = xr.DataArray(
            model.grid, coords=[out.lat, out.lon], dims=['lat', 'lon'])

        return out.where(tempMask)
