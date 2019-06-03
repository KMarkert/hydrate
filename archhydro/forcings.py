import warnings
import numpy as np
from . import core

class Forcing(object):
    def __init__(self,project,collection,bands=None,timestep='1D',reductions=None):
        self.name = '{}/assets/{}'.format(project, collection)
        self.project = project
        self.id = collection

        self.bands = bands

        self.step = timestep

        if reductions != None:
            if type(reductions) != list:
                reductions =[reductions]

            if (len(bands) == len(reductions)):
                opts = {'mean':np.mean,'min':np.min,'max':np.max,'std':np.std,'sum':np.sum}
                self.reducers = [opts[i] for i in reductions]
            else:
                raise ValueError('the length of reductions must equal the length of bands')

        else:
            self.reducers = None

        return

    def getTimeSeries(self,http,model):
        data = core.getTimeSeries(http,self.project,self.collection,bands=bands,
                    resolution=model.res,startTime=model.start,endTime=model.end,dims=model.dims,
                    initPt=model.initPt)

        if self.step != model.step:
            warnings.warn("selected forcing timestep is not equal to model timestep "
                          "continuing to process but there may be time misalignment")

        if self.reducers != None:
            # out = data.copy()
            for i,b in enumerate(self.bands):
                if i == 0:
                    out = data.resample(t=self.step).reduce(self.reducers[i],dim='t').transpose('y', 'x', 't')
                else:
                    out[b][:,:,:] = data[b].resample(t=self.step).reduce(self.reducers[i],dim='t').values
        else:
            out = data

        return out
