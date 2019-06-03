from __future__ import division, print_function
import os
import subprocess
from numba import jit
import numpy as np
import pandas as pd
import datetime

from .. import core
from . import routing


class Hymod(core.model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.P_LB = [1.0,
                     0.1,
                     0.1,
                     0.0,
                     0.1
                    ]
        self.P_UB = [500,
                     2.0,
                     0.99,
                     0.10,
                     0.99
                    ]

        return

    # Get random parameter set
    def get_random_params(self):
        return np.random.uniform(self.P_LB, self.P_UB)

    def execute(self,precipSeries=None,petSeries=None,**kwargs):
        cols = self.basins.columns
        if precipSeries not in cols:
            raise ValueError('Selected precipSeries key {} not in basin dataframe'.format(precipSeries))
        if petSeries not in cols:
            raise ValueError('Selected petSeries key {} not in basin dataframe'.format(petSeries))

        q = []
        for i in range(self.basins.shape[0]):
            print(i)
            subbasin = self.basins.iloc[i]
            prc = subbasin[precipSeries]
            eto = subbasin[petSeries]

            conversion = (1 / 86400.) #* float(subbasin['area'])

            simq = self.simulate(prc.values,eto.values,**kwargs)

            q.append(pd.Series(simq,index=prc.index,name='Q') * conversion)

        out = self.basins.copy()
        out['Q'] = q

        # finalQ = []
        # for i in range(out.shape[0]):
        #     thisShed = out.iloc[i]
        #     if i == 0:
        #         finalQ.append(thisShed['Q'])
        #     else:
        #         prevShed = out.iloc[i-1]
        #         finalQ.append(thisShed['Q'] + prevShed['Q'])
        # out['finalQ'] = finalQ

        return out

    @classmethod
    @jit
    def simulate(cls,Precip, PET, cmax=None,bexp=None,alpha=None,Rs=None,Rq=None):
        """
        See https://www.proc-iahs.net/368/180/2015/piahs-368-180-2015.pdf for a scientific paper.

        :param cmax:
        :param bexp:
        :param alpha:
        :param Rs:
        :param Rq:
        :return: Dataset of water in hymod (has to be calculated in litres)
        :rtype: list
        """
        lt_to_m = 0.001

        # HYMOD PROGRAM IS SIMPLE RAINFALL RUNOFF MODEL
        x_loss = 0.0
        # Initialize slow tank state
        x_slow = 2.3503 / (Rs * 22.5)
        x_slow = 0  # --> works ok if calibration data starts with low discharge
        # Initialize state(s) of quick tank(s)
        x_quick = [0,0,0]
        t = 0
        outflow = []
        output = []
        # START PROGRAMMING LOOP WITH DETERMINING RAINFALL - RUNOFF AMOUNTS

        while t <= len(Precip)-1:
            Pval = Precip[t]
            PETval = PET[t]
            # Compute excess precipitation and evaporation
            ER1, ER2, x_loss = cls._excess(x_loss, cmax, bexp, Pval, PETval)
            # Calculate total effective rainfall
            ET = ER1 + ER2
            #  Now partition ER between quick and slow flow reservoirs
            UQ = alpha * ET
            US = (1 - alpha) * ET
            # Route slow flow component with single linear reservoir
            x_slow, QS = routing.linearReservoir(x_slow, US, Rs)
            # Route quick flow component with linear reservoirs
            inflow = UQ

            for i in range(3):
                # Linear reservoir
                x_quick[i], outflow = routing.linearReservoir(x_quick[i], inflow, Rq)
                inflow = outflow

            # Compute total flow for timestep
            output.append((QS + outflow)/lt_to_m)
            t = t+1

        return output

    @classmethod
    @jit
    def _power(cls,X,Y):
        X=abs(X) # Needed to capture invalid overflow with netgative values
        return X**Y

    @classmethod
    @jit
    def _excess(cls,x_loss,cmax,bexp,Pval,PETval):
        # this function calculates excess precipitation and evaporation
        xn_prev = x_loss
        ct_prev = cmax * (1 - cls._power((1 - ((bexp + 1) * (xn_prev) / cmax)), (1 / (bexp + 1))))
        # Calculate Effective rainfall 1
        ER1 = max((Pval - cmax + ct_prev), 0.0)
        Pval = Pval - ER1
        dummy = min(((ct_prev + Pval) / cmax), 1)
        xn = (cmax / (bexp + 1)) * (1 - cls._power((1 - dummy), (bexp + 1)))

        # Calculate Effective rainfall 2
        ER2 = max(Pval - (xn - xn_prev), 0)

        # Alternative approach
        evap = (1 - (((cmax / (bexp + 1)) - xn) / (cmax / (bexp + 1)))) * PETval  # actual ET is linearly related to the soil moisture state
        xn = max(xn - evap, 0)  # update state

        return ER1,ER2,xn
