from __future__ import division, print_function
import os
import subprocess
from numba import jit
import numpy as np
import pandas as pd
import datetime

from .. import core

class HBV96(core.model):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.srcDir = os.path.dirname(os.path.abspath(__file__))
        self.outputFile = None

        # HBV base model parameters
        self.P_LB = [-1.5, #ltt
                0.001, #utt
                0.001, #ttm
                0.04, #cfmax [mm c^-1 h^-1]
                50.0, #fc
                0.6, #ecorr
                0.001, #etf
                0.2, #lp
                0.00042, #k [h^-1] upper zone
                0.0000042, #k1 lower zone
                0.001, #alpha
                1.0, #beta
                0.001, #cwh
                0.01, #cfr
                0.0, #c_flux
                0.001, #perc mm/h
                0.6, #rfcf
                0.4,  #sfcf
                1] # Maxbas

        self.P_UB = [2.5, #ttm
                3.0, #utt
                2.0, #ttm
                0.4, #cfmax [mm c^-1 h^-1]
                500.0, #fc
                1.4, #ecorr
                5.0, #etf
                0.5, #lp
                0.0167, #k upper zone
                0.00062, #k1 lower zone
                1.0, #alpha
                6.0, #beta
                0.1, #cwh
                1.0, #cfr
                0.08, #c_flux - 2mm/day
                0.125, #perc mm/hr
                1.4, #rfcf
                1.4, #sfcf
                10] # maxbas

        self.DEF_ST = [0.0, 10.0, 10.0, 10.0, 0.0]
        self.DEF_q0 = 10.0

        return

    # Get random parameter set
    def get_random_params(self):
        return np.random.uniform(self.P_LB, self.P_UB)

    def execute(self,precipSeries=None,tempSeries=None,petSeries=None,**kwargs):

        return

    @classmethod
    @jit
    def simulate(cls,avg_prec, temp, et,
                 par=None, p2=None, init_st=None, ll_temp=None,
                 q_0=10):
        '''
        ========
        Simulate
        ========
        Run the HBV model for the number of steps (n) in precipitation. The
        resluts are (n+1) simulation of discharge as the model calculates step n+1

        Parameters
        ----------
        avg_prec : array_like [n]
            Average precipitation [mm/h]
        temp : array_like [n]
            Average temperature [C]
        et : array_like [n]
            Potential Evapotranspiration [mm/h]
        par : array_like [18]
            Parameter vector, set up as:
            [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1,
            alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
        p2 : array_like [2]
            Problem parameter vector setup as:
            [tfac, area]
        init_st : array_like [5], optional
            Initial model states, [sp, sm, uz, lz, wc]. If unspecified,
            [0.0, 30.0, 30.0, 30.0, 0.0] mm
        ll_temp : array_like [n], optional
            Long term average temptearature. If unspecified, calculated from temp.
        q_0 : float, optional
            Initial discharge value. If unspecified set to 10.0

        Returns
        -------
        q_sim : array_like [n]
            Discharge for the n time steps of the precipitation vector [m3/s]
        st : array_like [n, 5]
            Model states for the complete time series [mm]
        '''

        if init_st is None:
            st = [DEF_ST, ]

        if ll_temp is None:
            ll_temp = [np.mean(temp), ] * len(avg_prec)

        q_sim = [q_0, ]

        for i in xrange(len(avg_prec)):
            v = [avg_prec[i], temp[i], et[i], ll_temp[i]]
            q_out, st_out = cls._step_run(par, p2, v, st[i])
            q_sim.append(q_out)
            st.append(st_out)

        if len(p2) > 2:  # Forcing maxbas to be predefined
            maxbas = p2[2]
        elif len(par) > 18:  # Putting maxbas as parameter to be optimised
            maxbas = par[18]
        else:
            maxbas = 1

        q_tr = cls._routing(np.array(q_sim), maxbas)

        return q_tr, st

    @classmethod
    @jit
    def _precipitation(cls,temp, ltt, utt, prec, rfcf, sfcf, tfac):
        '''
        ==============
        Precipitation
        ==============
        Precipitaiton routine of the HBV96 model.
        If temperature is lower than ltt, all the precipitation is considered as
        snow. If the temperature is higher than utt, all the precipitation is
        considered as rainfall. In case that the temperature is between ltt and
        utt, precipitation is a linear mix of rainfall and snowfall.
        Parameters
        ----------
        temp : float
            Measured temperature [C]
        ltt : float
            Lower temperature treshold [C]
        utt : float
            Upper temperature treshold [C]
        prec : float
            Precipitation [mm]
        rfcf : float
            Rainfall corrector factor
        sfcf : float
            Snowfall corrector factor
        Returns
        -------
        _rf : float
            Rainfall [mm]
        _sf : float
            Snowfall [mm]
        '''

        if temp <= ltt:
            _rf = 0.0
            _sf = prec*sfcf

        elif temp >= utt:
            _rf = prec*rfcf
            _sf = 0.0

        else:
            _rf = ((temp-ltt)/(utt-ltt)) * prec * rfcf
            _sf = (1.0-((temp-ltt)/(utt-ltt))) * prec * sfcf

        return _rf, _sf


    @classmethod
    @jit
    def _snow(cls,cfmax, tfac, temp, ttm, cfr, cwh, _rf, _sf, wc_old, sp_old):
        '''
        ====
        Snow
        ====

        Snow routine of the HBV-96 model.

        The snow pack consists of two states: Water Content (wc) and Snow Pack
        (sp). The first corresponds to the liquid part of the water in the snow,
        while the latter corresponds to the solid part. If the temperature is
        higher than the melting point, the snow pack will melt and the solid snow
        will become liquid. In the opposite case, the liquid part of the snow will
        refreeze, and turn into solid. The water that cannot be stored by the solid
        part of the snow pack will drain into the soil as part of infiltration.
        Parameters
        ----------
        cfmax : float
            Day degree factor
        tfac : float
            Temperature correction factor
        temp : float
            Temperature [C]
        ttm : float
            Temperature treshold for Melting [C]
        cfr : float
            Refreezing factor
        cwh : float
            Capacity for water holding in snow pack
        _rf : float
            Rainfall [mm]
        _sf : float
            Snowfall [mm]
        wc_old : float
            Water content in previous state [mm]
        sp_old : float
            snow pack in previous state [mm]
        Returns
        -------
        _in : float
            Infiltration [mm]
        _wc_new : float
            Water content in posterior state [mm]
        _sp_new : float
            Snowpack in posterior state [mm]
        '''

        if temp > ttm:

            if cfmax*(temp-ttm) < sp_old+_sf:
                _melt = cfmax*(temp-ttm)
            else:
                _melt = sp_old+_sf

            _sp_new = sp_old + _sf - _melt
            _wc_int = wc_old + _melt + _rf

        else:
            if cfr*cfmax*(ttm-temp) < wc_old+_rf:
                _refr = cfr*cfmax*(ttm - temp)
            else:
                _refr = wc_old + _rf

            _sp_new = sp_old + _sf + _refr
            _wc_int = wc_old - _refr + _rf

        if _wc_int > cwh*_sp_new:
            _in = _wc_int-cwh*_sp_new
            _wc_new = cwh*_sp_new
        else:
            _in = 0.0
            _wc_new = _wc_int

        return _in, _wc_new, _sp_new


    @classmethod
    @jit
    def _soil(cls,fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux, inf,
              ep, sm_old, uz_old):
        '''
        ====
        Soil
        ====

        Soil routine of the HBV-96 model.

        The model checks for the amount of water that can infiltrate the soil,
        coming from the liquid precipitation and the snow pack melting. A part of
        the water will be stored as soil moisture, while other will become runoff,
        and routed to the upper zone tank.
        Parameters
        ----------
        fc : float
            Filed capacity
        beta : float
            Shape coefficient for effective precipitation separation
        etf : float
            Total potential evapotranspiration
        temp : float
            Temperature
        tm : float
            Average long term temperature
        e_corr : float
            Evapotranspiration corrector factor
        lp : float _soil
            wilting point
        tfac : float
            Time conversion factor
        c_flux : float
            Capilar flux in the root zone
        _in : float
            actual infiltration
        ep : float
            actual evapotranspiration
        sm_old : float
            Previous soil moisture value
        uz_old : float
            Previous Upper zone value
        Returns
        -------
        sm_new : float
            New value of soil moisture
        uz_int_1 : float
            New value of direct runoff into upper zone
        '''

        qdr = max(sm_old + inf - fc, 0)
        _in = inf - qdr
        _r = ((sm_old/fc)** beta) * _in
        _ep_int = (1.0 + etf*((temp - tm)/tm))*e_corr*ep
    #    _ep_int = e_corr*ep
        _ea = min(_ep_int, (sm_old/(lp*fc))*_ep_int)
        _cf = tfac*c_flux*((fc - sm_old)/fc)
        sm_new = max(sm_old + _in - _r + _cf - _ea, 0)
        uz_int_1 = uz_old + _r - _cf + qdr

        return sm_new, uz_int_1, qdr

    @classmethod
    @jit
    def _response(cls,tfac, perc, alpha, k, k1, area, lz_old, uz_int_1, qdr):
        '''
        ========
        Response
        ========
        The response routine of the HBV-96 model.

        The response routine is in charge of transforming the current values of
        upper and lower zone into discharge. This routine also controls the
        recharge of the lower zone tank (baseflow). The transformation of units
        also occurs in this point.

        Parameters
        ----------
        tfac : float
            Number of hours in the time step
        perc : float
            Percolation value [mm\hr]
        alpha : float
            Response box parameter
        k : float
            Upper zone recession coefficient
        k1 : float
            Lower zone recession coefficient
        area : float
            Catchment area [Km2]
        lz_old : float
            Previous lower zone value [mm]
        uz_int_1 : float
            Previous upper zone value before percolation [mm]
        qdr : float
            Direct runoff [mm]

        '''
        perc = tfac*perc
        lz_int_1 = lz_old + np.min([perc, uz_int_1])
        uz_int_2 = np.max([uz_int_1 - perc, 0.0])

        _q_0 = k*(uz_int_2**(1.0 + alpha))
        _q_1 = k1*lz_int_1

        uz_new = max(uz_int_2 - (_q_0), 0)
        lz_new = max(lz_int_1 - (_q_1), 0)

        q_new = area*(_q_0 + _q_1)/(3.6*tfac)

        return q_new, uz_new, lz_new

    @classmethod
    @jit
    def _tf(cls,maxbas):
        ''' Transfer function weight generator '''
        wi = []
        for x in range(1, maxbas+1):
            if x <= (maxbas)/2.0:
                # Growing transfer
                wi.append((x)/(maxbas+2.0))
            else:
                # Receding transfer
                wi.append(1.0 - (x+1)/(maxbas+2.0))

        #Normalise weights
        wi = np.array(wi)/np.sum(wi)
        return wi

    @classmethod
    @jit
    def _routing(cls,q, maxbas=1):
        """
        This function implements the transfer function using a triangular
        function
        """
        assert maxbas >= 1, 'Maxbas value has to be larger than 1'
        # Get integer part of maxbas
        maxbas = int(maxbas)

        # get the weights
        w = cls._tf(maxbas)

        # rout the discharge signal
        q_r = np.zeros_like(q, dtype='float64')
        q_temp = q
        for w_i in w:
            q_r += np.array(q_temp)*w_i
            q_temp = np.insert(q_temp, 0, 0.0)[:-1]

        return q_r


    @classmethod
    @jit
    def _step_run(cls,p, p2, v, St):
        '''
        ========
        Step run
        ========

        Makes the calculation of next step of discharge and states

        Parameters
        ----------
        p : array_like [18]
            Parameter vector, set up as:
            [ltt, utt, ttm, cfmax, fc, ecorr, etf, lp, k, k1,
            alpha, beta, cwh, cfr, c_flux, perc, rfcf, sfcf]
        p2 : array_like [2]
            Problem parameter vector setup as:
            [tfac, area]
        v : array_like [4]
            Input vector setup as:
            [prec, temp, evap, llt]
        St : array_like [5]
            Previous model states setup as:
            [sp, sm, uz, lz, wc]
        Returns
        -------
        q_new : float
            Discharge [m3/s]
        St : array_like [5]
            Posterior model states
        '''
        ## Parse of parameters from input vector to model
        ltt = p[0]
        utt = p[1]
        ttm = p[2]
        cfmax = p[3]
        fc = p[4]
        e_corr = p[5]
        etf = p[6]
        lp = p[7]
        k = p[8]
        k1 = p[9]
        alpha = p[10]
        beta = p[11]
        cwh = p[12]
        cfr = p[13]
        c_flux = p[14]
        perc = p[15]
        rfcf = p[16]
        sfcf = p[17]

        ## Non optimisable parameters
        tfac = p2[0]
        area = p2[1]


        ## Parse of Inputs
        avg_prec = v[0] # Precipitation [mm]
        temp = v[1] # Temperature [C]
        ep = v[2] # Long terms (monthly) Evapotranspiration [mm]
        tm = v[3] #Long term (monthly) average temperature [C]

        ## Parse of states
        sp_old = St[0]
        sm_old = St[1]
        uz_old = St[2]
        lz_old = St[3]
        wc_old = St[4]

        rf, sf = cls._precipitation(temp, ltt, utt, avg_prec, rfcf, sfcf, tfac)
        inf, wc_new, sp_new = cls._snow(cfmax, tfac, temp, ttm, cfr, cwh, rf, sf,
                                   wc_old, sp_old)
        sm_new, uz_int_1, qdr = cls._soil(fc, beta, etf, temp, tm, e_corr, lp, tfac, c_flux,
                                inf, ep, sm_old, uz_old)
        q_new, uz_new, lz_new = cls._response(tfac, perc, alpha, k, k1, area, lz_old,
                                        uz_int_1, qdr)

        return q_new, [sp_new, sm_new, uz_new, lz_new, wc_new]
