import numpy as np
import pandas as pd

def hargreaves(Tmin,Tmax):
    """
    accepts xarray dataarrays
    returns xarray dataarray
    """
    Gsc = 367
    lhov = 2.257
    Tavg = (Tmin+Tmax)/2. # assumes that the averaging is accuring on time...?

    date_list = pd.to_datetime(Tavg.coords['t'].values)
    doy = [x.timetuple().tm_yday for x in date_list]

    ETo = Tavg.copy().rename('eto')

    for i,t in enumerate(doy):
        tmin = Tmin.isel(t=i)
        tmax = Tmax.isel(t=i)
        tavg = Tavg.isel(t=i)

        b = 2 * np.pi * (t/365)
        Rav = 1.00011 + 0.034221*np.cos(b) + 0.00128*np.sin(b) + 0.000719*np.cos(2*b) + 0.000077*np.sin(2*b)
        Ho = ((Gsc * Rav) * 86400)/1e6

        pet = (0.0023 * Ho * (tmax-tmin)**0.5 * (tavg+17.8))

        ETo[:,:,i] = pet.values

    return ETo
