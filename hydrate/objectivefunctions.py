import numpy as np

def nse(q_rec, q_sim):
    '''
    ===
    NSE
    ===

    Nash-Sutcliffe efficiency. Metric for the estimation of performance of the
    hydrological model

    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n]
        Simulated discharge [m3/s]

    Returns
    -------
    f : float
        NSE value
    '''
    a = np.square(np.subtract(q_rec, q_sim))
    b = np.square(np.subtract(q_rec, np.nanmean(q_rec)))
    if a.any < 0.0:
        return(np.nan)
    f = 1.0 - (np.nansum(a)/np.nansum(b))
    return f


def rmse(q_rec,q_sim):
    '''
    ====
    RMSE
    ====

    Root Mean Squared Error. Metric for the estimation of performance of the
    hydrological model.

    Parameters
    ----------
    q_rec : array_like [n]
        Measured discharge [m3/s]
    q_sim : array_like [n]
        Simulated discharge [m3/s]

    Returns
    -------
    f : float
        RMSE value
    '''
    erro = np.square(np.subtract(q_rec,q_sim))
    if erro.any < 0:
        return(np.nan)
    f = np.sqrt(np.nanmean(erro))
    return f
