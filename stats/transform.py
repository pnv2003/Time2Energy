import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

def log_transform(ts):
    return np.log(ts)

def moving_average(ts, window=12):
    return ts.rolling(window=window).mean()

def difference(ts, period=1):
    return ts - ts.shift(period)

def ewma(ts, halflife=12):
    return ts.ewm(halflife=halflife).mean()

def residual_component(ts, model='additive'):
    decomposition = seasonal_decompose(ts, model=model)
    return decomposition.resid

def inverse_log_transform(ts):
    return np.exp(ts)

def inverse_difference(ts, orig, period=1):
    return ts.cumsum() + orig.shift(period)
