import numpy as np
import pandas as pd

# Import a Kalman filter and other useful libraries
from pykalman import KalmanFilter
from scipy import poly1d

from time import *
from sklearn import tree
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import pandas_datareader.data as web
import time
start_time = time.time()
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

def get_p(prices, m, d, k):
    """ Returns the dth-degree rolling momentum of data using lookback window length k """
    x = np.log(prices)
    v = x.diff()
    m = np.array(m)
    
    if d == 0:
        return pd.rolling_sum(v, k)
    elif d == 1:
        return pd.rolling_sum(m*v, k)
    elif d == 2:
        return pd.rolling_sum(m*v, k)/pd.rolling_sum(m, k)
    elif d == 3:
        return pd.rolling_mean(v, k)/pd.rolling_std(v, k)
    
def backtest_get_p(prices, m, d):
    """ Returns the dth-degree rolling momentum of data"""
    v = np.diff(np.log(prices))
    m = np.array(m)
    
    if d == 0:
        return np.sum(v)
    elif d == 1:
        return np.sum(m*v)
    elif d == 2:
        return np.sum(m*v)/np.sum(m)
    elif d == 3:
        return np.mean(v)/np.std(v)
