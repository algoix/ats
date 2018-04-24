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


# loading csv file
def get_csv_pd(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd=pd.read_csv(path,sep=',',dtype={'askPrice':np.float32,'askSize':np.float32,
                                           'bidPrice':np.float32,'bidSize':np.float32},index_col=0,parse_dates=True)
    #spy_pd = pd.read_csv(path, usecols=['askPrice','askSize','bidPrice','bidSize'], engine='python', skipfooter=3)
    return spy_pd

def get_csv_pd_notime(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd = pd.read_csv(path, usecols=['askPrice','askSize','bidPrice','bidSize'], engine='python', skipfooter=3)
    return spy_pd

def BA(df):
    df.bidPrice=df.loc[:,'bidPrice'].replace(to_replace=0, method='ffill')
    df.bidSize=df.loc[:,'bidSize'].replace(to_replace=0, method='ffill')
    df.askPrice=df.loc[:,'askPrice'].replace(to_replace=0, method='ffill')
    df.askSize=df.loc[:,'askSize'].replace(to_replace=0, method='ffill')
    df=df.dropna()
    return df

def preprocessing_(df):
    df=df.dropna()
    # to exclude 0
    #data=data[data['bidPrice']>240]
    #data=data[data['askPrice']>240]
    df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['mid']=(df.askPrice+df.bidPrice)/2
    df['vwap']=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['high']=df.askPrice.rolling(60).max()
    df['low']=df.bidPrice.rolling(60).min()
    df=df.dropna()
    return df


def preprocessing(df):
    df=df.dropna()
    # to exclude 0
    #data=data[data['bidPrice']>240]
    #data=data[data['askPrice']>240]
    df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['Open']=(df.askPrice+df.bidPrice)/2
    df['Close']=df.Open.shift(1)
    df['High']=df.askPrice.rolling(10).max()
    df['Low']=df.bidPrice.rolling(10).min()
    df['Volume']=df.askSize+df.bidSize
    df['vwap']=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['change']=(df.Open-df.Close)
    #df['change']=df.change/df.change.rolling(60).mean()
    df['liq']=(df.askPrice-df.bidPrice)
    #df['liq']=df.liq/df.liq.rolling(60).mean()	
    df['spread']=(df.Open-df.vwap)
    df['vel']=df.Open-df.Open.shift(60)
    #df['vel']=df.vel/df.vel.rolling(60).mean()	
    #df['flow']=df.change*df.liq*df.Volume/df.Volume.rolling(60).mean()*df.vel
    df['return']=(df.askPrice/df.bidPrice.shift(10))-1
    #df['return']=(df.askPrice/df.bidPrice.shift(1))-1
    #df['ret'] = np.log(df.Close/df.Close.shift(1))
    #df['sigma']=df.spread.rolling(60).std()
    #df['sigma']=df.Close.rolling(5).std()
    #df['mom']=np.where(np.logical_and(df.vel_c==1,df.Close>df.price),1,np.where(np.logical_and(df.vel_c==-1,df.Close<df.price),-1,0))
    #flagD=np.logical_and(np.logical_and(df.Close.shift(10)<df.Close.shift(15),df.Close.shift(15)< df.Close.shift(20)),df.Close< df.Close.shift(10))
    #flagU=np.logical_and(np.logical_and(df.Close.shift(15)>df.Close.shift(20),df.Close.shift(10)> df.Close.shift(15)),df.Close> df.Close.shift(10))
    #df['UD']= np.where(flagU,-1,np.where(flagD,1,0))
    
    #df['P']=(df.High+df.Low+df.Close)/3
    #df['UT']=(pd.rolling_max(df.High,60)+pd.rolling_max(df.P+df.High-df.Low,60))*0.5
    #df['DT']=(pd.rolling_min(df.Low,60)+pd.rolling_min(df.P+df.High-df.Low,60))*0.5
    #df['BA']=np.where(df.Close<=df.DT,-1,np.where(df.Close>=df.UT,1,0))# below or above
    df=df.dropna()
    return df
"""
def normalise(df,window_length=60):
    dfn=(df-df.rolling(window_length).min())/(df.rolling(window_length).max()-df.rolling(window_length).min())
    return dfn

def de_normalise(data,df,window_length=60):
    dn=(df*(data.rolling(window_length).max()-data.rolling(window_length).min()))+data.rolling(window_length).min()
    return dn
"""
'''
def normalise(df,window_length=60):
    dfn=(df-df.rolling(window_length).min())/(df.rolling(window_length).max()-df.rolling(window_length).min())
    return dfn

def de_normalise(data,df,window_length=60):
    dn=(df*(data.rolling(window_length).max()-data.rolling(window_length).min()))+data.rolling(window_length).min()
    return dn

def normalise_z(df,window_length=12):
    dfn=(df-df.rolling(window_length).mean())/(df.rolling(window_length).std())
    return dfn

'''
def preprocessing_feb(df):
    df=df.dropna()
    # to exclude 0
    #data=data[data['bidPrice']>240]
    #data=data[data['askPrice']>240]
    df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['Open']=(df.askPrice+df.bidPrice)/2
    df['Close']=df.Open.shift(1)
    df['High']=df.askPrice.rolling(10).max()
    df['Low']=df.bidPrice.rolling(10).min()
    df['Volume']=df.askSize+df.bidSize
    df['trade']=df.askSize-df.bidSize
    df['liq']=(df.askPrice-df.bidPrice)
    df['vwap']=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['spread']=(df.Close-df.vwap)
    df['vel']=df.Close-df.Close.shift(60)
    df['return']=(df.askPrice/df.bidPrice.shift(12))-1
    df=df.dropna()
    return df

def preprocessing_mar(df):
    df=df.dropna()
    # to exclude 0
    #data=data[data['bidPrice']>240]
    #data=data[data['askPrice']>240]
    df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['Open']=(df.askPrice+df.bidPrice)/2
    df['Close']=df.Open.shift(1)
    df['High']=df.askPrice.rolling(10).max()
    df['Low']=df.bidPrice.rolling(10).min()
    df['Volume']=df.askSize+df.bidSize
    vwap=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['spread']=(df.Close-vwap)
    df['Up']=np.where(np.logical_and(df.Close.diff(300)>0.01,df.Close.diff(1)>0.0),1,0)
    df['Dn']=np.where(np.logical_and(df.Close.diff(300)<-0.01,df.Close.diff(1)<0.0),-1,0)
    df['UD']=np.where(np.logical_and(df.Close.diff(300)>0.01,df.Close.diff(1)>0.0),1,
                     np.where(np.logical_and(df.Close.diff(300)<-0.01,df.Close.diff(1)<0.0),-1,0))
    df=df.dropna()
    return df

def normalise(df,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','Open','Close','High','Low','Volume']]
    dfn=data/data.shift(60)
    return dfn

def de_normalise(dfn,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','Open','Close','High','Low','Volume']]
    data=dfn*data.shift(60)
    return data

#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

