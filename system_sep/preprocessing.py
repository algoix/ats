import numpy as np
import pandas as pd
import math

from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.svm import SVC

df = pd.DataFrame()
pdf= pd.DataFrame()

def get_csv_pd(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd=pd.read_csv(path,sep=',',dtype={'askPrice':np.float32,'askSize':np.float32,
                                           'bidPrice':np.float32,'bidSize':np.float32},index_col=0,parse_dates=True)
    return spy_pd

def get_csv_pd_notime(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd = pd.read_csv(path, usecols=['askPrice','askSize','bidPrice','bidSize'], engine='python', skipfooter=3)
    return spy_pd
def preprocessing_df(df):
    #df.Stock=df.Stock
    df.bidPrice=df.bidPrice.replace(to_replace=0, method='ffill')
    df.bidSize=df.bidSize.replace(to_replace=0, method='ffill')
    df.askPrice=df.askPrice.replace(to_replace=0, method='ffill')
    df.askSize=df.askSize.replace(to_replace=0, method='ffill')
    df['Close']=(df.bidPrice+df.askPrice)/2
    df['price']=(df.bidPrice*df.bidSize+df.askPrice*df.askSize)/(df.bidSize+df.askSize)
    #velP=np.where(df.Close>df.Close.shift(60),1,0)
    #velN=np.where(df.Close<df.Close.shift(60),-1,0)
    #U=np.where(df.Close>df.price.rolling(60).max(),1,0)
    #D=np.where(df.Close<df.price.rolling(60).max(),-1,0)
    #df['U']= np.where(velP*U==1,1,0)
    #df['D']= np.where(velN*D==1,-1,0)
    #df['U']= np.where(velP==1,1,0)
    #df['D']= np.where(velN==1,-1,0)
    df['U']= np.where(df.Close>df.price,1,0)
    df['D']= np.where(df.Close<df.price,-1,0)
    df['log']=np.log(df.Close)
    #df['logDiff'] = df.log-df.log.rolling(60).mean()# almost 1 min
    df['logDiff'] = df.log-df.log.shift(60)# almost 1 min
    df['sigma']=df.log.rolling(60).std()
    data=df[['Stock','bidPrice','bidSize','askPrice','askSize','Close','price','U','D','log','logDiff','sigma']]
    return data
