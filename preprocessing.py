import zmq
import datetime
import pandas as pd
import numpy as np
import numpy
from numpy import inf

import json
import plotly_stream as plyst
import plotly.tools as plyt
import plotly.plotly as ply
#!pip install plotly
import tpqib
import datetime

#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import matplotlib.pyplot as plt

import pickle

iterations = 0
df = pd.DataFrame()
pdf= pd.DataFrame()
final=pd.DataFrame()

port = "7000"

# socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Collecting & plotting stock prices.")
socket.connect("tcp://localhost:%s" % port)

socket.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

def preprocessing_df(data):
    
    data.bidPrice=data.bidPrice.replace(to_replace=0, method='ffill')
    data.bidSize=data.bidSize.replace(to_replace=0, method='ffill')
    data.askPrice=data.askPrice.replace(to_replace=0, method='ffill')
    data.askSize=data.askSize.replace(to_replace=0, method='ffill')
    
    bp=np.nonzero(data.bidPrice==0)[0]
    data.bidPrice[bp]=data.bidPrice[bp-1]
    ap=np.nonzero(data.askPrice==0)[0]
    data.askPrice[ap]=data.askPrice[ap-1]
    bs=np.nonzero(data.bidSize==0)[0]
    data.bidSize[bs]=data.bidSize[bs-1]
    As=np.nonzero(data.askSize==0)[0]
    data.askSize[As]=data.bidSize[As-1]
    
    
    data['close']=(data.bidPrice+data.askPrice)/2
    data['price']=(data.bidPrice*data.bidSize+data.askPrice*data.askSize)/(data.bidSize+data.askSize)
    data['ret'] = np.log(data['close'] /data['close'].shift(1))
    data['vel'] = data['close'] -data['close'].shift(60)
    data['spread']=data.price -data.close
    data['sigma']=data.close.rolling(5).std()
    
    data['vel_c']=np.where(data.close>data.close.shift(60),1,np.where(data.close<data.close.shift(60),-1,0))
    data['mom']=np.where(np.logical_and(data.vel_c==1,data.close>data.price),1,np.where(np.logical_and(data.vel_c==-1,data.close<data.price),-1,0))
    flagD=np.logical_and(np.logical_and(data.close.shift(10)<data.close.shift(15),data.close.shift(15)< data.close.shift(20)),data.close< data.close.shift(10))
    flagU=np.logical_and(np.logical_and(data.close.shift(15)>data.close.shift(20),data.close.shift(10)> data.close.shift(15)),data.close> data.close.shift(10))
    #data['UD']= np.where(flagU,1,np.where(flagD,-1,0))
    data['U']= np.where(flagU*data.mom,1,0)
    data['D']= np.where(flagD*data.mom,-1,0)
    data['High']=data.askPrice.rolling(5).max()
    data['Low']=data.bidPrice.rolling(5).max()
    data['P']=(data.High+data.Low+data.close)/3
    data['UT']=(pd.rolling_max(data.High,60)+pd.rolling_max(data.P+data.High-data.Low,60))*0.5
    data['DT']=(pd.rolling_min(data.Low,60)+pd.rolling_min(data.P+data.High-data.Low,60))*0.5
    data['BA']=np.where(data.close<=data.DT,-1,np.where(data.close>=data.UT,1,0))
    data=data[['close','price','P','ret','vel','sigma','U','D','BA']]
    #data=data.dropna()
    return data
