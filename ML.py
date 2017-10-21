
# coding: utf-8

# In[5]:

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

context = zmq.Context()

#Forwarding ML output
socket_pub = context.socket(zmq.PUB)
socket_pub.bind('tcp://127.0.0.1:7010')

port = "7000"
# socket to talk to server
socket_sub = context.socket(zmq.SUB)
print ("Collecting & plotting stock prices.")
socket_sub.connect("tcp://localhost:%s" % port)

socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')


# In[6]:

df = pd.DataFrame()
pdf= pd.DataFrame()
final=pd.DataFrame()


# In[24]:

df_ml=pd.DataFrame()


# In[98]:

def preprocessing(df):
    df.bidPrice=df.loc[:,'bidPrice'].replace(to_replace=0, method='ffill')
    df.bidSize=df.loc[:,'bidSize'].replace(to_replace=0, method='ffill')
    df.askPrice=df.loc[:,'askPrice'].replace(to_replace=0, method='ffill')
    df.askSize=df.loc[:,'askSize'].replace(to_replace=0, method='ffill')
    #df=df.dropna()
    # to exclude 0
    #df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    #df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['mid']=(df.askPrice+df.bidPrice)/2
    df['vwap']=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['spread']=df.vwap-(df.askPrice+df.bidPrice)/2
    df['v']=(df.askPrice+df.bidPrice)/2-((df.askPrice+df.bidPrice)/2).shift(60)
    df['return']=(df.askPrice/df.bidPrice.shift(1))-1
    df['sigma']=df.spread.rolling(60).std()
    return df

def normalise(df,window_length=60):
    dfn=(df-df.rolling(window_length).min())/(df.rolling(window_length).max()-df.rolling(window_length).min())
    return dfn

def normalise_z(df,window_length=12):
    dfn=(df-df.rolling(window_length).mean())/(df.rolling(window_length).std())
    return dfn


def de_normalise(data,df,window_length=60):
    dn=(df*(data.rolling(window_length).max()-data.rolling(window_length).min()))+data.rolling(window_length).min()
    return dn

#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
##### ARIMA        

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
        
###ARIMA preprocessing
def arima_processing(df):
    #data=df[['vwap','mid']]
    #df=df.dropna()
    df['Lvwap']=np.log(df.vwap)
    df['Lmid']=np.log(df.mid)
    df['LDvwap']=df.Lvwap-df.Lvwap.shift(60)
    df['LDmid']=df.Lmid-df.Lmid.shift(60)
    #df=df.dropna()
    return df   

###Model is already saved from "/Dropbox/DataScience/ARIMA_model_saving.ipynb". Here loaded and added to "df_ml"
def ARIMA_(data):
    ### load model
    #data=data.dropna()
    #df=data[['Lvwap','Lmid']].tail(60)
    df_arima=data
    predictions_mid=ARIMA_mid(df_arima.LDmid)
    predictions_vwap=ARIMA_vwap(df_arima.LDvwap) 
    df_arima['predictions_mid']=np.exp(float(predictions_mid[-1])+df_arima.LDmid.shift(60))
    #df.predictions_mid=df.loc[:,'predictions_mid'].replace(to_replace='NaN', method='ffill')
    df_arima['predictions_vwap']=np.exp(float(predictions_vwap[-1])+df_arima.LDvwap.shift(60))

    df_ml['arima']=df_arima.predictions_mid+df_arima.mid-df_arima.predictions_vwap
    return df_arima.predictions_mid+df_arima.mid-df_arima.predictions_vwap
    
def ARIMA_mid(data):
    ### load model
    
    mid_arima_loaded = ARIMAResults.load('mid_arima.pkl')
    predictions_mid = mid_arima_loaded.predict()
    return predictions_mid

def ARIMA_vwap(data):
    ### load model
    vwap_arima_loaded = ARIMAResults.load('vwap_arima.pkl')
    predictions_vwap = vwap_arima_loaded.predict()
    return predictions_vwap   

def data_class(data):
    #df_ml=df_ml.dropna()
    data_cl=data.tail(len(df_ml))
    a= np.where(df_ml.mid>df_ml.km,1,0)
    b= np.where(df_ml.mid<df_ml.km,-1,0)
    c=np.where(df_ml.mid>df_ml.arima,1,0)
    d=np.where(df_ml.mid<df_ml.arima,-1,0)
    e=np.where(df_ml.mid>df_ml.REG,1,0)
    f=np.where(df_ml.mid<df_ml.REG,-1,0)
    g=np.where(df_ml.mid>df_ml.SVR,1,0)
    h=np.where(df_ml.mid<df_ml.SVR,-1,0)
    data_cl['U']=np.where(a[-1]*c[-1]*e[-1]*g[-1]==1,1,0)
    data_cl['D']=np.where(b[-1]*d[-1]*f[-1]*h[-1]==1,-1,0)
    data_cl=data_cl.dropna()
    return data_cl  

#### KALMAN moving average

##KF moving average
#https://github.com/pykalman/pykalman

# Import a Kalman filter and other useful libraries
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

def kalman_ma(data):
    #x=data.mid
    x=data.mid
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 248,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    df_ml['km']=state_means
    #return df_ml

### Linear Regression, sklearn, svm:SVR,linear_model
import pickle
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


## loading model saved from /Dropbox/DataScience/REG_model_saving.ipynb
filename_rgr = 'rgr.sav'
filename_svr = 'svr.sav'
# load the model from disk
loaded_rgr_model = pickle.load(open(filename_rgr, 'rb'))
loaded_svr_model = pickle.load(open(filename_svr, 'rb'))

def strat_lr(data,df):
    df=df.dropna()
    data=data.dropna()
    X=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    y=df.mid
    predict_regr=loaded_rgr_model.predict(X)
    predict_svr=loaded_svr_model.predict(X)
    df['predict_regr']=predict_regr
    df['predict_svr']=predict_svr
    df_ml['REG']=de_normalise(data.mid,df.predict_regr)
    df_ml['SVR']=de_normalise(data.mid,df.predict_svr)
    #return df_ml
    
#### loading classification model from /Dropbox/DataScience/ML_20Sep
filename_svm_model_up = 'svm_model_up.sav'
filename_lm_model_up = 'lm_model_up.sav'
filename_svm_model_dn = 'svm_model_dn.sav'
filename_lm_model_dn = 'lm_model_dn.sav'
# load the model from disk
loaded_svm_up_model = pickle.load(open(filename_svm_model_up, 'rb'))
loaded_lm_up_model = pickle.load(open(filename_lm_model_up, 'rb'))
loaded_svm_dn_model = pickle.load(open(filename_svm_model_dn, 'rb'))
loaded_lm_dn_model = pickle.load(open(filename_lm_model_dn, 'rb'))

def classification_up_dn(data):
    X=data[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    y1=data.U
    y2=data.D
    
    
    predict_svm_up=loaded_svm_up_model.predict(X)
    predict_lm_up=loaded_lm_up_model.predict(X)
    predict_svm_dn=loaded_svm_dn_model.predict(X)
    predict_lm_dn=loaded_lm_dn_model.predict(X)
    
    data['predict_svm_up']=predict_svm_up
    data['predict_lm_up']=predict_lm_up
    data['predict_svm_dn']=predict_svm_dn
    data['predict_lm_dn']=predict_lm_dn
    
    data['predict_svm']=data.predict_svm_up+data.predict_svm_dn
    data['predict_lm']=data.predict_lm_up+data.predict_lm_dn
    
    data['UD']=np.where(np.logical_and(data.predict_svm>0,data.predict_lm>0),1,np.where(np.logical_and(data.predict_svm<0,data.predict_lm<0),-1,0))  
       
    df_ml['UD']=data.UD

### LSTM

#df.loc[:, cols].prod(axis=1)
def lstm_processing(df):
    #df=df.dropna()
    df_price=df[['mid','vwap','arima','km','REG','SVR']]
    #normalization
    dfn=normalise(df_price,12)
    dfn['UD']=df.UD
    return dfn


import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import load_model
model = load_model('21sep.h5')

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[i:(i+look_back), 1]
        c = dataset[i:(i+look_back), 2]
        d = dataset[i:(i+look_back), 3]
        e=  dataset[i:(i+look_back), 4]
        f = dataset[i:(i+look_back), 5]
        g=  dataset[i:(i+look_back), 6]
        dataX.append(np.c_[b,c,d,e,f,g])
        #dataX.append(b)
        #dataX.append(c)
        #dataX.append(d)
        #dataX.append(e)
        #dataX.concatenate((a,bT,cT,dT,eT),axis=1)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX), np.array(dataY)


def strat_LSTM(df_lstm):
    
    #normalization
    #df_lstm=lstm_processing(df_ml)
    #df_lstm=df_lstm.dropna()
    dataset=df_lstm.values
    dataset = dataset.astype('float32')
    # reshape into X=t and Y=t+1
    look_back = 3
    X_,Y_ = create_dataset(dataset,look_back)
    
    # reshape input to be [samples, time steps, features]
    X_ = numpy.reshape(X_, (X_.shape[0],X_.shape[1],X_.shape[2]))
    # make predictions
    predict = model.predict(X_)
    df_lstm=df_lstm.tail(len(predict))
    df_lstm['LSTM']=predict

    #LSTM=(df_lstm.LSTM*(df_ml.mid.rolling(60).max()-df_ml.midClose.rolling(60).min()))+df_LSTM.Close.rolling(60).min()
    LSTM=de_normalise(df_ml.mid,df_lstm.LSTM,window_length=12)
    df_ml['LSTM']=LSTM
    
    return LSTM


# In[107]:

df_ml=pd.DataFrame()


# In[108]:

final=pd.DataFrame()


# In[114]:

## warm up upto preprocessing
#final=pd.DataFrame()
window=20
for _ in range(window):
#while True:
    iterations += 1
    string = socket_sub.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'Stock':sym,'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    df=preprocessing(df)
    df=df.tail(200)
    #df=df.tail(100)
    
    data=df[['askPrice','askSize','bidPrice','bidSize','mid','vwap','spread','v','return','sigma']]
    
    df_ml['mid']=df.mid
    df_ml['vwap']=df.vwap
    
    df_arima=arima_processing(df)
    df_ml['arima']=ARIMA_(df_arima)  
    
    dfn=normalise(data)
    #dfn=normalise_z(data)
    
    kalman_ma(data)
    strat_lr(data,dfn)
    
    df_cl=data_class(data)
    classification_up_dn(df_cl)
    
    df_lstm=lstm_processing(df_ml)
    df_lstm=strat_LSTM(df_lstm.tail(60))
    final['lstm']=df_lstm
    final['mid']=df.mid
    #final['stock']=df.Stock
    #final.insert(loc=0, column='Stock', value=df.Stock)
    
    #df_ml.insert(loc=0, column='Stock', value=df.Stock)
    
    #print(df.tail(1))
    #print(data.tail(1))
    #print(df_arima.tail(1))
    #print(df_ARIMA.tail(1))
    #print(dfn.tail(1))
    #print(df_cl.tail(1))
    #print(df_lstm[-1])
    print(df_ml.tail(1))
    #print(final.tail(1))
 
    
    
    #x = df_ml.to_string(header=False,index=False,index_names=False).split('\n')
    #socket_pub.send_string(x[-1])
    #print(x[-1]) 


# In[62]:

len(df_ml.dropna())


# In[103]:

