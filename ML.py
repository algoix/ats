
# coding: utf-8

# In[8]:

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
print ("Collecting from <7000> for ML['mid','vwap','arima','km','REG','SVR'].")
socket_sub.connect("tcp://localhost:%s" % port)

socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')


# In[13]:

def preprocessing():
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
    #return df


# In[110]:

def normalise(data,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]   
    dfn=data/data.shift(60)
    return dfn

def de_normalise(dfn,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    data=dfn*data.shift(60)
    return data


# In[140]:

##### ARIMA        

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults
        
###ARIMA preprocessing
def arima_processing():
    df['Lvwap']=np.log(df.vwap)
    df['Lmid']=np.log(df.mid)
    df['LDvwap']=df.Lvwap-df.Lvwap.shift(60)
    df['LDmid']=df.Lmid-df.Lmid.shift(60)
    #return df   

###Model is already saved from "/Dropbox/DataScience/ARIMA_model_saving.ipynb". Here loaded and added to "df_ml"
def ARIMA_():
    predictions_mid=ARIMA_mid(df.LDmid)
    predictions_vwap=ARIMA_vwap(df.LDvwap) 
    data=df.mid
    arima=df.mid.tail(1)+np.exp(predictions_vwap[-1]+df.LDvwap.shift(2).tail(1))-np.exp(predictions_mid[-1]+df.LDmid.shift(2).tail(1))
    data['arima']=arima
    return data

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

#### KALMAN moving average

##KF moving average
#https://github.com/pykalman/pykalman

# Import a Kalman filter and other useful libraries
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

def kalman_ma():
    x=df.mid
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
    x['km']=state_means
    return x

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

def strat_lr():
    X=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    X=X.dropna()
    y=df[['mid']]
    y=y.dropna()
    predict_regr=loaded_rgr_model.predict(X)
    predict_svr=loaded_svr_model.predict(X)
    X["nREG"]=predict_regr
    X['nSVR']=predict_svr
    
    #y["REG"]=X.nREG*df.mid.dropna().shift(60)
    #y["SVR"]=X.nSVR*df.mid.dropna().shift(60)

    return X

    
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

def classification_up_dn():
    X=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    X=X.dropna()
    predict_svm_up=loaded_svm_up_model.predict(X)
    predict_lm_up=loaded_lm_up_model.predict(X)
    predict_svm_dn=loaded_svm_dn_model.predict(X)
    predict_lm_dn=loaded_lm_dn_model.predict(X)
    
    predict_svm=predict_svm_up+predict_svm_dn
    predict_lm=predict_lm_up+predict_lm_dn
    predict= (float(predict_svm[-1])+float(predict_lm[-1]))
    X['UD']=predict
    return X

### LSTM

#df.loc[:, cols].prod(axis=1)
def lstm_processing(df_LSTM):
    
    df_price=df_LSTM[['mid','vwap','arima','km','REG','SVR']]
    #df_price=df_price.dropna()
    df_lstm=df_price/df_price.shift(60)
    df_lstm['UD']=df_LSTM.UD
    df_lstm=df_lstm.dropna()
    return df_lstm


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
model = load_model('28sep.h5')

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
        dataX.append(np.c_[b,c,d,e,f])
        #dataX.append(b)
        #dataX.append(c)
        #dataX.append(d)
        #dataX.append(e)
        #dataX.concatenate((a,bT,cT,dT,eT),axis=1)
        dataY.append(dataset[i + look_back,0])
    return np.array(dataX), np.array(dataY)


def strat_LSTM(df_LSTM):
    
    #normalization
    df_lstm_n=lstm_processing(df_LSTM)
    dataset=df_lstm_n.values
    dataset = dataset.astype('float32')
    # reshape into X=t and Y=t+1
    look_back = 3
    X_,Y_ = create_dataset(dataset,look_back)
    # reshape input to be [samples, time steps, features]
    X_ = numpy.reshape(X_, (X_.shape[0],X_.shape[1],X_.shape[2]))
    # make predictions
    predict = model.predict(X_)
    df_LSTM=df_LSTM.tail(len(predict))
    df_LSTM['nLSTM']=predict
    df_LSTM['LSTM']=(df_LSTM.nLSTM/df_LSTM.nLSTM.shift(60))*df_LSTM.mid
    
    return df_LSTM


# In[10]:

df = pd.DataFrame()


# In[144]:

print ("publishing to  <7010> for plot.")


# In[ ]:

## warm up upto preprocessing
#window=5
#for _ in range(window):
while True:
    iterations += 1
    string = socket_sub.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'Stock':sym,'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    df=df.tail(200)
    preprocessing()
    arima_processing()# 60 data points needed for this process.
    #dfn=normalise(df,60)
    #data=de_normalise(dfn,60)
    
    arima=ARIMA_()#ARIMA
    km=kalman_ma()#kalman
    UD=classification_up_dn()#classification
    RS=strat_lr()#regression
    #df[['mid','vwap','arima','km','REG','SVR']]
    df_LSTM=df[['mid','vwap']]
    df_LSTM['arima']=arima
    df_LSTM['km']=km
    df_LSTM['UD']=UD.UD
    df_LSTM['REG']=RS.nREG
    df_LSTM['SVR']=RS.nSVR
    LSTM=strat_LSTM(df_LSTM)
    final=LSTM[['mid','REG','SVR','arima','km','LSTM','UD']]
    final.insert(loc=0, column='Stock', value=df.Stock)
    
    
    #print(data.tail(1))
    #print(final.tail(1))
    x = final.to_string(header=False,index=False,index_names=False).split('\n')
    socket_pub.send_string(x[-1])
    print(x[-1]) 


# In[145]:

len(df)


# In[146]:

len(df.dropna())


# In[33]:

len(data)


# In[80]:

len(UD.dropna())


# In[77]:

df.mid.tail()


# In[147]:

len(final.dropna())


# In[92]:

RS.tail()


# In[139]:

LSTM.tail()


# In[ ]:

df_
