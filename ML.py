
# coding: utf-8

# In[1]:

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


# In[2]:

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
    return df

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

# Import a Kalman filter and other useful libraries
from pykalman import KalmanFilter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

def kalman_ma(data):
    x=data.price.tail(60)
    y=data.Close.tail(60)
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 246,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    data['km']=state_means
    return data


# In[5]:

## warm up upto preprocessing

final=pd.DataFrame()
window=1000
for _ in range(window):
#while True:
    iterations += 1
    string = socket.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    data=preprocessing_df(df)
    #print(df.tail(1))
    print(data.tail(1))   


# In[10]:

#testing to delete inf and NaN
#df=df[100:]


# In[12]:

#checking
#df.head()


# In[6]:

import pickle
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# In[13]:

# saving linear model
df=df[1:].dropna()
X=df[['askPrice','askSize','bidPrice','bidSize','Close','U','D','sigma']]
y=df[['logDiff']]
regr = linear_model.LinearRegression()
regr_model=regr.fit(X,y)
regr_model = pickle.dumps(regr_model)
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.9) #kernel='linear' #kernel='poly'
svr_model = svr_rbf.fit(X, y)
svr_model = pickle.dumps(svr_model)


# In[14]:

# saving logistics and SVC model
df=df[1:].dropna()
X=df[['askPrice','askSize','bidPrice','bidSize','Close','price','sigma']]
y1=df[['U']]
y2=df[['D']]

svm = SVC(kernel='linear')
lm = linear_model.LogisticRegression(C=1e4)
svm_model_up= svm.fit(X,y1)
svm_model_up = pickle.dumps(svm_model_up)
lm_model_up= lm.fit(X, y1)
lm_model_up = pickle.dumps(lm_model_up)
svm_model_dn= svm.fit(X, y2)
svm_model_dn = pickle.dumps(svm_model_dn)
lm_model_dn= lm.fit(X, y2)
lm_model_dn = pickle.dumps(lm_model_dn)


# In[15]:

#loading regression model, first save the model
svr_model = pickle.loads(svr_model)
regr_model = pickle.loads(regr_model)

#loading classification model, first save the model
svm_model_up = pickle.loads(svm_model_up)
svm_model_dn = pickle.loads(svm_model_dn)
lm_model_up = pickle.loads(lm_model_up)
lm_model_dn = pickle.loads(lm_model_dn)


# In[16]:

def strat_lr(data):
    data=data.tail(60).dropna()
    X=data[['askPrice','askSize','bidPrice','bidSize','Close','U','D','sigma']]
    y=data[['logDiff']]
    predict_regr=regr_model.predict(X)
    predict_svr=svr_model.predict(X)
    dt=data[['Close']]
    dt['predict_regr']=predict_regr
    dt['predict_svr']=predict_svr
        
    pdf=data
    pdf['pREG']=np.exp(dt.predict_regr+data.log.shift(59))
    pdf['pSVR']=np.exp(dt.predict_regr+data.log.shift(59))
         
    #dt=data[['price','predict']]
    return pdf


# In[17]:

def classification_up_dn(data):
    X=data[['askPrice','askSize','bidPrice','bidSize','Close','price','sigma']]
    y1=data[['U']]
    y2=data[['D']]
    pr_df=data.tail(60)
    predict_svm_up=svm_model_up.predict(X.tail(60))
    pr_df['predict_svm_up']=predict_svm_up
    predict_lm_up=lm_model_up.predict(X.tail(60))
    pr_df['predict_lm_up']=predict_lm_up
    predict_svm_dn=svm_model_dn.predict(X.tail(60))
    pr_df['predict_svm_dn']=predict_svm_dn
    predict_lm_dn=lm_model_dn.predict(X.tail(60))
    pr_df['predict_lm_dn']=predict_lm_dn
    pr_df['predict_svm']=pr_df.predict_svm_up+pr_df.predict_svm_dn
    pr_df['predict_lm']=pr_df.predict_lm_up+pr_df.predict_lm_dn
    return pr_df


# In[40]:

def ARIMA_df(df):
    ### load model
    arima_model_loaded = ARIMAResults.load('sevennine_arima.pkl')
    predictions = arima_model_loaded.predict()
    #predictions =arima_model_loaded.fittedvalues
    #df['pr_arima']=np.exp(predictions+df.log.shift(60))    
    return predictions


# In[20]:

from keras.models import load_model
model = load_model('sevensep.h5')

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
        h=  dataset[i:(i+look_back), 7]
        dataX.append(np.c_[a,b,c,d,f,g,h])
        #dataX.append(b)
        #dataX.append(c)
        #dataX.append(d)
        #dataX.append(e)
        #dataX.concatenate((a,bT,cT,dT,eT),axis=1)
        dataY.append(dataset[i + look_back,4])
    return np.array(dataX), np.array(dataY)

def strat_LSTM(data):
    #data=preprocessing_df(df)
    #pr=strat_class(data)
    #data=data[['close','vel','sigma','P','pREG','predict_svm','predict_lm']]
    
    #data=data[['Close','km','logDiff','price','pREG','pSVR','UD']]
    dataset = data.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # reshape into X=t and Y=t+1
    look_back = 3
    dataX, dataY = create_dataset(dataset,look_back)
    # reshape input to be [samples, time steps, features]
    dataX = numpy.reshape(dataX, (dataX.shape[0],dataX.shape[1],dataX.shape[2]))
    # make predictions
    Predict = model.predict(dataX)
    #plt.plot(dataY)
    #plt.plot(Predict)
    #plt.show()
    #return Predict
    return numpy.array(Predict), numpy.array(dataY)


# In[ ]:

#window=20
#for _ in range(window):

while True:
    iterations += 1
    string = socket.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    data=preprocessing_df(df)
    pre_arima=ARIMA_df(data.logDiff)
    data['pr_arima']=np.exp(float(pre_arima[-1])+df.log.shift(60)) 
    data=kalman_ma(data)
    data=strat_lr(data)
    data=classification_up_dn(data)
    data['predict_svm']=data.predict_svm_up+data.predict_svm_dn
    data['predict_lm']=data.predict_lm_up+data.predict_lm_dn
    #data['UD']=np.where(data.predict_svm+data.predict_lm>0,1,np.where(data.predict_svm+data.predict_lm<0,-1,0))  
    data['UD']=np.where(np.logical_and(data.predict_svm>0,data.predict_lm>0),1,np.where(np.logical_and(data.predict_svm<0,data.predict_lm<0),-1,0))  
    #data=data.dropna()   
    data['spread']=data.Close-data.pr_arima
    df_LSTM= data[['askPrice','askSize','bidPrice','bidSize','Close','spread','pr_arima','sigma']]
    pr,y=strat_LSTM(df_LSTM)
    UD=pr[-1]-y[-1]
    final=final.append(pd.DataFrame({'LSTM':float(UD)},index=[dt]))
    output=data[['Close','price','km','pr_arima','pREG','pSVR','UD']]
    #output['spread']=data.pREG-data.SVR
    output['LSTM']=final.LSTM
    
    #plt.plot(pr)
    #plt.plot(y)
    #plt.plot(moving_average(y,60))
    #plt.plot(moving_average(pr,60))
    #plt.show()
     
    
    #print(df.tail(1))
    #print(data.tail(1))

    #print(data_arima.tail(1))

    #print(X.tail(1),y1.tail(1),y2.tail(1))
    #s0.write({'x': str(dt)[11:-3], 'y': float(output.error[-1])})
    
    print(output.tail(1))
    
    output.tail(1).to_csv('/home/octo/Dropbox/ml_output.txt', sep=',', encoding='utf-8')


# In[83]:

output


# In[77]:

dataX, dataY=create_dataset(df_LSTM, look_back=1)


# In[78]:

dataY


# In[ ]:


