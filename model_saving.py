
# coding: utf-8

# In[10]:

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import csv
import glob

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

import pickle
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



# In[2]:

filename = '/home/octo/Dropbox'+ '/SPY4Aug17.csv'


# In[3]:

# loading csv file
def get_csv_pd(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd=pd.read_csv(path,sep=',',dtype={'askPrice':np.float32,'askSize':np.float32,
                                           'bidPrice':np.float32,'bidSize':np.float32},index_col=0,parse_dates=True)
    #spy_pd = pd.read_csv(path, usecols=['askPrice','askSize','bidPrice','bidSize'], engine='python', skipfooter=3)
    return spy_pd
'''
def get_csv_pd_notime(path):
    #spy_pd=pd.read_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv',sep=' ',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    #spy_pd=pd.read_csv(path+'\SPY.csv',sep=',',names=['askPrice','askSize','bidPrice','bidSize'],index_col=0,parse_dates=True)
    spy_pd = pd.read_csv(path, usecols=['askPrice','askSize','bidPrice','bidSize'], engine='python', skipfooter=3)
    return spy_pd
'''

def preprocessing(df):
    df.bidPrice=df.loc[:,'bidPrice'].replace(to_replace=0, method='ffill')
    df.bidSize=df.loc[:,'bidSize'].replace(to_replace=0, method='ffill')
    df.askPrice=df.loc[:,'askPrice'].replace(to_replace=0, method='ffill')
    df.askSize=df.loc[:,'askSize'].replace(to_replace=0, method='ffill')
    df=df.dropna()
    # to exclude 0
    df=df[df['bidPrice']>df.bidPrice.mean()-df.bidPrice.std()]
    df=df[df['askPrice']>df.askPrice.mean()-df.askPrice.std()]
    df['mid']=(df.askPrice+df.bidPrice)/2
    df['vwap']=((df.loc[:,'bidPrice']*df.loc[:,'bidSize'])+(df.loc[:,'askPrice']*df.loc[:,'askSize']))/(df.loc[:,'bidSize']+df.loc[:,'askSize'])
    df['spread']=df.vwap-(df.askPrice+df.bidPrice)/2
    df['v']=(df.askPrice+df.bidPrice)/2-((df.askPrice+df.bidPrice)/2).shift(60)
    df['return']=(df.askPrice/df.bidPrice.shift(1))-1
    df['sigma']=df.spread.rolling(60).std()
    return df

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
def normalise(df,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    dfn=data/data.shift(60)
    return dfn

def de_normalise(dfn,window_length=60):
    data=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    data=dfn*data.shift(60)
    return data

#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


# In[5]:

# monkey patch around bug in ARIMA class
def __getnewargs__(self):
	return ((self.endog),(self.k_lags, self.k_diff, self.k_ma))
ARIMA.__getnewargs__ = __getnewargs__


def arima_processing(df):
    #data=df[['vwap','mid']]
    df=df.dropna()
    df['Lvwap']=np.log(df.vwap)
    df['Lmid']=np.log(df.mid)
    df['LDvwap']=df.Lvwap-df.Lvwap.shift(60)
    df['LDmid']=df.Lmid-df.Lmid.shift(60)
    df=df.dropna()
    return df  

def ARIMA_saving(data):
    data=data.dropna()
    data1=data.LDvwap
    data2=data.LDmid
    
    model_vwap = ARIMA(data1,order=(2,1,2))  # tested from ARIMA.ipynb
    #predictions = model.fit(disp=0).predict()
    predictions_vwap =model_vwap.fit(disp=0).fittedvalues
    # save model
    model_vwap.fit().save('vwap_arima.pkl')
    vwap_arima=np.exp(predictions_vwap+data.Lvwap.shift(60))
    
    model_mid = ARIMA(data2,order=(2,1,2))  # tested from ARIMA.ipynb
    #predictions = model.fit(disp=0).predict()
    predictions_mid =model_mid.fit(disp=0).fittedvalues
    # save model
    model_mid.fit().save('mid_arima.pkl')


# In[13]:

data=get_csv_pd(filename)
data=preprocessing(data)
data=data.dropna()
data=arima_processing(data)
data=data.dropna().tail(10000)
data= ARIMA_saving(data)


# ## No need to save KM model

# ## Linear Regression, sklearn, svm:SVR,linear_model

# In[15]:

data=get_csv_pd(filename)
data=preprocessing(data)
#data=normalise(data)
data=data.dropna()


# In[16]:

data.tail()


# In[17]:

df=data.tail(10000)
X=df[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
y=df.mid


# ### saving linear model

# In[18]:

regr = linear_model.LinearRegression()
regr_model=regr.fit(X,y)
# save the model to disk
filename_rgr = 'rgr.sav'
pickle.dump(regr_model, open(filename_rgr, 'wb'))

svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.9) #kernel='linear' #kernel='poly'
svr_model = svr_rbf.fit(X, y)
# save the model to disk
filename_svr = 'svr.sav'
pickle.dump(svr_model, open(filename_svr, 'wb'))


# ## Classification is based on the previous predictions, so need to build the dataframe df_ml

# In[19]:

###Model is already saved from "/Dropbox/DataScience/ARIMA_model_saving.ipynb". Here loaded and added to "df_ml"
def ARIMA_(data):
    ### load model
    data=data.dropna()
    predictions_mid=ARIMA_mid(data.LDmid)
    predictions_vwap=ARIMA_vwap(data.LDvwap) 
    vwap_arima=np.exp(predictions_vwap+data.Lvwap.shift(60))
    mid_arima=np.exp(predictions_mid+data.Lmid.shift(60))
    df_ml['arima']=data.mid+vwap_arima-mid_arima
    
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

def strat_lr(data):
    #no normalization
    
    data=data.dropna()
    X=data[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
    #y=df[['mid']]
    predict_regr=loaded_rgr_model.predict(X)
    predict_svr=loaded_svr_model.predict(X)
    
    df_ml['REG']=predict_regr
    df_ml['SVR']=predict_svr
    
    ## strat_lr(data,dfn) and below needed for normalized data
    #df_ml['predict_regr']=predict_regr
    #df_ml['predict_svr']=predict_svr
    #df_ml['REG']=de_normalise(data.mid,df.predict_regr)
    #df_ml['SVR']=de_normalise(data.mid,df.predict_svr)


# In[20]:

df_ml=pd.DataFrame()

#creating the ml dataset
data=get_csv_pd(filename)
data=preprocessing(data)
data=data.dropna()
dfn=normalise(data)
df_arima=arima_processing(data)
### prediction for last 60 points
data=data.dropna().tail(5000)
dfn=dfn.dropna().tail(5000)
df_arima=df_arima.dropna().tail(5000)

df_ml['mid']=data.mid
df_ml['vwap']=data.vwap


ARIMA_(df_arima)
kalman_ma(data)
strat_lr(data)


# In[21]:

len(df_ml)


# ## Classification

# In[32]:

df_ml=df_ml.dropna()
data_cl=data.tail(len(df_ml))
'''
a= np.where(df_ml.mid>df_ml.km,1,0)
b= np.where(df_ml.mid<df_ml.km,-1,0)
c=np.where(df_ml.mid>df_ml.arima,1,0)
d=np.where(df_ml.mid<df_ml.arima,-1,0)
e=np.where(df_ml.mid>df_ml.REG,1,0)
f=np.where(df_ml.mid<df_ml.REG,-1,0)
g=np.where(df_ml.mid>df_ml.SVR,1,0)
h=np.where(df_ml.mid<df_ml.SVR,-1,0)
'''
data_cl['U']=np.where(df_ml.mid>df_ml.vwap,1,0)
data_cl['D']=np.where(df_ml.mid<df_ml.vwap,-1,0)
data_cl=data_cl.dropna()


# In[33]:

df_ml.tail()


# ### saving classification model

# In[34]:

X=data_cl[['askPrice','askSize','bidPrice','bidSize','vwap','spread','v','return','sigma']]
y1=data_cl[['U']]
y2=data_cl[['D']]

svm = SVC(kernel='linear')
lm = linear_model.LogisticRegression(C=1e4)


# In[35]:

svm_model_up=svm.fit(X,y1)
lm_model_up=lm.fit(X,y1)
svm_model_dn=svm.fit(X,y2)
lm_model_dn =lm.fit(X,y2)


# In[36]:

# save the model to disk
filename_svm_model_up = 'svm_model_up.sav'
filename_lm_model_up = 'lm_model_up.sav'
filename_svm_model_dn = 'svm_model_dn.sav'
filename_lm_model_dn = 'lm_model_dn.sav'
pickle.dump(svm_model_up, open(filename_svm_model_up, 'wb'))
pickle.dump(lm_model_up, open(filename_lm_model_up, 'wb'))
pickle.dump(svm_model_dn, open(filename_svm_model_dn, 'wb'))
pickle.dump(lm_model_dn, open(filename_lm_model_dn , 'wb'))


# ## loading classification for LSTM model saving

# In[37]:

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


# In[38]:

classification_up_dn(data_cl)


# In[39]:

df_ml.tail()


# In[40]:

### LSTM

#df.loc[:, cols].prod(axis=1)
def lstm_processing(df):
    df=df.dropna()
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


# In[41]:

# Another function to handle normalization. normalizing and adding UD is not done rather nl.log() only of the 6 columns.
def lstm_processing(df):
    df=df.dropna()
    df_price=df[['mid','vwap','arima','km','REG','SVR']]
    #normalization
    dfn=np.log(df_price)
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


# In[42]:

#normalization
df_lstm=lstm_processing(df_ml)
df_lstm=df_lstm.dropna()
dataset=df_lstm.values
dataset = dataset.astype('float32')
# reshape into X=t and Y=t+1
look_back = 3
X_,Y_ = create_dataset(dataset,look_back)
    
# reshape input to be [samples, time steps, features]
X_ = numpy.reshape(X_, (X_.shape[0],X_.shape[1],X_.shape[2]))


# In[45]:

epochs=10
batch_size=50


# In[46]:

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back,5)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_,Y_, epochs, batch_size, verbose=2)


# In[47]:

model.save("28sep.h5")


# In[ ]:


