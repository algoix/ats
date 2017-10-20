
# coding: utf-8

# In[95]:

import tpqib
import datetime
import zmq
import pandas as pd

conn = tpqib.tpqibcon()

spy_contract = conn.create_contract('SPY', 'STK', 'SMART', 'SMART', 'USD')

context = zmq.Context()

# publishing for ML and plotly
socket_pub = context.socket(zmq.PUB)
socket_pub.bind('tcp://127.0.0.1:7000')

# Subscribing to forwarder for trading
port = "7010"
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:%s" % port)
socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

field = ['lastTimestamp', 'askPrice', 'askSize',
         'bidPrice', 'bidSize',
         'low', 'high', 'close',
         'volume', 'lastPrice', 'lastSize', 'halted']

def send_tick(field, value):
    bid_price = ''
    bid_size=''
    ask_price=''
    ask_size=''
    df = pd.DataFrame()
    
    if field == 'bidPrice':
        bid_price = value
        bid_size=0
        ask_price=0
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        #print(msg)
        socket_pub.send_string(msg)
        
    if field == 'bidSize':
        bid_price = 0
        bid_size= value
        ask_price=0
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        #print(msg)
        socket_pub.send_string(msg)
        
    if field == 'askPrice':
        bid_price = 0
        bid_size= 0
        ask_price =value
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        #print(msg)
        socket_pub.send_string(msg)

    if field == 'askSize':
        bid_price = 0
        bid_size= 0
        ask_price =0
        ask_size=value
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        #print(msg)
        socket_pub.send_string(msg)


request_id = conn.request_market_data(spy_contract, send_tick)        
        
#if __name__ == '__main__':
#    request_id = conn.request_market_data(spy_contract, send_bid_price)

#conn.cancel_market_data(request_id)
#conn.close()      


# In[94]:

#conn.cancel_market_data(request_id)
#conn.close()


# ## Subscribing to forwarder [plot_forward.ipynb] for TRADING

# ## Trading Zone 

# In[ ]:

'''
#Warming up

df = pd.DataFrame()
## warm up upto preprocessing
#final=pd.DataFrame()

window=1000
for _ in range(window):
#while True:
    #iterations += 1
    # after forwarder's start
    ml=socket_sub.recv_string()
    sym,BP,BS,AP,AS,close,price,U,D,log,logDiff,sigma = ml.split()
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'BP':BP,'BS':BS,'AP':AP,'AS':AS,'close': float(close),'price': float(price),'U': float(U),'D': float(D),'log':log,'logDiff':logDiff,'sigma':sigma},index=[dt]))
    print(df.tail(1))
'''


# In[100]:

money = 50000
buy_10_order = conn.create_order('MKT',400, 'Buy')
sell_10_order = conn.create_order('MKT',400, 'Sell')


# In[ ]:


df = pd.DataFrame()
## warm up upto preprocessing
#final=pd.DataFrame()

window=1000
for _ in range(window):
#while True:
    #iterations += 1
    # after forwarder's start
    ml=socket_sub.recv_string()
    sym,close,price,km,arima,UD,LSTM = ml.split()
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'close': float(close),'price': float(price),'km': float(km),'arima': float(arima),'LSTM':LSTM,'UD':UD},index=[dt]))
    
    global money
    global request_id
    
    value=float(df.close.tail(1))
    
    #if df.close>df.km and df.arima>df.arima.shift(1) and df.LSTM>df.LSTM.shift(1):
    if np.where(df.close.tail(1)>df.km.tail(1),1,0)==1:
        print('10 shares cost %s USD, I have %s USD' % (10 * value, money))
        if 10 * value< money:
            print('I buy!')
            conn.place_order(spy_contract, buy_10_order)
            money = money - 10 *value
        else:
            print('I quit')
            conn.cancel_market_data(request_id)
    
    #if df.close<df.km and df.arima>df.arima.shift(1) and df.LSTM>df.LSTM.shift(1):
    if np.where(df.close.tail(1)<df.km.tail(1),1,0)==1:
        print('10 shares cost %s USD, I have %s USD, so .. '% (10 * value, money))
        if 10 *value< money:
            print('I buy!')
            conn.place_order(spy_contract, sell_10_order)
            money = money - 10 * value
        else:
            print('I quit')
            conn.cancel_market_data(request_id)    
    
    
    print(df.tail(1))


# In[91]:

'''
data = pd.DataFrame()
## warm up upto preprocessing
final=pd.DataFrame()

window=200
for _ in range(window):
#while True:
    #iterations += 1
    # after forwarder's start
    ml=socket_sub.recv_string()
    sym,BP,BS,AP,AS,close,price,U,D,log,logDiff,sigma = ml.split()
    dt = datetime.datetime.now()
    data =data.append(pd.DataFrame({'BP':BP,'BS':BS,'AP':AP,'AS':AS,'close': float(close),'price': float(price),'U': float(U),'D': float(D),'log':log,'logDiff':logDiff,'sigma':sigma},index=[dt]))
    
    pre_arima=ARIMA_df(data.logDiff)
    data['pr_arima']=np.exp(pre_arima[-1]+float(data.log.shift(1).tail(1)))

    data=kalman_ma(data)
    
    data=strat_lr(data)
    data=classification_up_dn(data)
    data['predict_svm']=data.predict_svm_up+data.predict_svm_dn
    data['predict_lm']=data.predict_lm_up+data.predict_lm_dn
    data['UD']=np.where(np.logical_and(data.predict_svm>0,data.predict_lm>0),1,np.where(np.logical_and(data.predict_svm<0,data.predict_lm<0),-1,0))  
    data['spread']=data.close-data.pr_arima
    #data=data.dropna()
    UD=0
    if len(data.dropna())>1:
        df_LSTM= data[['AP','AS','BP','BS','close','spread','pr_arima','sigma']]
        pr,y=strat_LSTM(df_LSTM)
        UD=pr[-1]-y[-1]
    final=final.append(pd.DataFrame({'LSTM':float(UD)},index=[dt]))
    #output=data[['Close','price','km','pr_arima','pREG','pSVR','UD']]
    output=data[['close','price','km','pr_arima','UD']]
    #output['spread']=data.pREG-data.SVR
    output['LSTM']=final.LSTM
   
    print(output.tail(1))
    #print(ar)
 '''   


# In[6]:

#df.head()


# In[34]:

#len(df)


# In[88]:

import pickle
#from sklearn.cross_validation import train_test_split
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# saving linear model
df=df[1:].dropna()
X=df[['AP','AS','BP','BS','close','U','D','sigma']]
y=df[['logDiff']]
regr = linear_model.LinearRegression()
regr_model=regr.fit(X,y)
regr_model = pickle.dumps(regr_model)
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.9) #kernel='linear' #kernel='poly'
svr_model = svr_rbf.fit(X, y)
svr_model = pickle.dumps(svr_model)

# saving logistics and SVC model
df=df[1:].dropna()
X=df[['AP','AS','BP','BS','close','price','sigma']]
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


# In[59]:


#loading regression model, first save the model
svr_model = pickle.loads(svr_model)
regr_model = pickle.loads(regr_model)

#loading classification model, first save the model
svm_model_up = pickle.loads(svm_model_up)
svm_model_dn = pickle.loads(svm_model_dn)
lm_model_up = pickle.loads(lm_model_up)
lm_model_dn = pickle.loads(lm_model_dn)


# In[87]:

def strat_lr(data):
    #data=data.tail(60).dropna()
    X=data[['AP','AS','BP','BS','close','U','D','sigma']]
    y=data[['logDiff']]
    predict_regr=regr_model.predict(X)
    predict_svr=svr_model.predict(X)
    dt=data[['close']]
    dt['predict_regr']=predict_regr
    dt['predict_svr']=predict_svr
        

    data['pREG']=np.exp(float(dt.predict_regr.tail(1))+float(data.log.shift(1).tail(1)))
    data['pSVR']=np.exp(float(dt.predict_regr.tail(1))+float(data.log.shift(1).tail(1)))
    return data

def classification_up_dn(data):
    X=data[['AP','AS','BP','BS','close','price','sigma']]
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


# In[ ]:

import json
import plotly_stream as plyst
import plotly.tools as plyt
import plotly.plotly as ply

import datetime
import time
import random
import pandas as pd
import plotly.plotly as py
import plotly.tools as tls 
from plotly.graph_objs import *
import cufflinks


# In[ ]:

pc = json.load(open('creds/plotly_creds.json', 'r'))

plyt.set_credentials_file(username=pc['username'], api_key=pc['api_key'])
plyst.plotly_stream.set_stream_tokens(pc['stream_ids'])
#!pip install twisted
# solving module 'twisted' has no attribute '__version__'
#!pip install --upgrade pyopenssl
pcreds = json.load(open('creds/plotly_creds.json', 'r'))
py.sign_in(pcreds['username'], pcreds['api_key'])
from autobahn.twisted.websocket import WebSocketClientProtocol,                                        WebSocketClientFactory
    
# plotly preparations

# get stream id from stream id list
stream_ids = pcreds['stream_ids']

# generate Stream object with maximum points 150
stream_0 = Stream(
    token=stream_ids[0],
    maxpoints=150)

# generate Scatter & Data objects
trace0 = Scatter(
    x=[], y=[],
    mode='lines+markers',
    stream=stream_0,
    name='price')

dats = Data([trace0])    


# In[ ]:

# generate figure object
layout = Layout(title=' ')
fig = Figure(data=dats, layout=layout)
unique_url = py.plot(fig, filename='stream_plot', auto_open=False)

print('URL of the streaming plot:\n%s' % unique_url)

s0 = py.Stream(stream_ids[0])
s0.open()
s1 = py.Stream(stream_ids[1])
s1.open()


# In[ ]:

df = pd.DataFrame()
## warm up upto preprocessing
#final=pd.DataFrame()

window=100
for _ in range(window):
#while True:
    #iterations += 1
    # after forwarder's start
    ml=socket_sub.recv_string()
    sym,BP,BS,AP,AS,close,price,U,D,log,logDiff,sigma = ml.split()
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'BP':BP,'BS':BS,'AP':AP,'AS':AS,'close': float(close),'price': float(price),'U': float(U),'D': float(D),'log':log,'logDiff':logDiff,'sigma':sigma},index=[dt]))
    print(df.tail(1))


# In[55]:

len(data)


# In[ ]:


