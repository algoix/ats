
# coding: utf-8

# In[1]:

import tpqib
import datetime
import zmq
import pandas as pd
import numpy as np
import numpy
from numpy import inf


import matplotlib.pyplot as plt

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

print ("Collecting from <7010>.....")


# In[2]:

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

# generate Stream objects
stream_0 = Stream(
    token=stream_ids[0],
    maxpoints=150)
stream_1 = Stream(
    token=stream_ids[1],
    maxpoints=150)

# generate Scatter & Data objects
trace0 = Scatter(
    x=[], y=[],
    mode='lines+markers',
    stream=stream_0,
    name='price')
trace1 = Scatter(
    x=[], y=[],
    mode='lines+markers',
    stream=stream_1,
    name='predicted price')

dats = Data([trace0, trace1])

# generate figure object
layout = Layout(title='Streaming Plot')
fig = Figure(data=dats, layout=layout)
unique_url = py.plot(fig, filename='stream_plot', auto_open=False)

print('URL of the streaming plot:\n%s' % unique_url)

s0 = py.Stream(stream_ids[0])
s1 = py.Stream(stream_ids[1])

s0.open()
s1.open()


# In[3]:

context = zmq.Context()
# Subscribing to ML 
port = "7010"
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:%s" % port)
socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

port1 = "7000"
socket_sub1 = context.socket(zmq.SUB)
socket_sub1.connect("tcp://localhost:%s" % port1)
socket_sub1.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

# publishing for acount, holding and trading info
socket_pub1 = context.socket(zmq.PUB)
socket_pub1.bind('tcp://127.0.0.1:7040')


# In[4]:

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


# In[5]:

df = pd.DataFrame()
df_val=pd.DataFrame()


# In[ ]:

#window=20
#for _ in range(window):
while True:
    #iterations += 1
    # after forwarder's start
    
    string = socket_sub1.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'Stock':sym,'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    
    df=df.tail(50)
    preprocessing()
    #df.to_csv('C:\\Users\Michal\Desktop\df_.csv', sep=',', encoding='utf-8')
    ml=socket_sub.recv_string()
    sym,mid,REG,SVR,arima,km,LSTM,UD= ml.split()
    dt = datetime.datetime.now()
    df_val = df_val.append(pd.DataFrame({'Stock':sym,'mid': float(mid),'REG':float(REG),'SVR':float(SVR),'arima': float(arima),'km': float(km),'LSTM':float(LSTM),'UD':float(UD)},index=[dt]))
    
    predicted_value=df_val.LSTM.tail(360)+(df.mid.tail(360)-df_val.mid.tail(360))
    predicted_mean=df_val.km.tail(360)+(df.mid.tail(360)-df_val.mid.tail(360))
        #plotting
    dt = datetime.datetime.now()
    #s0.write({'x': str(dt)[11:-3], 'y': float(df['mid'].tail(1))})
    s0.write({'x': str(dt)[11:-3], 'y': float(df_val['km'].tail(1))+(float(df['mid'].tail(1))-float(df_val['mid'].tail(1)))})
    s1.write({'x': str(dt)[11:-3], 'y': float(df_val['LSTM'].tail(1))+(float(df['mid'].tail(1))-float(df_val['mid'].tail(1)))})
    
    #s0.write({'x': str(dt)[11:-3], 'y': float(predicted_mean.tail(1))})
    #s1.write({'x': str(dt)[11:-3], 'y': float(predicted_value.tail(1))})

        
    x = df[['Stock','mid']].to_string(header=False,index=False).split('\n')
    socket_pub1.send_string(x[-1])
    #print(x[-1])    


# In[ ]:

