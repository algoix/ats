
# coding: utf-8

# ### Collecting from IB, publishing and subscription to forward

# In[1]:

#import tpqib
import predicsenseIB
import datetime
import zmq
import pandas as pd

conn = predicsenseIB.tpqibcon()

spy_contract = conn.create_contract('SPY', 'STK', 'SMART', 'SMART', 'USD')

context = zmq.Context()

# publishing for ML and plotly
socket_pub = context.socket(zmq.PUB)
socket_pub.bind('tcp://127.0.0.1:7000')

# publishing for acount, holding and trading info
socket_pub1 = context.socket(zmq.PUB)
socket_pub1.bind('tcp://127.0.0.1:7020')

# Subscribing to forwarder for trading
port = "7010"
socket_sub = context.socket(zmq.SUB)
socket_sub.connect("tcp://localhost:%s" % port)
socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

field = ['lastTimestamp', 'askPrice', 'askSize',
         'bidPrice', 'bidSize',
         'low', 'high', 'close',
         'volume', 'lastPrice', 'lastSize', 'halted']
#,'position','accountSummary'


# In[2]:

# for streaming market data
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


# In[4]:

conn.cancel_market_data(request_id)
conn.close()   


# In[12]:

#pos=conn.req_positions().to_string(header=False,index=False,index_names=False).split('\n')
#socket_pub1.send_string(pos[-1])


# In[17]:

pos=conn.req_positions()


# In[25]:

conn.req_positions()


# In[38]:

average=pos[pos['sym']=='SPY'].tail(1).avgCost


# In[39]:

quantity=pos[pos['sym']=='SPY'].tail(1).quantity


# ### Trading

# In[62]:




# ##### Subscription to port:7010

# In[74]:

import numpy as np
df = pd.DataFrame()
## warm up upto preprocessing
#final=pd.DataFrame()
money = 120000
buy_cover_order = conn.create_order('MKT',400, 'Buy')
sell_Short_order = conn.create_order('MKT',400, 'Sell')
#window=10
#for _ in range(window):
while True:
    #iterations += 1
    # after forwarder's start
    ml=socket_sub.recv_string()
    sym,close,price,km,arima,UD,LSTM = ml.split()
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'close': float(close),'price': float(price),'km': float(km),'arima': float(arima),'LSTM':float(LSTM)},index=[dt]))
    
    global money
    global request_id
    
    value=float(df.close.tail(1))
        
    #if df.close>df.km and df.arima>df.arima.shift(1) and df.LSTM>df.LSTM.shift(1):
    if np.where(df.price.tail(1)>df.km.tail(1),1,0)==1 and np.where(df.close.tail(1)>df.arima.shift(1).tail(1),1,0)==1:
        
        if money==120000:
            print('400 shares cost %s USD, I have %s USD, so .. '% (400 * value, money))
            print('I buy!')
            conn.place_order(spy_contract, buy_cover_order)
            money = money - 400*value
        if  400* value> money:
            print('I quit')
            conn.cancel_market_data(request_id)
    
    if np.where(df.price.tail(1)<df.km.tail(1),1,0)==1 and np.where(df.km.tail(1)<df.arima.shift(1).tail(1),1,0)==1:
        print('I sell!')
        conn.place_order(spy_contract,sell_Short_order)
        #money = money - 400* value
        money = 120000

    if np.where(df.price.tail(1)<df.km.tail(1),1,0)==1 and np.where(df.close.tail(1)<df.arima.shift(1).tail(1),1,0)==1:
        
        if money==120000:
            print('400 shares cost %s USD, I have %s USD' % (400* value, money))
            print('I short!')
            conn.place_order(spy_contract,sell_Short_order)
            money = money - 400*value
        if 400 * value>money:
            print('I quit')
            conn.cancel_market_data(request_id)
    
    #if df.close<df.km and df.arima>df.arima.shift(1) and df.LSTM>df.LSTM.shift(1):
    if np.where(df.price.tail(1)>df.km.tail(1),1,0)==1 and np.where(df.km.tail(1)>df.arima.shift(1).tail(1),1,0)==1:
        print('I cover!')
        conn.place_order(spy_contract,buy_cover_order)
        money = 120000
      
    df['money']=money
    print(df.tail(1))


# In[ ]:


