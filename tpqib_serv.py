import tpqib
import datetime
import zmq
import pandas as pd

conn = tpqib.tpqibcon()

spy_contract = conn.create_contract('SPY', 'STK', 'SMART', 'SMART', 'USD')

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://127.0.0.1:7000')

field = ['lastTimestamp', 'askPrice', 'askSize',
         'bidPrice', 'bidSize',
         'low', 'high', 'close',
         'volume', 'lastPrice', 'lastSize', 'halted']

def send_tick(field, value):
    bid_price = ''
    bid_size=''
    ask_price=''
    ask_size=''
    #df = pd.DataFrame()
    
    if field == 'bidPrice':
        bid_price = value
        bid_size=0
        ask_price=0
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        print(msg)
        socket.send_string(msg)
        #dt = datetime.datetime.now()
        #df = df.append(pd.DataFrame({'Bid': float(value)}, index=[dt]))
    if field == 'bidSize':
        bid_price = 0
        bid_size= value
        ask_price=0
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        print(msg)
        socket.send_string(msg)
    if field == 'askPrice':
        bid_price = 0
        bid_size= 0
        ask_price =value
        ask_size=0
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        print(msg)
        socket.send_string(msg)
    if field == 'askSize':
        bid_price = 0
        bid_size= 0
        ask_price =0
        ask_size=value
        msg = 'SPY %s %s %s %s' %(bid_price,bid_size,ask_price,ask_size)
        print(msg)
        socket.send_string(msg)

request_id = conn.request_market_data(spy_contract, send_tick)