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


average=0
quantity=0
pos=conn.req_positions()
average=pos[pos['sym']=='SPY'].tail(1).avgCost
quantity=pos[pos['sym']=='SPY'].tail(1).quantity

# TRADING
port1 = "7040"
socket_sub1 = context.socket(zmq.SUB)
socket_sub1.connect("tcp://localhost:%s" % port1)
socket_sub1.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

port2 = "7010"
socket_sub2 = context.socket(zmq.SUB)
socket_sub2.connect("tcp://localhost:%s" % port2)
socket_sub2.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

df = pd.DataFrame()
df_val=pd.DataFrame()
     
        
if __name__ == '__main__':
    request_id = conn.request_market_data(spy_contract, send_tick) 

    #conn.cancel_market_data(request_id)
    #conn.close()      

    #conn.cancel_market_data(request_id)
    #conn.close()  

    while True:
      
        dt = datetime.datetime.now()

        string = socket_sub1.recv_string()
        sym,mid = string.split()
        df = df.append(pd.DataFrame({'Stock':sym,'mid': float(mid)},index=[dt]))

        ml=socket_sub.recv_string()
        sym,mid,REG,SVR,arima,km,LSTM,UD= ml.split()
        df_val = df_val.append(pd.DataFrame({'Stock':sym,'mid': float(mid),'REG':float(REG),'SVR':float(SVR),'arima': float(arima),'km': float(km),'LSTM':float(LSTM),'UD':float(UD)},index=[dt]))

        #global request_id
    
        #value=df.mid.tail(1)
        predicted_value=df_val.LSTM.tail(300)+(df.mid.tail(300)-df_val.mid.tail(300))
        predicted_mean=df_val.km.tail(300)+(df.mid.tail(300)-df_val.mid.tail(300))
        df=df.tail(300)
        df['pv']=predicted_value
        #df['pm']=predicted_mean
        df['UD']=df_val.UD.tail(300)
        df.to_csv('/home/octo/Dropbox/final_TRADE.csv', sep=',', encoding='utf-8')
    
        #final=df[['pv','UD']]
        #final.to_csv('/home/octo/Dropbox/ml_output.txt', sep=',', encoding='utf-8')
        pos=conn.req_positions()
        #print(pos)
        average=pos[pos['sym']=='SPY'].tail(1).avgCost
        quantity=pos[pos['sym']=='SPY'].tail(1).quantity
    
        #df_val['quantity']=float(quantity)
        #df_val['average']=float(average)
       
    
        x = final.to_string(header=False,index=False).split('\n')
        print(x[-1])
