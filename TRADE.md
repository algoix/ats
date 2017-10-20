# Trading with Streaming Data [Main]:: Connected to ML and PLOT

https://github.com/algoix/Quant_Trade/blob/L2P0/TRADE.py

##### Creating contract [next: portfolio]  
    conn = tpqib.tpqibcon()
    spy_contract = conn.create_contract('SPY', 'STK', 'SMART', 'SMART', 'USD')
spy contract is created.

##### 7000 and 7020 for PUB
    socket_pub = context.socket(zmq.PUB)
    socket_pub.bind('tcp://127.0.0.1:7000')
using **7000** for publishing IB data for ML and PLOT operation    

    socket_pub1.bind('tcp://127.0.0.1:7020')
**7020** for account,trading and holding information

##### 7010 for SUB
    port = "7010"
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect("tcp://localhost:%s" % port)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')
    
Subscribing to ML output forwarder for trading

#### Streaming market data from IB and PUB via 7000
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
            ...
        
        if field == 'askPrice':
            ...
        if field == 'askSize':
            ...

    request_id = conn.request_market_data(spy_contract, send_tick)        
        
    #if __name__ == '__main__':
    #    request_id = conn.request_market_data(spy_contract, send_bid_price)
    #conn.cancel_market_data(request_id)
    #conn.close()  
  
#### To get holding size and average size
    pos=conn.req_positions()
    average=pos[pos['sym']=='SPY'].tail(1).avgCost
    quantity=pos[pos['sym']=='SPY'].tail(1).quantity

#### TRADING
    Subscription to port:7010    
