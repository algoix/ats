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

##### 7010 SUB
    port = "7010"
    socket_sub = context.socket(zmq.SUB)
    socket_sub.connect("tcp://localhost:%s" % port)
    socket_sub.setsockopt_string(zmq.SUBSCRIBE, u'SPY')
    
Subscribing to ML output forwarder for trading

