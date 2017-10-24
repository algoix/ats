import zmq
import datetime
import pandas as pd

iterations = 0
df = pd.DataFrame()
dt = datetime.datetime.now()
#df = pd.DataFrame({'bidPrice':200,'bidSize':200,'askPrice': 200,'askSize':200},index=[dt])
#h5s = pd.HDFStore('/home/octo/Downloads/YH/SPY' + '.h5s', 'w') 
#h5s = pd.HDFStore('C:\\Users\Michal\Dropbox\IB_data\SPY' + '.h5s', 'w')

port = "7000"

# socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print ("Collecting & plotting stock prices.")
socket.connect("tcp://localhost:%s" % port)

socket.setsockopt_string(zmq.SUBSCRIBE, u'SPY')

#for _ in range(500):
while True:
    iterations += 1
    string = socket.recv_string()
    sym, bidPrice,bidSize,askPrice,askSize = string.split()
    #print('%s %s %s %s %s' % (sym, bidPrice,bidSize,askPrice,askSize))
    dt = datetime.datetime.now()
    df = df.append(pd.DataFrame({'bidPrice': float(bidPrice),'bidSize': float(bidSize),'askPrice': float(askPrice),'askSize': float(askSize)},index=[dt]))
    df.bidPrice=df.bidPrice.replace(to_replace=0, method='ffill')
    df.bidSize=df.bidSize.replace(to_replace=0, method='ffill')
    df.askPrice=df.askPrice.replace(to_replace=0, method='ffill')
    df.askSize=df.askSize.replace(to_replace=0, method='ffill')
    #h5s['SPY'] = df
    #df.to_csv('C:\\Users\Michal\Dropbox\IB_data\SPY.csv', sep=',', encoding='utf-8')
    df.to_csv('C:\\Users\Michal\Desktop\SPY.csv', sep=',', encoding='utf-8')