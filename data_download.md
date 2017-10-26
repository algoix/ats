### Data Downloading from Interactive Broker
R code will be used for downloading data in .dat format.
library(IBrokers)
tws <- twsConnect()
fh <- file('C:/Users/Michal/Desktop/SPY.dat',open='a')
reqMktData(tws,twsEquity("SPY"), file=fh)
close(fh)
Above R code is to get live tick data from IB during market. This raw data not useful so need to process.

### Processing .dat IB data
Data will be stored in hard disk C:\Users\Michal\Dropbox\IB_data\  in .dat format during market. 
During market python code IB_data_analysis.ipynb will be used for analysis. 
https://gist.github.com/parthasen/2c0c6aca00d8fd005e9523f4d874ba41
https://github.com/parthasen/ALGO/blob/func/IB_BAIVV.py

A python dataframe will be formed where columns are : askPrice askSize bidPrice bidSize Volume AvgVolume CallOpenInterest putOpenInterest CallVolume PutVolume ImpVol

IB_BAIVV.py  file is also stored at C:\Users\Michal\Dropbox\IB_BAIVV.py. This file can be used for get_data() function in Ipython notebook. use any:
%load C:\Users\Michal\Dropbox\IB_BAIVV.py
%loadpy C:\Users\Michal\Dropbox\IB_BAIVV.py
to load dataframe data:  data= get_data() 
