### 1. Azure
##### IB paper account to start
##### zmq server to start
1. activate ALGO3
2. jupyter notebook
3. open tpqib_serv_paper.ipynb
##### Data will save at hard disk. Now ( August 2017) data is saving at desktop.
1. activate ALGO3
2. cd c:\Users\Michal\Dropbox\IB-ZMQ
3. python tpqib_data_save.py
Amibroker paper Account to run with AFL code. Data feed IB5. This system to start after running python codes.
### 2. Ubuntu
IB live account to start
zmq server to start
1. activate carnd-term1
2. jupyter notebook
3. open tpqib_serv.ipynb
ipynb at “Dropbox\IB-ZMQ” useful for streaming data based files

Details of the Routine
Opening python Jupyter notebook 
--
1.  activate ALGO3
2. jupyter notebook
3. files to open and save  in dropbox
### 3. Saving tick data 

##### R code
Tick data can be saved using R code or python code. R code saves in dat format. Handling date is problematic.`IB_download.R`
Data will  be stored at `C:/Users/Michal/Dropbox/IB_data/`. Above R code is to get live tick data from IB during market. This raw data not useful so need to process. Data will be stored in hard disk C:\Users\Michal\Dropbox\IB_data\  in .dat format during market. 
##### python
A python dataframe will be formed where columns are : askPrice askSize bidPrice bidSize Volume AvgVolume CallOpenInterest putOpenInterest CallVolume PutVolume ImpVol

IB_BAIVV.py  file is also stored at C:\Users\Michal\Dropbox\IB_BAIVV.py. This file can be used for get_data() function in Ipython notebook. use any:
%load C:\Users\Michal\Dropbox\IB_BAIVV.py
%loadpy C:\Users\Michal\Dropbox\IB_BAIVV.py
to load dataframe data:  data= get_data() 

Python code can save tick data in csv . 

tpqib_data_save.ipynb

Data will  be stored at C:/Users/Michal/Dropbox/IB_data/. 

### 4. Analysis
During market python code IB_data_analysis.ipynb will be used for analysis. 
https://gist.github.com/parthasen/2c0c6aca00d8fd005e9523f4d874ba41
https://github.com/parthasen/ALGO/blob/func/IB_BAIVV.py

