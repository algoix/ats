https://github.com/algoix/Quant_Trade/blob/L2P0/ML.py

##### use of sklearn

         #from sklearn.cross_validation import train_test_split
         from sklearn import linear_model
         from sklearn.svm import SVR
         from sklearn.preprocessing import MinMaxScaler
         from sklearn.model_selection import train_test_split
         from sklearn.svm import SVC

##### PUB-SUB
7000(sub), 7010(pub). Publishing 7010 is subscribed by TRADE.

##### ARIMA
         from statsmodels.tsa.arima_model import ARIMA
         from statsmodels.tsa.arima_model import ARIMAResults

         from sklearn.preprocessing import MinMaxScaler
         from sklearn.model_selection import train_test_split

##### Import a Kalman filter and other useful libraries
         from pykalman import KalmanFilter
         import numpy as np
         import numpy
         import pandas as pd
         import matplotlib.pyplot as plt
         from scipy import poly1d

##### Loading model

Models are saved from datascience pages. Here in pipeline ARIMA and LSTM models are loaded
         
    arima_model_loaded = ARIMAResults.load('sevennine_arima.pkl')
    predictions = arima_model_loaded.predict(

    from keras.models import load_model
    model = load_model('sevensep.h5')
---

#### Changes
https://github.com/algoix/Quant_Trade/blob/3f419a48ca1242d62c06f6bd337f7cccb657e411/ML.py
1. pickle.dumps(regr_model) are used to save regression and classification model
2. pickle.loads(svr_model) are used in same pipeline for loading models
3. ARIMA and LSTM models are saved previously from datascience section here only loading for prediction
https://github.com/algoix/Quant_Trade/blob/4c8c1d37788172f775d51746c067d189fadf43e3/ML.py
1. Same structure but changes in functions but not worked well
https://github.com/algoix/Quant_Trade/blob/f3527739a3c1ff8fc62ee2105a7be9c8cf80799d/ML.py
1. Changes in normalization function and use of global dataframe df to store each algo result.
https://github.com/algoix/Quant_Trade/blob/1f3bf7c5f363eb5c32e4b334fb2382eb4b4b7182/ML.py
First link uses single page for ML,plot and trade. Except that all links fail to provide complete 'df'. This dev helps to get
a complete row but resulted a single row which is not desirable. 
https://github.com/algoix/Quant_Trade/blob/cb82fb3151ebe5c95af1f15bf351b3c243244061/ML.py
This dev results complete multiplw rows 


