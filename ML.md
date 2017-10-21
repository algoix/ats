https://github.com/algoix/Quant_Trade/blob/L2P0/ML.py

##### use of sklearn

         #from sklearn.cross_validation import train_test_split
         from sklearn import linear_model
         from sklearn.svm import SVR
         from sklearn.preprocessing import MinMaxScaler
         from sklearn.model_selection import train_test_split
         from sklearn.svm import SVC

##### PUB-SUB
7000(sub), 7010(pub)

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

         



