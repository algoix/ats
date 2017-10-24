from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARIMAResults

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


# Import a Kalman filter and other useful libraries
from pykalman import KalmanFilter
import numpy as np
import numpy
import pandas as pd
import matplotlib.pyplot as plt
from scipy import poly1d

def kalman_ma(data):
    x=data.price.tail(60)
    y=data.close.tail(60)
    # Construct a Kalman filter
    kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 246,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

    # Use the observed values of the price to get a rolling mean
    state_means, _ = kf.filter(x.values)
    state_means = pd.Series(state_means.flatten(), index=x.index)
    data['km']=state_means
    return data

def ARIMA_df(df):
    ### load model
    arima_model_loaded = ARIMAResults.load('sevennine_arima.pkl')
    predictions = arima_model_loaded.predict()
    #predictions =arima_model_loaded.fittedvalues
    #df['pr_arima']=np.exp(predictions+df.log.shift(60))    
    return predictions

from keras.models import load_model
model = load_model('sevensep.h5')

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        b = dataset[i:(i+look_back), 1]
        c = dataset[i:(i+look_back), 2]
        d = dataset[i:(i+look_back), 3]
        e=  dataset[i:(i+look_back), 4]
        f = dataset[i:(i+look_back), 5]
        g=  dataset[i:(i+look_back), 6]
        h=  dataset[i:(i+look_back), 7]
        dataX.append(np.c_[a,b,c,d,f,g,h])
        #dataX.append(b)
        #dataX.append(c)
        #dataX.append(d)
        #dataX.append(e)
        #dataX.concatenate((a,bT,cT,dT,eT),axis=1)
        dataY.append(dataset[i + look_back,4])
    return np.array(dataX), np.array(dataY)

def strat_LSTM(data):
    #data=preprocessing_df(df)
    #pr=strat_class(data)
    #data=data[['close','vel','sigma','P','pREG','predict_svm','predict_lm']]
    
    #data=data[['Close','km','logDiff','price','pREG','pSVR','UD']]
    #data=data.dropna()
    dataset = data.values
    dataset = dataset.astype('float32')

    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    # reshape into X=t and Y=t+1
    look_back = 3
    dataX, dataY = create_dataset(dataset,look_back)
    # reshape input to be [samples, time steps, features]
    dataX = numpy.reshape(dataX, (dataX.shape[0],dataX.shape[1],dataX.shape[2]))
    # make predictions
    Predict = model.predict(dataX)
    #plt.plot(dataY)
    #plt.plot(Predict)
    #plt.show()
    #return Predict
    return numpy.array(Predict), numpy.array(dataY)
