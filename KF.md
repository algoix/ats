#### Import a Kalman filter and other useful libraries
  
    from pykalman import KalmanFilter
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from scipy import poly1d

#### Moving Average

1.estimation of rolling parameters of the data,there's no window length that we need to specify
2. Kalman filter and an n-day moving average to estimate the rolling mean of a dataset. We hope that the mean describes our observations well, so it shouldn't change too much when we add an observation; therefore, we assume that it evolves as a random walk with a small error term. The mean is the model's guess for the mean of the distribution from which measurements are drawn, so our prediction of the next value is simply equal to our estimate of the mean. We assume that the observations have variance 1 around the rolling mean, for lack of a better estimate. Our initial guess for the mean is 0, but the filter quickly realizes that that is incorrect and adjusts.    

        x=data.price.tail(1000)
        y=data.close.tail(1000)

        #Construct a Kalman filter
        kf = KalmanFilter(transition_matrices = [1],
                  observation_matrices = [1],
                  initial_state_mean = 0,
                  initial_state_covariance = 1,
                  observation_covariance=1,
                  transition_covariance=.01)

        # Use the observed values of the price to get a rolling mean
        state_means, _ = kf.filter(x.values)
        state_means = pd.Series(state_means.flatten(), index=x.index)

        # Compute the rolling mean with various lookback windows
        mean30 = x.rolling(window = 30).mean()
        mean60 = x.rolling(window = 60).mean()
        mean90 = x.rolling(window = 90).mean()

        # Plot original data and estimated mean
        plt.plot(state_means.tail(200))
        plt.plot(x.tail(200))
        plt.plot(mean30.tail(200))
        plt.plot(mean60.tail(200))
        plt.plot(mean90.tail(200))
        plt.title('Kalman filter estimate of average')
        plt.legend(['Kalman Estimate', 'X', '30 Moving Average', '60 Moving Average','90 Moving Average'])
        #plt.legend(['Kalman Estimate', 'X'])
        plt.xlabel('time')
        plt.ylabel('Price');
        plt.show()
        
https://github.com/algoix/Quant_Trade/blob/store/kf1.png
