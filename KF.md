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

    plt.plot(state_means[-200:])
    plt.plot(x[-200:])
    plt.plot(mean30[-200:])
    plt.plot(mean60[-200:])
    plt.plot(mean90[-200:])
    plt.title('Kalman filter estimate of average')
    plt.legend(['Kalman Estimate', 'X', '30-day Moving Average', '60-day Moving Average','90-day Moving Average'])
    plt.xlabel('Day')
    plt.ylabel('Price');
    plt.show()
https://github.com/algoix/Quant_Trade/blob/store/kf2.png

The advantage of the Kalman filter is that we don't need to select a window length, so we run less risk of overfitting. We do open ourselves up to overfitting with some of the initialization parameters for the filter, but those are slightly easier to objectively define. There's no free lunch and we can't eliminate overfitting, but a Kalman Filter is more rigorous than a moving average and generally better.

linear regression
    Let's try using a Kalman filter to find linear regression lines for a dataset. We'll be comparing a stock price with the S&P 500, so the result will be a sort of rolling alpha and beta for the stock, where  αα  and  ββ  are the parameters of the linear regression equation
yt≈α+βxt
yt≈α+βxt

Below we use colors to indicate the dates that the data points  (xt,yt)(xt,yt)  correspond to.

    # Plot data and use colormap to indicate the date each point corresponds to
    cm = plt.get_cmap('jet')
    colors = np.linspace(0.1, 1, len(x))
    sc = plt.scatter(x, y, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
    cb = plt.colorbar(sc)
    cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x)//9].index])
    plt.xlabel('SPY')
    plt.show()


Our inital guesses for these parameters is (0,0), with a covariance matrix (which describes the error of our guess) of all ones. As in the example of the rolling mean, we assume that our parameters follow a random walk (transition matrix is the identity) with a small error term (transition covariance is a small number times the identity).
To get from the state of our system to an observation, we dot the state  (β,α)(β,α)  with  (xi,1)(xi,1)  to get  βxi+α≈yiβxi+α≈yi , so our observation matrix is just a column of 1s glued to  xx . We assume that the variance of our observations  yy  is 2. Now we are ready to use our observations of  yy  to evolve our estimates of the parameters  αα  and  ββ .

    delta = 1e-3
    trans_cov = delta / (1 - delta) * np.eye(2) # How much random walk wiggles
    obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2, # y is 1-dimensional, (alpha, beta) is 2-dimensional
                  initial_state_mean=[0,0],
                  initial_state_covariance=np.ones((2, 2)),
                  transition_matrices=np.eye(2),
                  observation_matrices=obs_mat,
                  observation_covariance=2,
                  transition_covariance=trans_cov)

    # Use the observations y to get running estimates and errors for the state parameters
    state_means, state_covs = kf.filter(y.values)

    _, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(x.index, state_means[:,0], label='slope')
    axarr[0].legend()
    axarr[1].plot(x.index, state_means[:,1], label='intercept')
    axarr[1].legend()
    plt.tight_layout();
    plt.show()

 https://github.com/algoix/Quant_Trade/blob/store/kf3.png
 
