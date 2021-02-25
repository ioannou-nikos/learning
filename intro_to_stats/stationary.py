#tests that you can use to check if a time series is stationary or not.

data = [0,1,2,3,4,5,6,7,8,9]

def dickey_fuller(dt):
    """
    Tests whether a time series has a unit root, e.g. has a trend or 
    more generally is autoregressive
    """
    from statsmodels.tsa.stattools import adfuller
    stat, p, lags, obs, crit, t = adfuller(dt)
    print(f"stat={stat:.3f}, p={p:.3f}")
    if p > 0.05:
        print("Probably not Stationary")
    else:
        print("Probably Stationary")