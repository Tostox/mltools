import pandas as pd
import numpy as np


class TrendAnalysis():
    """ In this class are implemented a series of methods for extract the trend of a given time series
    and analyze the volatily of the data.

    Args:
    -----
        method (string)             >>>
        smooth (float), optional    >>>
        period (int), optional      >>>
        window_size (int), optional >>>
        delta (float), optional     >>>

    Note:
    -----
        The lowess regression methods has an high computational cost for value of smooth close to 1.
    """

    def __init__(self, method="linear", smooth=0.5, period=1, windows_size=2, delta=2):

        methods_avaible = ['linear', 'lowess', 'expanding_mean', "bollinger_bands"]

        if method not in methods_avaible:
            raise Exception("Invalid method. Use 'linear', 'lowess', 'expanding_mean', 'bollinger_bands'")
        if (smooth > 1) | (smooth < 0):
            raise Exception('The value of "smooth" for lowess regression must be between 0 and 1')

        self.method = method
        self.smooth = smooth
        self.period = period
        self.windows_size = windows_size
        self.delta = delta

    def linear(self, time_points, values):
        """ Extract trend using sk-learn linear regression.

        Args:
        -----
            time_points (ndarray datetime) >>> independent variable
            values (pandas.Series) >>> target variable

        Returns:
        --------
            pandas.Series where values are the result of fitting, index the timestamp
        """

        from sklearn import linear_model

        # encoding the timestamp into int values
        X = [i for i in range(0, len(time_points))]
        X = np.reshape(X, (len(X), 1))

        model = linear_model.LinearRegression()
        model.fit(X, values)
        # calculate trend
        trend = model.predict(X)

        return pd.Series(data=trend, index=time_points)

    def lowess(self, time_points, values):
        """ Extract trend using lowess regression.

        Args:
        -----
            time_points (ndarray datetime) >>> independent variable
            values (pandas.Series) >>> target variable

        Returns:
        --------
            pandas.Series where values are the result of fitting, index the timestamp
        """

        from statsmodels.nonparametric import smoothers_lowess

        trend = smoothers_lowess.lowess(endog=values, exog=time_points, return_sorted=False, frac=self.smooth)
        return pd.Series(data=trend, index=time_points)

    def expanding_mean(self, data):
        """ Extract trend using pandas expanding mean.

        Args:
        -----
            (pandas.Series) >>> independent variable

        Returns:
        --------
            pandas.Series where values are the result of fitting, index the timestamp
        """

        return data.expanding(min_periods=self.period).mean()

    def bollinger_bands(self, data):
        """ Extract the bollinger bands from the time series.

        Args:
        -----
            (pandas.Series) >>> independent variable

        Returns:
        --------
            tuple of two pandas.Series that corresponding at the upper and the lower band, index the timestamp
        """
        upper = data + self.delta * data.rolling(window=self.windows_size).std()
        lower = data - self.delta * data.rolling(window=self.windows_size).std()

        return (upper, lower)

    def fit(self, data):
        """ Methods used to apply the functions for the trend analysis to a pandas time series.

        Args:
        -----
            data >>> pandas Series. index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the results
        """

        if self.method == "linear":
            return self.linear(data.index, data.values)
        elif self.method == "lowess":
            return self.lowess(data.index, data.values)
        elif self.method == "expanding_mean":
            return self.expanding_mean(data)
        elif self.method == "bollinger_bands":
            return self.bollinger_bands(data)
        else:
            raise Exception("Invalid method")


def adf_test(data):
    """Pass in a time series, returns ADF report
    Args:
    -----
        data >>> pandas Series. index datetime64[ns], values the data observed.
    """

    from statsmodels.tsa.stattools import adfuller

    result = adfuller(data)
    print('Augmented Dickey-Fuller Test:')
    labels = ['ADF Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used']

    for value, label in zip(result, labels):
        print(label + ' : ' + str(value))

    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis, reject the null hypothesis.\
         Data has no unit root and is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
