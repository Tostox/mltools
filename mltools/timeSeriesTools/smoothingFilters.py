import pandas as pd
import numpy as np


class AggregateData():
    """ This class compute the aggregation of a time series data by using a different methods, for a
    specific period of time. All of these methods work on a Pandas Series object.

    Args:
    -----

        interval (string) >>> Time interval which calculate the aggregate function.
        method (string) >>> It's possible to aggragate the data by mean, max, min and percentile. Default is mean
        percentile  (int) >>> Optional. Percentile, integer number beetween 0 and 100. Default is 0.5 (median).
    """

    def __init__(self, interval, method="mean", percentile=50):

        methods_avaible = ['mean', 'max', 'min', 'percentile']

        if method not in methods_avaible:
            raise Exception('Invalid method')
        if percentile <= 0 or percentile >= 100:
            raise Exception('Invalid percentile')

        self.method = method
        self.interval = interval
        self.percentile = percentile / 100

    def meanAgg(self, data):
        """
        Args:
        -----
            data (pandas.Series) >>> index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the mean calculated for the time interval specified
        """

        return data.resample(self.interval).mean()

    def maxAgg(self, data):
        """
        Args:
        -----
            data (pandas.Series) >>> index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the maximum calculated for the time interval specified
        """

        return data.resample(self.interval).max()

    def minAgg(self, data):
        """
        Args:
        -----
            data (pandas.Series) >>> index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the mininum calculated for the time interval specified
        """

        return data.resample(self.interval).min()

    def percentileAgg(self, data):
        """
        Args:
        -----
            data (pandas.Series) >>> index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the percentile calculated for the time interval specified
        """

        return data.resample(self.interval).apply(lambda x: x.quantile(self.percentile))

    def fit(self, data):
        """ Methods used to apply a specific aggregation function to a pandas time series.

        Args:
        -----
            data (pandas.Series) >>> index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the results of the specifc aggregation method invoke
        """

        if self.method == "mean":
            return self.meanAgg(data)
        elif self.method == "max":
            return self.maxAgg(data)
        elif self.method == "min":
            return self.minAgg(data)
        elif self.method == "percentile":
            return self.percentileAgg(data)
        else:
            raise Exception("Invalid method")


class MovingAverage():

    """ This class include a methods for the calculus of the moving average ('simple' and 'exponential') for a time series

    Args:
    -----
        method (string)     >>> Method for the calculus of the moving average, it can be 'simple' or 'exponential'.
                                Default is 'simple'.

        window_size (int)   >>> Time window size. Default is equal to 2.
    """

    def __init__(self, method="simple", window_size=2):

        methods_avaible = ['simple', 'exponential']

        if method not in methods_avaible:
            raise Exception('Invalid method')
        if window_size <= 1:
            raise Exception('window_size too short')

        self.method = method
        self.window_size = window_size

    def simple(self, data):
        """ Computes moving average using discrete linear convolution of two one dimensional sequences.

        Args:
        -----
            data (pandas.Series) >>> independent variable

        Returns:
        --------
            pandas.Series
        """

        return data.rolling(self.window_size).mean()

    def exponential(self, data):
        """ Computes exponential moving average

        Args:
        -----
            data (pandas.Series) >>> independent variable

        Returns:
        --------
            pandas.Series
        """

        weights = np.exp(np.linspace(-1., 0., self.window_size))
        weights /= weights.sum()
        exp_ma = np.convolve(data, weights, mode='full')[:len(data)]
        exp_ma[:self.window_size] = np.NaN
        return pd.Series(data=exp_ma, index=data.index)

    def fit(self, data):
        """ Methods used to apply a specific moving average function to a pandas time series.

        Args:
        -----
            data >>> pandas Series. index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the results of the specifc method invoke.
        """

        if self.method == "simple":
            return self.simple(data)
        elif self.method == "exponential":
            return self.exponential(data)
        else:
            raise Exception("Invalid method")


class SavGol_smoothing():
    """ This class include a methods for the calculus of the Savitzky-Golay filter

    Args:
    -----
        polyorder (int)     >>> Order of the polynomial used for the interpolation. Default is 1.
        window_size (int)   >>> Time window size. Default is equal to 2.
        deriv (int)         >>> Order of the derivate. If greater than 0, the method return the series of the derivates.
                                Default is 0.
    """

    def __init__(self, polyorder=1, window_size=2, deriv=0):

        if window_size <= 1:
            raise Exception('window_size too short')

        self.polyorder = polyorder
        self.window_size = window_size
        self.deriv = deriv

    def sav_gol(self, data):
        """ Computes Sav-Gol smoothing filter

        Args:
        -----
            data (pandas.Series) >>> independent variable

        Returns:
        --------
            pandas.Series
        """

        from scipy.signal import savgol_filter

        return pd.Series(data=savgol_filter(data, polyorder=self.polyorder,
                                            window_length=self.window_size,
                                            deriv=self.deriv),
                         index=data.index)

    def fit(self, data):
        """ Methods used to apply the Sav-Gol filter function to a pandas time series.

        Args:
        -----
            data >>> pandas Series. index datetime64[ns], values the data observed.

        Returns:
        --------
            pandas.Series with the results
        """

        return self.sav_gol(data)
