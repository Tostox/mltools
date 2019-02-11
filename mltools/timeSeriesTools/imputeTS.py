import pandas as pd
import numpy as np


def ReconstructTS(data, start, end, freq, interpolate = False, window = None):
    """ With this function we can reconstruct a time serie using pandas functionalities.

    Args:
    -----
        data (pandas Series or DataFrame)   >>> Original time series.
        start (string or datetime)          >>> Left bound for generating dates.
        end (string or datetime)            >>> Right bound for generating dates.
        freq (string)                       >>> Frequency strings can have multiples, e.g. '5H'.
        interpolate (boolean), optional     >>> If True, NaN values are replace by using linear interpolation.
        window (int), optional              >>> Maximum number of consecutive NaNs to fill. Must be greater than 0.
        
    Returns
    -------
        pandas DataFrame with datetime index.
    """

    complete_index = pd.date_range(start = start, end = end, freq=freq)
    complete_series = pd.DataFrame(index = complete_index)
    complete_series = pd.merge(left=complete_series, right=pd.DataFrame(data), how='left', left_index=True, right_index=True)

    if interpolate:
        complete_series = complete_series.interpolate(limit=window)

    return complete_series
