import pandas as pd
import numpy as np
import scipy
import statsmodels.api as sm
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pylab as plt
import seaborn as sns
from itertools import *

# Models
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from fbprophet import Prophet

# Metric
from sklearn.metrics import mean_squared_error

import warnings

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')


class ForecastingTS():
    """
        This class implements three models (SARIMAX, HWES and Prophet) for a time series forecasting process;
        forecasting involves taking models fit on historical data and using them to predict future observations.
        For this aim, we fit each model on a training set and evaluate it on a test set, which represents 20% of the
        data by default; this proportion can be changed when we fit the model.
        We estimate the performance of predictions using the mean squared error (MSE).

        Args:
        -----
            model (string)          >>> Optional. The model that we want to compute.
                                        Choose between: 'SARIMAX', 'HWES' and 'Prophet'.
                                        By default is set to 'SARIMAX'.

            model_params (dict)     >>> Optional. Dictionary that contain the additional parameters for the models.

            1) SARIMAX:
                exog (array_like)       >>> Optional. Array of exogenous regressors.
                                            By default is set to None.

                order (tuple)           >>> Optional. The (p,d,q) order of the model for the number of AR
                                            parameters, differences, and MA parameters. 'd' must be an integer
                                            indicating the integration order of the process, while 'p' and 'q'
                                            may either be an integers indicating the AR and MA orders (so that
                                            all lags up to those orders are included) or else iterables giving
                                            specific AR and / or MA lags to include.
                                            Default is an AR(1) model: (1,0,0).

                seasonal_order (tuple)  >>> Optional. The (P,D,Q,s) order of the seasonal component of the model
                                            for the AR parameters, differences, MA parameters, and periodicity.
                                            'd' must be an integer indicating the integration order of the process,
                                            while 'p' and 'q' may either be an integers indicating the AR and MA
                                            orders (so that all lags up to those orders are included) or else
                                            iterables giving specific AR and / or MA lags to include. 's' is an
                                            integer giving the periodicity (number of periods in season), often it
                                            is 4 for quarterly data or 12 for monthly data. Default is no seasonal
                                            effect.

                about the other parameters, look at: https://www.statsmodels.org/dev/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html

            2) HWES:
                trend (string)          >>> Optional. Type of trend component: {"add", "mul", "additive",
                                            "multiplicative", None}. By default is set to None.

                seasonal (string)       >>> Optional. Type of seasonal component: {"add", "mul", "additive",
                                            "multiplicative", None}. By default is set to None.

                seasonal_periods (int)  >>> Optional. The number of seasons to consider for the holt winters.
                                            Default is 12.

            3) Prophet:
                growth (string)         >>> Optional. Can be 'linear' or 'logistic' to specify a linear or
                                            logistic trend. If growth is 'logistic', then df must also have a
                                            column cap that specifies the capacity at each ds.

                yearly_seasonality      >>> Optional. Fit yearly seasonality. Can be 'auto', True, False, or a
                                            number of Fourier terms to generate. Default is 'auto'.

                weekly_seasonality      >>> Optional. Fit weekly seasonality. Can be 'auto', True, False, or a
                                            number of Fourier terms to generate. Default is 'auto'.

                daily_seasonality       >>> Optional. Fit daily seasonality. Can be 'auto', True, False, or a
                                            number of Fourier terms to generate. Default is 'auto'.

                seasonality_mode        >>> Optional. Can be 'additive' or 'multiplicative'. Default is 'additive'.

                seasonality_prior_scale >>> Optional. Parameter modulating the strength of the seasonality model.
                                            Larger values allow the model to fit larger seasonal fluctuations,
                                            smaller values dampen the seasonality. Can be specified for individual
                                            seasonalities using add_seasonality. Default is 10.

                interval_width (float)  >>> Optional. Width of the uncertainty intervals provided for the forecast.
                                            Default is 0.8.

                about the other parameters, look at: https://github.com/facebook/prophet/blob/master/python/fbprophet/forecaster.py
    """
    models_available = ['SARIMAX', 'HWES', 'Prophet']

    def __init__(self, model='SARIMAX', model_params={}):

        self.model = model
        self.model_params = model_params
        self.residuals = None

        if model not in self.models_available:
            raise Exception('Invalid model. Choose a model in: {}'.format(self.models_available))

    def fit(self, series, gridsearch=False, prop=0.80, fit_params={}):
        """
            This function fits the forecasting model specified during the class initialization.

            Args:
            -----
                series (pandas Series)  >>> Series with index datetime64[ns] and values the observed time-series
                                            process.

                prop (float)            >>> Optional. Training set proportion. By default is set to 0.80.

                gridsearch (bool)       >>> Optional. If True, we performe hyperparameter optimization with a
                                            grid search methodology. By default is set to False.

                fit_params (dict)       >>> Optional. Dictionary that contain the additional parameters for fit.

                    1) SARIMAX:
                        s (int)                 >>> Optional. It is an integer giving the periodicity (number of
                                                    periods in season). Used only in the grid search hyperparameter
                                                    optimization for the model selection. Default is 12.

                        max_value (int)         >>> Optional. Maximum value of the interval defined for the
                                                    optimization of p,d and q. Used only in the grid search
                                                    hyperparameter optimization for the model selection.
                                                    Default is 2.

                        enforce_stat (bool)     >>> Optional. Whether or not to transform the AR parameters to
                                                    enforce stationarity in the AR component of the model.
                                                    Default is False.

                        enforce_invert (bool)   >>> Optional. Whether or not to transform the MA parameters to
                                                    enforce invertibility in the MA component of the model.
                                                    Default is False.

                    2) HWES:
                        step (float)            >>> Optional. Step for the half-open interval [start, stop) defined
                                                    for the optimization of alpha, beta and gamma. Used only in the
                                                    grid search hyperparameter optimization for the model selection.
                                                    Default is 0.1.

                        smoothing_level         >>> Optional. The smoothing level value of the simple exponential
                        (float)                     smoothing, if the value is set then this value will be used as
                                                    the value. By default is set to None.

                        smoothing_slope (float) >>> Optional. The smoothing slope value of the Holts trend method,
                                                    if the value is set then this value will be used as the value.
                                                    By default is set to None.

                        smoothing_seasonal      >>> Optional. The smoothing seasonal value of the holt winters
                        (float)                     seasonal method, if the value is set then this value will be
                                                    used as the value. By default is set to None.

                        damping_slope (float)   >>> Optional. The phi value of the damped method, if the value is
                                                    set then this value will be used as the value.
                                                    By default is set to None.

                    3) Prophet:
                        cap                     >>> Optional. To make forecasts using a logistic growth
                        (float, int or pd.Series)   trend model, it's necessary to specify the carrying
                                                    capacity. This parameter is not necessarily constant
                                                    and may vary over time. You can assume either a
                                                    particular value or set it using data (time series
                                                    percentile, for example). Default is None.

            Returns:
            --------
                The scores computed on the test set. The model instance is stored into the variable results.
        """
        self.series = series
        self.prop = prop

        if self.model == 'SARIMAX':
            return self._sarimax(self.series, self.prop, gridsearch, **fit_params)
        elif self.model == 'HWES':
            return self._hwes(self.series, self.prop, gridsearch, **fit_params)
        elif self.model == 'Prophet':
            return self._prophet(self.series, self.prop, gridsearch, **fit_params)
        else:
            raise Exception("Invalid method")

    def _sarimax(self, series, prop, gridsearch, s=12, max_value=2, enforce_stat=False, enforce_invert=False):
        """
            Compared with the basic ARIMA model ('Auto Regressive Integrated Moving Average'), SARIMAX has two
            distinct features:
                1) A seasonal component;
                2) Exogenous variables that exert influence on time series values.

            Using the 'global' SARIMAX model from statsmodels, we can also compute the AR, MA, ARMA, ARIMA and
            SARIMA models setting parameters ('exog', 'order' and 'seasonal_order') in the right way.
            For example, setting exog = None, we obtain the seasonal ARIMA model, which incorporates both
            non-seasonal and seasonal factors in a multiplicative model. One shorthand notation for the model is
            ARIMA(p, d, q) × (P, D, Q, s), where 's' is the time span of repeating seasonal pattern.
            More information: https://otexts.org/fpp2/seasonal-arima.html.

            Args:
            -----
                series (pd.Series)      >>> Series with index datetime64[ns] and values the observed time-series
                                            process.

                prop (float)            >>> Optional. Training set proportion. By default is set to 0.80.

                s (int)                 >>> Optional. It is an integer giving the periodicity (number of periods
                                            in season). Used only in the grid search hyperparameter optimization
                                            for the model selection. Default is 12.

                max_value (int)         >>> Optional. Maximum value of the interval defined for the optimization
                                            of p,d and q. Used only in the grid search hyperparameter optimization
                                            for the model selection. Default is 2.

                enforce_stat (bool)     >>> Optional. Whether or not to transform the AR parameters to enforce
                                            stationarity in the AR component of the model. Default is False.

                enforce_invert (bool)   >>> Optional. Whether or not to transform the MA parameters to enforce
                                            invertibility in the MA component of the model. Default is False.

            Returns:
            --------
                The scores computed on the test set.
        """

        # Train-test split
        train_size = int(len(series) * prop)
        train = series[:train_size]
        test = series[train_size:]

        """-------------------------- FITTING --------------------------"""
        if gridsearch:
            print('Grid search for the SARIMAX model...')
            p = d = q = range(0, max_value)

            # Generate all different combinations of p, q and q triplets
            pdq = list(product(p, d, q))

            # Generate all different combinations of seasonal p, q and q triplets
            seasonal_pdq = [(x[0], x[1], x[2], s) for x in pdq]

            bestAIC = np.inf
            bestParam = None
            bestSParam = None

            # Grid search (or hyperparameter optimization) for model selection.
            for param in pdq:
                for param_seasonal in seasonal_pdq:
                    try:
                        model = SARIMAX(train, order=param, seasonal_order=param_seasonal,
                                        enforce_stationarity=enforce_stat, enforce_invertibility=enforce_invert,
                                        **self.model_params)
                        results = model.fit()

                        if results.aic < bestAIC:
                            bestAIC = results.aic
                            bestParam = param
                            bestSParam = param_seasonal

                    except Exception:
                        continue

            print('Finished parameters optimization...\n')
            print('The best (p,d,q) x (P,D,Q,{}) are:'.format(s), bestParam, 'x', bestSParam)
            print('The best AIC score is:', bestAIC)

            model = SARIMAX(train, order=bestParam, seasonal_order=bestSParam, enforce_stationarity=enforce_stat,
                            enforce_invertibility=enforce_invert, **self.model_params)
            results = model.fit()

        else:
            print('No Grid search optimization for the SARIMAX model...\n')
            model = SARIMAX(train, enforce_stationarity=enforce_stat, enforce_invertibility=enforce_invert,
                            **self.model_params)
            results = model.fit()
            print('Finished...')

        self.results = results
        self.residuals = results.resid

        """-------------------------- EVALUATION --------------------------"""
        y_pred = results.get_prediction(start=test.index[0], end=test.index[-1])
        y_true = series[test.index[0]:]
        mse = mean_squared_error(y_true, y_pred.predicted_mean)
        output = pd.Series()
        output['AIC'] = results.aic
        output['BIC'] = results.bic
        output['mse'] = mse

        return output

    def _hwes(self, series, prop, gridsearch, step=0.1):
        """
            The HWES (Holt-Winter's Exponential Smoothing) model is suitable univariate time series with trend
            and/or seasonal components. The Holt-Winters seasonal method, that is an extension of the Holt’s method
            to capture seasonality, comprises the forecast equation and three smoothing equations — one for the
            level, one for the trend and one for the seasonal component, with corresponding smoothing parameters
            α, β and γ. We use a 'seasonal_periods' parameter to denote the frequency of the seasonality, i.e.,
            the number of seasons in a year. There are two variations to this method that differ in the nature of
            the seasonal component. The additive method is preferred when the seasonal variations are roughly
            constant through the series, while the multiplicative method is preferred when the seasonal variations
            are changing proportional to the level of the series.
            More information: https://otexts.org/fpp2/holt-winters.html

            Args:
            -----
                series (pd.Series)  >>> Series with index datetime64[ns] and values the observed time-series
                                        process.

                prop (float)        >>> Optional. Training set proportion. By default is set to 0.80.

                step (float)        >>> Optional. Step for the half-open interval [start, stop) defined for the
                                        optimization of alpha, beta and gamma. Used only in the grid search
                                        hyperparameter optimization for the model selection. Default is 0.1.

            Returns:
            --------
                The scores computed on the test set.
        """

        # Train-test split
        train_size = int(len(series) * prop)
        train = series[:train_size]
        test = series[train_size:]

        """-------------------------- FITTING --------------------------"""
        if gridsearch:
            print('Grid search for the HWES model...')
            # Generate all different combinations of parameters
            alpha = beta = gamma = [round(i, 2) for i in np.arange(0.01, 1, step)]
            trend = seasonal = ['add', 'mul']
            params = list(product(alpha, beta, gamma, trend, seasonal))

            bestAIC = np.inf
            bestalpha = bestbeta = bestgamma = besttrend = bestseasonal = bestBIC = None

            # Grid search (or hyperparameter optimization) for model selection.
            for (a, b, g, t, s) in params:
                model = ExponentialSmoothing(train, trend=t, seasonal=s, **self.model_params)
                results = model.fit(smoothing_level=a, smoothing_slope=b, smoothing_seasonal=g)

                if results.aic < bestAIC:
                    bestAIC = results.aic
                    bestBIC = results.bic
                    bestalpha = a
                    bestbeta = b
                    bestgamma = g
                    besttrend = t
                    bestseasonal = s

            print('Finished parameters optimization...\n')
            print('The best alpha, beta and gamma are: {}, {}, {}'.format(bestalpha, bestbeta, bestgamma))
            print('The best trend and seasonal components are: {}, {}'.format(besttrend, bestseasonal))
            print('The best AIC score is:', bestAIC)

            model = ExponentialSmoothing(train, trend=besttrend, seasonal=bestseasonal, **self.model_params)
            results = model.fit(smoothing_level=bestalpha, smoothing_slope=bestbeta, smoothing_seasonal=bestgamma)

        else:
            print('No Grid search optimization for the HWES model...\n')
            model = ExponentialSmoothing(train, **self.model_params)
            results = model.fit()
            print('Finished...')

        self.results = results
        self.residuals = results.resid

        """-------------------------- EVALUATION --------------------------"""
        y_pred = results.predict(start=test.index[0], end=test.index[-1])
        y_true = series[test.index[0]:]
        mse = mean_squared_error(y_true, y_pred)
        output = pd.Series()
        output['AIC'] = results.aic
        output['BIC'] = results.bic
        output['mse'] = mse

        return output

    def _prophet(self, series, prop, gridsearch, cap=None):
        """
            The Prophet model consists of three components:
            - Trend, models non-periodic changes;
            - Seasonality, represents periodic changes;
            - Holidays component, contributes information about holidays and events.

            In addition to these one, there is the error term, representing information that was not reflected
            in the model. Usually it's modeled as normally distributed noise.

            Args:
            -----
                series (pd.Series)              >>> Series with index datetime64[ns] and values the observed
                                                    time-series process.

                prop (float)                    >>> Optional. Training set proportion. By default is set to 0.80.

                cap (float, int or pd.Series)   >>> Optional. To make forecasts using a logistic growth trend model,
                                                    it's necessary to specify the carrying capacity.
                                                    This parameteris not necessarily constant and may vary over
                                                    time. You can assume either a particular value or set it using
                                                    data (time series percentile, for example). Default is None.

            Returns:
            --------
                The scores computed on the test set.
        """

        # Dataframe creation containing the history. Must have columns 'ds' (datetype) and 'y', the time series.
        df = pd.DataFrame({'ds': series.index, 'y': series.values})
        freq = series.index.inferred_freq
        # print(freq)
        self.freq = freq

        if ('growth', 'logistic') in self.model_params.items():
            df['cap'] = cap

        # Train-test split
        train_size = int(len(series) * prop)
        train = df[:train_size]
        test = df[train_size:]

        y_true = series[test['ds'].values[0]:]

        """-------------------------- FITTING --------------------------"""
        if gridsearch:
            print('Grid search for the Prophet model...')

            seas_mode = ['additive', 'multiplicative']
            seas_prior_scale = np.arange(10, 110, 10)
            params = list(product(seas_mode, seas_prior_scale))

            bestmse = np.inf
            best_sm = None
            best_sps = None

            for (s_m, s_p_s) in params:
                model = Prophet(seasonality_mode=s_m, seasonality_prior_scale=s_p_s, **self.model_params)

                model.fit(train)

                # In this case we use the same freq of the originally df
                future = model.make_future_dataframe(periods=len(test), freq=freq)

                if ('growth', 'logistic') in self.model_params.items():
                    future['cap'] = cap

                forecast = model.predict(future)
                forecast_test = forecast[forecast['ds'] >= test['ds'].values[0]]
                y_forecast = pd.Series(forecast_test['yhat'].values,
                                       index=pd.to_datetime(forecast_test['ds'].values))
                mse = mean_squared_error(y_true, y_forecast)

                if mse < bestmse:
                    bestmse = mse
                    best_sm = s_m
                    best_sps = s_p_s

            print('Finished parameters optimization...\n')
            print('The best seasonality_mode and seasonality_prior_scale are: {}, {}'.format(best_sm, best_sps))
            print('The best MSE score is:', bestmse)

            model = Prophet(seasonality_mode=best_sm, seasonality_prior_scale=best_sps, **self.model_params)
            results = model.fit(train)

        else:
            print('No Grid search optimization for the Prophet model...\n')
            model = Prophet(**self.model_params)
            results = model.fit(train)
            print('Finished...')

        self.results = results

        """-------------------------- EVALUATION --------------------------"""
        future = results.make_future_dataframe(periods=len(test), freq=freq)
        if ('growth', 'logistic') in self.model_params.items():
            future['cap'] = cap

        forecast = results.predict(future)
        forecast_test = forecast[forecast['ds'] >= test['ds'].values[0]]
        y_forecast = pd.Series(forecast_test['yhat'].values,
                               index=pd.to_datetime(forecast_test['ds'].values))
        mse = mean_squared_error(y_true, y_forecast)

        return mse

    def predict(self, start_date=None, end_date=None, pred_params={}):
        """
            This function calculates predictions (in-sample and out-sample) for the fitted model.

            Args:
            -----
                start_date (nt, str, or datetime)   >>> Optional. Zero-indexed observation number at which to
                                                        start forecasting, ie., the first forecast is start.
                                                        Can also be a date string to parse or a datetime type.
                                                        Default is the the zeroth observation.

                end_date (nt, str, or datetime)     >>> Optional. Zero-indexed observation number at which to end
                                                        forecasting, ie., the first forecast is start. Can also be
                                                        a date string to parse or a datetime type. However, if the
                                                        dates index does not have a fixed frequency, end must be an
                                                        integer index if you want out of sample prediction.
                                                        Default is the last observation in the sample.

                pred_params (dict)                  >>> Optional. Dictionary that contain the additional parameters
                                                        for predict.

                1) SARIMAX
                    exog (array_like)                       >>> Optional. If the model includes exogenous
                                                                regressors, you must provide exactly enough
                                                                out-of-sample values for the exogenous variables
                                                                if end is beyond the last observation in the sample.
                                                                Default is None.

                    dynamic (bool, int, str, or datetime)   >>> Optional. Integer offset relative to 'start' at
                                                                which to begin dynamic prediction. Can also be an
                                                                absolute date string to parse or a datetime type
                                                                (these are not interpreted as offsets).
                                                                Prior to this observation, true endogenous values
                                                                will be used for prediction; starting with this
                                                                observation and continuing through the end of
                                                                prediction, forecasted endogenous values will be
                                                                used instead. Default is False.

                    full_results (bool)                     >>> Optional. If True, returns a FilterResults instance;
                                                                if False returns a tuple with forecasts, the
                                                                forecast errors, and the forecast error covariance
                                                                matrices. Default is False.

                3) Prophet
                    cap (float, int or pd.Series)   >>> Optional. To make forecasts using a logistic growth trend
                                                        model, it's necessary to specify the carrying capacity.
                                                        This parameter is not necessarily constant and may vary
                                                        over time. You can assume either a particular value or set
                                                        it using data (time series percentile, for example).
                                                        Default is None.

                    freq (str)                      >>> Optional. Any valid frequency for pd.date_range, such as
                                                        'D' or 'M'. Default is 'D'.

                    include_history (bool)          >>> Optional. To include the historical dates in the data frame
                                                        for predictions. Default is True.

            Returns:
            --------
                1) SARIMAX: the object 'PredictionResultsWrapper';
                2) HEWS: a pandas Series of the predicted values with index datetime64[ns];
                3) Prophet: a pandas Series of the predicted values with index datetime64[ns].

        """
        if self.model == 'SARIMAX':
            return self.results.get_prediction(start=start_date, end=end_date, **pred_params)

        elif self.model == 'HWES':
            return self.results.predict(start=start_date, end=end_date, **pred_params)

        elif self.model == 'Prophet':
            t = pd.date_range(start=start_date, end=end_date, freq=self.freq)
            future = pd.DataFrame({'ds': t})
            forecast = self.results.predict(future)
            return pd.Series(forecast['yhat'].values,
                             index=pd.to_datetime(forecast['ds'].values))
        else:
            raise Exception("Invalid method")


"""-------------------------- PLOT FOR THE RESIDUALS EVALUATION --------------------------"""


def plot_diagram(series, figsize=(13, 13), dpi=200, bins=20, lags=10, alpha=.05):
    sns.set_style('whitegrid')

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

    axes[0, 0].plot(series, color='blue', lw=1)
    axes[0, 0].set_title('Time series', fontsize='large')

    sns.distplot(series, bins=bins, hist=True, kde=True, color='blue',
                 hist_kws={'color': 'blue', 'label': 'Hist'},
                 kde_kws={'linewidth': 1, 'label': 'KDE'}, ax=axes[0, 1])

    axes[0, 1].set_title('Series Histogram and Density', fontsize='large')

    plot_acf(series, ax=axes[1, 0], lags=lags, alpha=alpha,
             vlines_kwargs={'color': 'darkblue'})

    plot_pacf(series, ax=axes[1, 1], lags=lags, alpha=alpha,
              vlines_kwargs={'color': 'darkblue'})

    plt.show()


def plot_diagnostics(residuals, figsize=(13, 13), dpi=200, bins=20, lags=10, alpha=.05):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=figsize, dpi=dpi)

    axes[0, 0].plot(residuals, color='blue', lw=1)
    axes[0, 0].axhline(y=0, color='r', linestyle='-', lw=1)
    axes[0, 0].set_title('Standardized Residuals', fontsize='large')

    sns.distplot(residuals, bins=bins, hist=True, kde=True, color='blue',
                 hist_kws={'color': 'blue', 'label': 'Hist'},
                 kde_kws={'linewidth': 1, 'label': 'KDE'}, ax=axes[0, 1])

    value = np.random.normal(loc=0, scale=1, size=10000000)
    sns.distplot(value, hist=False, ax=axes[0, 1], color='red', label='N(0,1)')
    axes[0, 1].set_title('Residuals Histogram and Density', fontsize='large')

    stats.probplot(residuals, plot=axes[1, 0])

    plot_acf(residuals, ax=axes[1, 1], lags=lags, alpha=alpha,
             vlines_kwargs={'color': 'darkblue'})

    plt.show()
