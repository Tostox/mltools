from rpy2.robjects import *
from rpy2.robjects import pandas2ri
import rpy2.robjects.numpy2ri
import json
import pandas as pd
import time
from ..plottingTool.mltools_plot import time_series_plot
from bokeh.models.widgets import Panel, Tabs
from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models.annotations import Span
from bokeh import palettes


pandas2ri.activate()

rpy2.robjects.numpy2ri.activate()


class CPA():

    """ The goal of this class is the change point detection (CPA - Change Points Analysis),
    that means the identification of abrupt variation in the process behavior due to distributional or structural
    changes. A 'changepoint' can be defined as unexpected, structural, changes in time series data properties
    such as the mean or variance.

    We examine eight different change point detection methods, implemented using the R code.
    Some of these methods are unavailable to detect variance alterations (prophet, wbs, ecp, bcp).

    - The 'changepoint' package provides four techniques to achieve the goal: the changes are found using the
    method supplied which can be single changepoint (AMOC) or multiple changepoints using exact (PELT or SegNeigh)
    or approximate (BinSeg) methods.
    More info: https://cran.r-project.org/web/packages/changepoint/changepoint.pdf

    - The 'wbs' package implements 'Wild Binary Segmentation', a technique for consistent estimation of the number
    and locations of multiple change-points in data. It also provides a fast implementation of the standard Binary
    Segmentation algorithm, but we didn't use it. WBS overcomes the problems of the binary segmentation technique
    and is an improvement in terms of computation. It uses the idea of computing Cumulative Sum (CUSUM) from randomly
    drawn intervals considering the largest CUSUMs to be the first change point candidate to test against the
    stopping criteria. This process is repeated for all the samples.
    More info: https://cran.r-project.org/web/packages/wbs/wbs.pdf

    - In the 'ecp' package, two different methods, E-Divisive and E-Agglomerative algorithms, are provided for
    univariate as well as multivariate data. Divisive methods estimate change points using a bisection algorithm,
    whereas agglomerative methods detect abrupt change using agglomeration. In this case, we implemented only
    the first method.
    More info: https://cran.r-project.org/web/packages/ecp/ecp.pdf

    - The 'bcp' package implements the Bayesian change point analysis.
    More info: https://cran.r-project.org/web/packages/bcp/bcp.pdf

    - The 'prophet' package is also available in Python. It implements an automatic procedure for forecasting
    time series data based on an additive model where non-linear trends are fit with yearly, weekly, and daily
    seasonality, plus holiday effects. Another task solved by prophet is trend changepoints detection.
    We can set some parameters in the model instance to perform detection in the best way.
    More info: https://cran.r-project.org/web/packages/prophet/prophet.pdf

    Args:
    -----
        methods (list or String) >>> Optional. A list or a string of all methods that you want to apply.
                                     See the results of 'getInfo_models' for the list of models that can be used.
                                     By default we use Prophet.

        compute_mean (boolean)   >>> Optional. Set to True if you want to compute changepoints for mean.
                                     Default is set to True.

    """

    # Available methods in the class.
    methods_av = ['AMOC', 'BinSeg', 'SegNeigh', 'PELT', 'PROPHET', 'wbs', 'ecp', 'bcp']

    def __init__(self, method='PROPHET'):

        self.method = method

        if method not in CPA.methods_av:
            raise Exception("Method '{}' is not available. Choose between {}".format(method, methods_av))

    def get_libraries():
        """ This function checks if all the necessary packages are installed in R and, in this case, loads them.
            If there are missing packages, it asks to the user to confirm (or not) the installation;
            giving a positive answer we continue with the installation.
        """

        list_packages = ['bcp', 'wbs', 'prophet', 'ecp', 'xts', 'changepoint', 'RJSONIO', 'hash']
        uninstalled_packages = []

        for name in list_packages:
            try:
                r('library({})'.format(name))
                print("The package '{}' is already installed.".format(name))
            except:
                uninstalled_packages.append(name)

        print("\nMissing Packages: ", uninstalled_packages)

        if uninstalled_packages:
            decision = input("Do you want to continue with the installation of the missing packages? y/n: \n")

            if decision == 'y':
                for package in uninstalled_packages:
                    print(package)
                    try:
                        r("""install.packages("{}")
                            library({})
                            """.format(package, package))
                        print("The package '{}' has been successfully installed!".format(package))
                    except:
                        print("The package '{}' doesn't exist...".format(package))
            else:
                print("The installation has been interrupted...")

        else:
            print("No packages nedeed")

    def fit(self, data, model_param={}, compute_mean=True, format_dt="%Y-%m-%d"):
        """
            This function fits the change point detection methods. If 'compute_mean' is set to True, we search
            changes in time series mean, otherwise we try to find out variance anomalous behavior.

            Args:
            -----
                data >>> pd.Series.index datetime64[ns], values the data observed.
                To get informations about the other parameters, look at the respective functions '_mean' and '_var'.

            Returns:
            --------
                A dict with the methods as keys and the changepoints location as values.
        """
        values = data  # <-- E' inutile
        date = data.index

        if compute_mean:
            d = self._mean(values, date, format_dt, model_param)
        else:
            d = self._var(values, date, format_dt, model_param)

        return d

    def _init_wbs(model_param):
        # wbs parameters
        r.assign('M', model_param.get('n_intervals', 5000))
        r.assign('penalty_wbs', model_param.get('penalty_wbs', 'MBIC').lower() + '.penalty')
        r.assign('th.const', model_param.get('th_wbs', 1.3))
        r.assign('n.checkpoints', model_param.get('n_checkpoints', 5))

    def _init_mean_changepoint(model_param):
        # changepoint parameters for mean
        r.assign('penalty', model_param.get('penalty_bs', 'MBIC'))
        r.assign('n.checkpoints', model_param.get('n_checkpoints', 5))
        r.assign('test.stat', model_param.get('test_stat', 'Normal'))
        r.assign('minseqlen', model_param.get('minseqlen', 1))

    def _init_var_changepoint(model_param):
        # changepoint parameters for variance
        r.assign('penalty', model_param.get('penalty', 'AIC'))
        r.assign('n.checkpoints', model_param.get('n_checkpoints', 5))
        r.assign('test.stat', model_param.get('test_stat', 'Normal'))
        r.assign('minseqlen', model_param.get('minseqlen', 1))

    def _init_ecp(model_param):
        # ecp parameters
        r.assign('sig.lvl', model_param.get('sig_lvl', 0.05))
        r.assign('R', model_param.get('R', 199))
        r.assign('alpha', model_param.get('alpha', 1))
        r.assign('min.size', model_param.get('min_size', 30))

    def _init_bcp(model_param):
        # bcp parameters
        r.assign('p0', model_param.get('p0', 0.2))
        r.assign('threshold', model_param.get('threshold', 0.99))

    def _init_prophet(model_param):
        # prophet parameters
        r.assign('growth', model_param.get('growth', 'linear'))
        r.assign('changepoint.prior.scale', model_param.get('changepoint_priorscale', 0.05))

    def _mean(self, values, date, format_dt, model_param):
        """
            This function detects changepoints based on unexpected mean changes of the time series.

        Args:
        -----
            - values (pd.Series or Numpy array) >>> Array that contains the numeric values of the time serie.
            - date (Numpy array of datetime)    >>> Array that contains the datetime index.
            - format_dt (str)                   >>> Optional. Date's format of the "field_dt" column.

            (1) For the methods "BinSeg", "PELT", "AMONG" and "SegNeigh":

            - n_checkpoints (int)       >>> Optional. Maximum number of changepoints to search for using the
                                            "BinSeg" method. The maximum number of segments (number of
                                            changepoints + 1) to search for using the "SegNeigh" method
                                            (no for AMONG and PELT).

            - penalty (string)          >>> Optional. Choice of "None", "SIC", "BIC", "MBIC", AIC", "Hannan-Quinn",
                                            "Asymptotic", "Manual" and "CROPS" penalties.

            - test_stat (string)        >>> Optional. The assumed test statistic / distribution of the data.
                                            Currently only "Normal" and "CUSUM" supported.

            - minseqlen (int)           >>> Optional. Positive integer giving the minimum segment length
                                            (no. of observations between changes), default is the minimum
                                            allowed by theory.

            (2) For the method "wbs":

            - n_intervals (int)         >>> Optional. Positive integer giving the numbers of random intervals
                                            in each step where will look for checkpoints.

            - n_checkpoints (int)       >>> Optional. Maximum number of change-points to be detected.

            - penalty_wbs (string)      >>> Optional. Name of penalty functions used. Choise of "BIC", "MBIC"
                                            and "SSIC".

            - th_wbs (float)            >>> Optional. Positive scalar for the threshold.

            (3) For the method "ecp":

            - n_checkpoints (int)       >>> Optional. Maximum number of change-points to be detected.

            - sig_lvl (float)           >>> Optional. The level at which to sequentially test if a proposed change
                                            point is statistically significant.

            - min.size (float)          >>> Optional. Minimum number of observations between change points.

            - R (int)                   >>> Optional. The maximum number of random permutations to use in each
                                            iteration of the permutation test. The permutation test p-value is
                                            calculated using the method outlined in Gandy (2009).

            - alpha (int)               >>> Optional. The moment index used for determining the distance between
                                            and within segments.

            (4) For the method "PROPHET":

            - n_checkpoints (int)               >>> Optional. Number of potential changepoints to include.

            - growth (string)                   >>> Optional. String 'linear' or 'logistic' to specify a linear
                                                    or logistic trend.

            - changepoint_priorscale (float)    >>> Optional. Parameter modulating the flexibility of the automatic
                                                    changepoint selection. Large values will allow many
                                                    changepoints, small values will allow few changepoints.

            (5) For the method "bcp":

            - p0 (float)                >>> Optional. The prior on change point probabilities, U(0, p0), on the
                                            probability of a change point at each location in the sequence;
                                            for data on a graph, it is the parameter in the partition prior,
                                            p0^{l(ρ)}, where l(ρ) is the boundary length of the partition.

            - threshold (float)         >>> Optional. Take the changepoints with a probability > of the threshold.

        Returns:
        --------
            A dict with the methods as keys and the changepoints location as values.

        """

        r.assign('format_dt', format_dt)
        r.assign('method', self.method)
        df_new = pd.DataFrame({'ds': date, 'y': values})
        r.assign('df_new', df_new)

        if self.method == 'wbs':
            CPA._init_wbs(model_param)
        elif self.method in ['PELT', 'AMOC', 'BinSeg', 'SegNeigh']:
            CPA._init_mean_changepoint(model_param)
        elif self.method == 'ecp':
            CPA._init_ecp(model_param)
        elif self.method == 'bcp':
            CPA._init_bcp(model_param)
        elif self.method == 'PROPHET':
            CPA._init_prophet(model_param)

        # create a file json with R
        file_json = r("""

        library(RJSONIO)           # necessary to create json file

        library(hash)

        h = hash()                 # hash as dictionary, where for each method (key) there will be the respective value

        library(xts)               # we need to convert df to xts and after to ts


        #---------------changepoint library........................................
           
          changepoint.models = c('PELT','AMOC','BinSeg','SegNeigh')
          
          if(method %in% changepoint.models){
             library(changepoint)

             value_xts = xts(df_new$y, order.by=as.Date(df_new$ds, format_dt))
             value_ts = as.ts(value_xts)
             mvalue = cpt.mean(value_ts, method=method, penalty=penalty, Q=n.checkpoints, test.stat=test.stat, minseglen=minseqlen)
             loc_cpt = cpts(mvalue)

             h[[method]] = loc_cpt

          } else if (method == 'PROPHET'){
            library("prophet")

            m = prophet(df_new, n.changepoints = n.checkpoints, changepoint.prior.scale = changepoint.prior.scale, growth = growth)

            cpt.loc = as.Date(m$changepoints, format_dt)      #checkpoint's location have to be index not date,so it will be converted
            date = as.Date(df_new$ds)                             
            loc_cpt = rep_len(1, length(cpt.loc))           
            i=1
            for(loc in cpt.loc){
              loc_cpt[i] = which(date==loc)
              i=i+1
            }
                      
            h[[method]] = loc_cpt

          } else if (method == 'wbs'){
            library("wbs")

            w = wbs(df_new$y, M = M)
            w.cpt = changepoints(w, Kmax = n.checkpoints, penalty = penalty_wbs, th.const = th.const)
            loc_w = w.cpt$cpt.ic[[penalty_wbs]]
            loc_cpt = sort(loc_w)
          
            h[[method]] = loc_cpt

          } else if (method == 'ecp'){
            library("ecp")

            e = e.divisive(df_new, k = n.checkpoints, sig.lvl = sig.lvl ,R = R, alpha = alpha, min.size = min.size)
            loc_cpt = e$estimates[2:(length(e$estimates)-1)]
          
            h[[method]] = loc_cpt

          } else if (method=='bcp'){

            library("bcp")

            b = bcp(df_new$y, p0 = p0)
            cond = b[["posterior.prob"]] > threshold             #take just the checkpoints with a prob > thereshold
            loc_cpt = which(cond)

            h[[method]] = loc_cpt

          }

        exportJson = toJSON(h)                                   # json creation

        """
        )

        # R'json on python is a array, so it will be convert firstly to string and after to dictionary
#        s = file_json[0].replace('\n', '').replace('   ', '')
#        json_acceptable_string = s.replace("'", "\"")
#        d = json.loads(json_acceptable_string)
        d = json.loads(file_json[0])

        return d

    def _var(self, values, date, format_dt, model_param):
        """This function detects changepoints based on unexpected variance changes of the time series.

        Args:
        -----

            - df (pd.DataFrame)     >>> Dataframe with datetime indexes and at least a numeric column (variable).

            - field (str)           >>> Optional. Name of the field (column) that contain numeric values.

            - field_dt (str)        >>> Optional. Name of the field (column) that contain dates.

            - format_dt (str)       >>> Optional. Date's format of the "field_dt" column.

            - n_checkpoints (int)   >>> Optional. Maximum number of changepoints to search for using the "BinSeg"
                                        method. The maximum number of segments (number of changepoints + 1) to
                                        search for using the "SegNeigh" method (no for AMOC and PELT).

            - penalty (string)      >>> Optional. Choice of "None", "SIC", "BIC", "MBIC", AIC", "Hannan-Quinn",
                                        "Asymptotic", "Manual" and "CROPS" penalties.

            - test_stat (string)    >>> Optional. The assumed test statistic / distribution of the data.
                                        Currently only "Normal" and "CUSUM" supported.

            - minseqlen (int)       >>> Optional. Positive integer giving the minimum segment length
                                        (no. of observations between changes), default is the minimum allowed
                                        by theory.

        Returns:
        --------
            A dict with the methods as keys and the changepoints location as values.
        """

        # check if method are avalaible for the variance checkpoints
        if self.method not in ['PELT', 'AMOC', 'BinSeg', 'SegNeigh']:
            raise Exception("Can't use %s to detect changepoints in variance." % self.method)

        r.assign('format_dt', format_dt)
        r.assign('method', self.method)
        df_new = pd.DataFrame({'ds': date, 'y': values})
        r.assign('df_new', df_new)
        CPA._init_var_changepoint(model_param)

        # create a file json with R
        file_json = r("""

        library(RJSONIO)

        library(hash)

        h = hash()

        library(xts)

        #---------------changepoint library.................................................................

          changepoint.models=c('PELT','AMOC','BinSeg','SegNeigh')

          if(method %in% changepoint.models){
          library(changepoint)

          value_xts = xts(df_new$y, order.by = as.Date(df_new$ds, format_dt))
          value_ts = as.ts(value_xts)
          mvalue = cpt.var(value_ts, method = method, penalty = penalty, Q = n.checkpoints-1, test.stat = test.stat, minseglen = minseqlen)
          loc_cpt=cpts(mvalue)

          h[[method]] = loc_cpt

            }

         exportJson = toJSON(h)
        """)

        # R'json on python is an array, so it will be convert firstly to string and after to dictionary
        s = file_json[0].replace('\n', '').replace('   ', '')
        json_acceptable_string = s.replace("'", "\"")
        d = json.loads(json_acceptable_string)

        return d

    @staticmethod
    def plot_changepoints(data, cp_list, colors=None):
        """
            This function plots vertical lines in correspondance with the changepoints location founded by models.
            The time series plot is given by the function plot_time_series(), previously implemented in the
            'plottingTools' class. The function returns a different plot for each model used.
        """
        results = {key: d[key] for d in cp_list for key in d}
        tab_list = []

        if not colors:
            colors = palettes.Set1[len(results.keys())]

        if isinstance(colors, str):
            colors = [colors] * len(results.keys())

        for key, color in zip(results.keys(), colors):
            f = figure(x_axis_type="datetime")
            f = time_series_plot.plot_time_series(data=data, fig=f, alpha=0.6)

            if type(results[key]) == int:
                    results[key] = [results[key]]

            for value in results[key]:
                start_date = time.mktime(data.index[value].timetuple()) * 1000
                span_4 = Span(location=start_date, dimension='height', line_color=color,
                              line_dash='dashed', line_width=1)
                f.add_layout(span_4)

            tab = Panel(child=f, title=key)
            tab_list.append(tab)

        tabs = Tabs(tabs=tab_list)
        show(tabs)
