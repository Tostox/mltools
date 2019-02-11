import pandas as pd
import numpy as np
from ..plottingTool.mltools_plot import time_series_plot


class ETS_decomposition():

    methods_avaible = ['ssa', 'hp_filter', 'seasonal_decomposition']

    def __init__(self, method="ssa", L=5, save_mem=True):

        if method not in self.methods_avaible:
            raise Exception('Invalid method')

        self.method = method

        if method == "ssa":
            self.L = L
            self.save_mem = save_mem

    class SSA(object):

        __supported_types = (pd.Series, np.ndarray, list)

        def __init__(self, tseries, L, save_mem=True):
            """
            Decomposes the given time series with a singular-spectrum analysis.
            Assumes the values of the time series are recorded at equal intervals.

            Args:
            -----
                tseries >>> The original time series, in the form of a Pandas Series, NumPy array or list.
                L >>> The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
                save_mem >>> Conserve memory by not retaining the elementary matrices.
                             Recommended for long time series with thousands of values. Defaults to True.

            Note:
            -----
                Even if an NumPy array or list is used for the initial time series, all time series returned will be
                in the form of a Pandas Series or DataFrame object.
            """

            # Tedious type-checking for the initial time series
            if not isinstance(tseries, self.__supported_types):
                raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

            # Checks to save us from ourselves
            self.N = len(tseries)
            if not 2 <= L <= self.N / 2:
                raise ValueError("The window length must be in the interval [2, N/2].")

            self.L = L
            self.orig_TS = pd.Series(tseries)
            self.K = self.N - self.L + 1

            # Embed the time series in a trajectory matrix
            self.X = np.array([self.orig_TS.values[i:L + i] for i in range(0, self.K)]).T

            # Decompose the trajectory matrix
            self.U, self.Sigma, VT = np.linalg.svd(self.X)
            self.d = np.linalg.matrix_rank(self.X)

            self.TS_comps = np.zeros((self.N, self.d))

            if not save_mem:
                # Construct and save all the elementary matrices
                self.X_elem = np.array([self.Sigma[i] * np.outer(self.U[:, i], VT[i, :]) for i in range(self.d)])

                # Diagonally average the elementary matrices, store them as columns in array.
                for i in range(self.d):
                    X_rev = self.X_elem[i, ::-1]
                    self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

                self.V = VT.T
            else:
                # Reconstruct the elementary matrices without storing them
                for i in range(self.d):
                    X_elem = self.Sigma[i] * np.outer(self.U[:, i], VT[i, :])
                    X_rev = X_elem[::-1]
                    self.TS_comps[:, i] = [X_rev.diagonal(j).mean() for j in range(-X_rev.shape[0] + 1, X_rev.shape[1])]

                self.X_elem = "Re-run with save_mem=False to retain the elementary matrices."

                # The V array may also be very large under these circumstances, so we won't keep it.
                self.V = "Re-run with save_mem=False to retain the V matrix."

            # Calculate the w-correlation matrix.
            self.calc_wcorr()

        def components_to_df(self, n=0):
            """
            Returns all the time series components in a single Pandas DataFrame object.
            """
            if n > 0:
                n = min(n, self.d)
            else:
                n = self.d

            # Create list of columns - call them F0, F1, F2, ...
            cols = ["F{}".format(i) for i in range(n)]
            return pd.DataFrame(self.TS_comps[:, :n], columns=cols, index=self.orig_TS.index)

        def reconstruct(self, indices):
            """
            Reconstructs the time series from its elementary components, using the given indices.
            Returns a Pandas Series object with the reconstructed time series.

            Parameters
            ----------
            indices: An integer, list of integers or slice(n,m) object, representing the elementary components to sum.
            """
            if isinstance(indices, int):
                indices = [indices]

            ts_vals = self.TS_comps[:, indices].sum(axis=1)
            return pd.Series(ts_vals, index=self.orig_TS.index)

        def __get_elbow_point_index__(self,curve):
            import numpy.matlib
            nPoints = len(curve)
            allCoord = np.vstack((range(nPoints), curve)).T
            #np.array([range(nPoints), curve])
            firstPoint = allCoord[0]
            lineVec = allCoord[-1] - allCoord[0]
            lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
            vecFromFirst = allCoord - firstPoint
            scalarProduct = np.sum(vecFromFirst * np.matlib.repmat(lineVecNorm, nPoints, 1), axis=1)
            vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
            vecToLine = vecFromFirst - vecFromFirstParallel
            distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
            idxOfBestPoint = np.argmax(distToLine)
            return idxOfBestPoint


        def get_main_components(self,corr_threshold=.45,adjust=0):
            '''
            Returns the most relevant aggregated components, that is the components that explain most of the 'mass' of the
            Sigma matrix, obtained by reconstructing highly correlated Xi's though a clustering method.
            '''

            # select number of components using elbow method
            n_relevant =  self.__get_elbow_point_index__(np.cumsum(self.Sigma**2)/(self.Sigma**2).sum())

            # look for correlated components and reconstruct
            n_groups = n_relevant + adjust
            self.groups = [[i] for i in range(n_groups)]
            dist_mx = np.copy(self.Wcorr[0:n_groups,0:n_groups])
            dist_mx[np.tril_indices(n_groups)] = 0 # set to 0 the lower part
            flag = 1
            while flag:
                max_corr = dist_mx.max()
                if max_corr < corr_threshold or dist_mx.shape == (1,):
                    flag = 0
                else:
                    r,c = np.unravel_index(dist_mx.argmax(), dist_mx.shape)
                    self.groups[r] = self.groups[r]+self.groups[c]
                    del self.groups[c]
                    dist_r = list(dist_mx[0:(r+1),r])+list(dist_mx[r,(r+1):])
                    dist_c = list(dist_mx[0:(c+1),c])+list(dist_mx[c,(c+1):])

                    for i in range(n_groups):
                        # update distace matrix
                        if r > i: dist_mx[i,r] = max((dist_r[i],dist_c[i]))
                        if r < i: dist_mx[r,i] = max((dist_r[i],dist_c[i]))
                    dist_mx = np.delete(dist_mx,c,0)
                    dist_mx = np.delete(dist_mx,c,1)
                    n_groups -= 1

            for group in self.groups: group.sort()

            main_components = self.reconstruct(self.groups[0])
            if len(self.groups) > 1:
                for group in self.groups[1:]:
                    main_components=pd.concat([main_components,self.reconstruct(group)],axis=1)

            main_components = pd.concat([main_components,self.reconstruct(range(n_relevant,self.L))],axis=1)
            main_components.columns = [r"$\tilde{{F}}^{{({0})}}$".format(i) for i in range(len(self.groups)+1)]

            return main_components

        def calc_wcorr(self):
            """
            Calculates the w-correlation matrix for the time series.
            """

            # Calculate the weights
            w = np.array(list(np.arange(self.L) + 1) + [self.L] * (self.K - self.L - 1) + list(np.arange(self.L) + 1)[::-1])

            def w_inner(F_i, F_j):
                return w.dot(F_i * F_j)

            # Calculated weighted norms, ||F_i||_w, then invert.
            F_wnorms = np.array([w_inner(self.TS_comps[:, i], self.TS_comps[:, i]) for i in range(self.d)])
            F_wnorms = F_wnorms**-0.5

            # Calculate Wcorr.
            self.Wcorr = np.identity(self.d)
            for i in range(self.d):
                for j in range(i + 1, self.d):
                    self.Wcorr[i, j] = abs(w_inner(self.TS_comps[:, i], self.TS_comps[:, j]) * F_wnorms[i] * F_wnorms[j])
                    self.Wcorr[j, i] = self.Wcorr[i, j]

        def plot_wcorr(self):
            """
            Plots the w-correlation matrix for the decomposed time series.
            """
            time_series_plot._plot_wcorr(self.Wcorr, self.L)

################################################################################

    def hp_filter(self, data):

        import statsmodels.api as sm

        return sm.tsa.filters.hpfilter(data)

    def seasonal_decomposition(self, data):
        from statsmodels.tsa.seasonal import seasonal_decompose
        decomposition = seasonal_decompose(data)

        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        return trend, seasonal, residual

    def fit(self, data):
        """ Methods used to apply the functions for the ets decomposition to a pandas time series.

        Args:
        -----
            data >>> pandas Series. index datetime64[ns], values the data observed.

        Returns:
        --------
            object with the results
        """

        if self.method == "ssa":
            return self.SSA(data, self.L, self.save_mem)
        if self.method == "hp_filter":
            return self.hp_filter(data)
        if self.method == "seasonal_decomposition":
            return self.seasonal_decomposition(data)
        else:
            raise Exception("Invalid method")
