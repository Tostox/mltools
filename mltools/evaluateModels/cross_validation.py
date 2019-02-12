import numpy as np
import pandas as pd
import scipy
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV, KFold
from sklearn.metrics import make_scorer
from scipy.stats import *
from hyperopt import hp
from .bayesian_optimization import BayesianCV
from ..plottingTool.mltools_plot import cross_validation_plot
from ._utils import *

# Classification metrics
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score

# Regression metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error
from sklearn.metrics import residual_mean_squared_error, mean_squared_log_error, r2_score


class CrossValidation():

    """ This class computes the k-fold cross validation for a list of models (and their specific parameters)
        in input.
        The models that we want to apply are written into a file txt. That file is read by the class into
        the init function. Also, in this configuration file the users can specify the list of parameters that
        they want to test.
        In this case the class compute a grid search cv to identify the best model by the optimization of a
        specific loss function. If no parameters are specified, or no params_file is pass to the class, all models
        will be instantiated with the default configuration.

        At the end of the fitting procedure, the class returns a list of models (the best results of grid search)
        and a pandas DataFrame which contains the mean and the standard deviation of all score metrics that the
        user wants to analyze. It's also possible to obtain the single folds results.
    Args:
    -----
        models                >>> The list of models that we want to evaluate. See the results of get_models_info
                                  for the list of models that can be used.

        scores                >>> The list of scores that we want to compute. See the results of get_scores_info
                                  for the list of scores that can be used.

        params_file           >>> Optional. File which contains the list of parameters to test for each model.
    """

    def __init__(self, models, scores, params_file=None):
        if type(models) == list:
            self.models = models
        else:
            self.models = [models]

        if type(models) == list:
            self.scores = scores
        else:
            self.scores = [scores]

        # dictionaries of models and scores used to import them
        self.model_dict = create_model_dict()
        self.scores_dict = create_score_dict()

        if params_file:
            with open(params_file) as param:
                exec(param.read())  # param_file contains a dict named "model_dict"
                self.params_file = model_dict
                # check if the models specified in input are defined
                for model in self.models:
                    if model not in self.params_file.keys():
                        raise Exception("The model {} is not defined into the param file".format(model))
                    elif model not in self.model_dict.keys():
                        raise Exception("The model {} is not avaible".format(model))
        else:
            print("No param_file in folder. Models will have the default configuration.")

        for score in self.scores:
            if score not in self.scores_dict.keys():
                raise Exception("The score {} it's not avaible".format(score))

    @property
    def get_models_info(self):
        ''' This function print the model's list '''
        for key in self.model_dict.keys():
            print(key)

    @property
    def get_scores_info(self):
        ''' This function print the score's list'''
        for key in self.scores_dict.keys():
            print(key)

    def create_model(self, model):
        """ This function create an instance of the specified model, by running che python code in the conf file.

            Args:
            -----
                model   >>> Name of the model that we want to create.

            Returns:
            --------
                an instance of the specified model.
        """

        exec(self.model_dict[model]['path'])
        return eval(self.model_dict[model]['init'])

    def set_model_params(self, model, param_dict):
        """ This function set the dict of the parameters for a specific ML models given in input.

            Args:
            -----
                model (instance of a ML model) >>> The instance of a model that we want to setting.
                param_dict                     >>> A dict that contains as a keys the name of the parameters
                                                    and the items as the values that we want to set.

            Returns:
            --------
                An instance of the specified model.
        """

        try:
            model.set_params(**param_dict)
        except Exception:
            Exception("Impossible to set the parameters in {} model.".format(model))
        return model

    def fit(self, train_data, target_variable, folds_results=False,
               k=10, seed=123, plot_iteration_results=False):
        """ This function fit the cross validation for the models read by the class during the initialization.

            Args:
            -----
                train_data               >>> An array that contains the features data.
                target_variable          >>> An array which contains the target variable to predict.
                folds_results            >>> Optional. Set to true if you want the results of your models
                                             for each fold. Default is set to False.
                k                        >>> Optional. Number of folds. Default is equal to 10.
                seed                     >>> Optional. Number of seed for k_fold. Default is equal to 123.
                plot_iteration_results   >>> Optional. Plot iteration results for bayesian optimization

            Returns:
            --------
                A tuple of a pandas Dataframe that contain the scoring results and a dict that has as a keys the
                name of models used and as values the instances built.
        """
        results_dict = {}
        models_dict = {}
        kf = KFold(k, shuffle=True, random_state=seed).get_n_splits(train_data)
        for model in self.models:
            print('Model: %s' % model)
            clf = self.create_model(model)
            if self.params_file:
                # extract the dictionary containing the params of cv methods
                model_params = self.params_file[model]

                if model_params["search_method"] == "cv":
                        model_to_fit = self.set_model_params(clf, model_params["params"])
                        print("Create a model: {}".format(model_to_fit))
                        print("Evaluate the model with cross validation...")
                else:
                    if model_params["search_method"] == "grid_search":
                        if plot_iteration_results:
                            print('Not possible to plot iteration results for %s' % model_params["search_method"])
                            plot_iteration_results = False

                        gs = GridSearchCV(clf, param_grid=model_params["params"], cv=kf,
                                          refit=True, **model_params['search_params'])
                        print("Searching the best {} with grid search cv...".format(model))

                    elif model_params["search_method"] == "random_search":
                        if plot_iteration_results:
                            print('Not possible to plot iteration results for %s' % model_params["search_method"])
                            plot_iteration_results = False

                        # distribution = self._createSpace(model_params["params"])
                        gs = RandomizedSearchCV(clf, param_distributions=model_params["params"], cv=kf,
                                                **model_params['search_params'])
                        print("Searching the best {} with random search cv...".format(model))

                    elif model_params["search_method"] == "bayesian_search":
                        gs = BayesianCV(model=clf, kf=kf, search_space=model_params["params"],
                                        task=model_params["task"],
                                        **model_params['search_params'])
                        print("Searching the best {} with bayesian optimization cv...".format(model))

                    else:
                        raise Exception("Error: invalid search_method. Use 'cv', 'grid_search',\
                                        'random_search' or 'bayesian_search'.")

                    results = gs.fit(train_data, target_variable)

                    model_to_fit = results.best_estimator_

                    if plot_iteration_results:  # only for BayesianCV
                        iteration = [x['iteration'] for x in results.results_]
                        loss = [x['loss'] for x in results.results_]
                        best_result = [results.best_score_] * len(iteration)

                        cross_validation_plot.plotting_iter_res(iteration, loss, best_result,
                                                                title='Sequence of Values for Bayesian Optimization')

                    print("\nBest_estimator: {}\n\nBest_scores: {}".format(model_to_fit, results.best_score_))
                    print("Evaluate the best model configuration with a new cross validation...\n")

            elif not self.params_file:  # if no parameters are specified we use the default configuration
                model_to_fit = clf
                print("Create a model: {}".format(model_to_fit))
                print("Evaluate the model with cross validation...")

            results = cross_validate(model_to_fit, train_data, target_variable,
                                     scoring=self._select_score(), cv=kf)

            results_dict[model] = results
            models_dict[model] = model_to_fit

        print("Finish")
        return self._create_DataFrame(results_dict, folds_results), models_dict

    def _select_score(self):
        """ This function compute the score by extract it from the configuration file.

            Returns:
            --------
                an instance of the sklearn metric.
        """

        custom_score = {k: eval(self.scores_dict[k]) for k in self.scores}
        return custom_score

    def _mean_confidence_interval(self, data, alpha, metric):
        ''' This function compute a confidence interval.

            Args:
            -----
                data: list      >>> A list of k-fold results
                alpha: float    >>> Coefficient for confidence level (1-alpha)

            Return:
            -------
                ci: list        >>> A list of 2 values, lowerbound and upperbound.

        '''
        n = len(data)
        m, std_error = np.mean(data), scipy.stats.sem(data)
        t_quantile = t.ppf(1 - alpha / 2, n - 1)
        radius = t_quantile * std_error / np.sqrt(n)
        lower_bound = m - radius
        upper_bound = m + radius

        ci = [np.round(lower_bound, 4), np.round(upper_bound, 4)]

        if metric in ['accuracy', 'auc', 'f1', 'f1_multiclass', 'precision',
                      'precision_multiclass', 'recall', 'recall_multiclass']:
            ci = list(np.clip(ci, a_min=0, a_max=1))

        elif metric in ['mcc', 'r2']:
            ci = list(np.clip(ci, a_min=-1, a_max=1))

        return ci

    def _create_DataFrame(self, results, folds_results):
        """ This function creates a pandas DataFrame that contains the scoring results for each model.
            In particular, for each score are reported the mean and the standard deviation. Also, if folds_results
            was setted to True, dataframe contains also a column with the list of the results for each single fold.

            Args:
            -----
                results (dict)          >>> dictionary that contains the results for each model tested.
                folds_results (boolean) >>> if it's True, are reported also the single fold results.

            Returns:
            --------
                pandas.DataFrame.
        """

        df_results = pd.DataFrame.from_dict(results, orient='index')

        output = pd.DataFrame()
        output['computation_total'] = (df_results['fit_time'] + df_results['score_time']).apply(np.sum)

        for col in df_results.columns[2:]:
            metric = col.split('_')[1]
            col_mean = col + '_mean'
            output[col_mean] = df_results[col].apply(np.mean)
            col_sd = col + '_sd'
            output[col_sd] = df_results[col].apply(np.std)

            col_ci1 = col + '_ci_95%'
            output[col_ci1] = df_results[col].apply(self._mean_confidence_interval, args=(0.05, metric,))
            col_ci2 = col + '_ci_99%'
            output[col_ci2] = df_results[col].apply(self._mean_confidence_interval, args=(0.01, metric,))

            if folds_results:
                col_partial = col + '_partial'
                output[col_partial] = df_results[col]

        return output.T
