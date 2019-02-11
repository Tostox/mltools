from hyperopt import fmin, tpe, hp, Trials, rand
from hyperopt import STATUS_OK
from sklearn.model_selection import cross_val_score
import numpy as np
from copy import deepcopy
from ..plottingTool.mltools_plot import cross_validation_plot


class BayesianCV():
    '''
    Bayesian optimization is a probabilistic model based approach for finding the minimum of
    any function that returns a real-value metric.

    Args:
    -----
    model:          >>> List of model to be tested
    kf:             >>> Number of fold
    search_space:   >>> File *.txt that contains params' distributions
    task:           >>> Flag for 'classification' or 'regression' problem
    n_iter:         >>> Optional. Number of iterations of optimization. Default 100.
    plot_loss:      >>> Optional. If True plot the comparison between Tpe and Random algorithm. Default False.


    Return:
    self

    '''

    def __init__(self, model, kf, search_space, scoring, task,
                 n_iter=100, n_jobs=1, plot_loss=False):

        self.model = model
        self.kf = kf
        self.search_space = search_space
        self.scoring = scoring

        if task == 'regression' or task == 'classification':
            self.task = task
        else:
            raise Exception("Error: invalid task. Use 'classification' or 'regression'.")

        self.n_iter = n_iter
        self.train_set = None
        self.target_variable = None
        self.n_jobs = n_jobs
        self.plot_loss = plot_loss

    def _model_eval(self, hyperparameters, model):
        ''' This function set the hyperparameters to the model and execute cross-validation.
            Return score (mean) results.'''
        model_to_fit = model.set_params(**hyperparameters)
        scoring = 'neg_mean_squared_error' if self.scoring == 'residual_mean_squared_error' else self.scoring

        cv_results = cross_val_score(model_to_fit, self.train_set, self.target_variable,
                                     cv=self.kf, scoring=scoring)

        if self.scoring == 'residual_mean_squared_error':
            score = np.sqrt(-cv_results).mean()

        else:
            score = cv_results.mean()

        return score

    def _objective(self, hyperparameters):
        """Objective function for Gradient Boosting Machine
        Hyperparameter Optimization"""

        # Keep track of evals
        global ITERATION
        global PROGRESS
        global min_loss_list

        model = deepcopy(self.model)
        ITERATION += 1

        best_score = self._model_eval(hyperparameters, model)
        if self.task == 'classification':
            loss = 1 - best_score
        else:
            loss = best_score

        min_loss_list.append(loss)
        # Display progress
        if ITERATION % PROGRESS == 0:
            print('Iteration: {}, Score: {}.'.format(ITERATION, (round(min(min_loss_list), 4))))
            min_loss_list = []

        # Dictionary with information for evaluation
        return {'iteration': ITERATION,
                'loss': loss,
                'hyperparameters': hyperparameters,
                'status': STATUS_OK}

    def fit(self, train_set, target_variable):
        self.train_set = train_set
        self.target_variable = target_variable

        # This istance save all results of all iteration
        tpe_trials = Trials()

        global ITERATION
        ITERATION = 0

        global min_loss_list
        min_loss_list = []

        global PROGRESS
        PROGRESS = int((self.n_iter) / 10)

        best = fmin(fn=self._objective, space=self.search_space,
                    algo=tpe.suggest, max_evals=self.n_iter,
                    trials=tpe_trials, rstate=np.random.RandomState(50))

        tpe_results = tpe_trials.best_trial['result']

        self.best_estimator_ = self.model.set_params(**tpe_results['hyperparameters'])
        self.results_ = tpe_trials.results

        if self.task == 'classification':
            self.best_score_ = 1 - tpe_results['loss']
        else:
            self.best_score_ = tpe_results['loss']

        if self.plot_loss:

            rand_trials = Trials()
            ITERATION = 0

            best = fmin(fn=self._objective, space=self.search_space,
                        algo=rand.suggest, max_evals=self.n_iter,
                        trials=rand_trials, rstate=np.random.RandomState(50))

            iterations = [x['iteration'] for x in tpe_trials.results]
            loss = [x['loss'] for x in tpe_trials.results]

            cross_validation_plot.plotting_iter_res(iterations, loss, title='Tpe Sequence of Values')

            iterations = [x['iteration'] for x in rand_trials.results]
            loss = [x['loss'] for x in rand_trials.results]

            cross_validation_plot.plotting_iter_res(iterations, loss, title='Random Sequence of Values')

        return self
