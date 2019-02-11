# numpy and pandas for data manipulation
import pandas as pd
import numpy as np

# model used for feature importances
import lightgbm as lgb
import xgboost as xgb
from sklearn import linear_model

# model for genetic selection
from .genetic_selection import GeneticSelectionCV

# utility for early stopping with a validation set
from sklearn.model_selection import train_test_split

# visualizations
from ..plottingTool.mltools_plot import feature_engineering_plot

# memory management
import gc

# utilities
from itertools import chain


class FeatureSelector():
    """
    Class for performing feature selection for machine learning or data preprocessing.

    Implements five different methods to identify features for removal

        1. Find columns with a single unique value
        2. Find collinear variables with a correlation greater than a specified correlation coefficient
        3. Find features with 0.0 feature importance from a gradient boosting machine (gbm)
        4. Find low importance features that do not contribute to a specified cumulative feature importance from the gbm

    Args:
    --------
        data : dataframe
            A dataset with observations in the rows and features in the columns

        labels : array or series, default = None
            Array of labels for training the machine learning model to find feature importances.
            These can be either binary labels (if task is 'classification') or continuous targets
            (if task is 'regression').
            If no labels are provided, then the feature importance based methods are not available.

    Attributes:
    --------

    ops : dict
        Dictionary of operations run and features identified for removal

    unique_stats : dataframe
        Number of unique values for all features

    record_single_unique : dataframe
        Records the features that have a single unique value

    corr_matrix : dataframe
        All correlations between all features in the data

    record_collinear : dataframe
        Records the pairs of collinear variables with a correlation coefficient above the threshold

    feature_importances : dataframe
        All feature importances from the gradient boosting machine

    record_zero_importance : dataframe
        Records the zero importance features in the data according to the gbm

    record_low_importance : dataframe
        Records the lowest importance features not needed to reach the threshold of cumulative importance
        according to the gbm


    Notes
    --------
        - If using feature importances, one-hot encoding is used for categorical variables which creates new columns

    """

    def __init__(self, data, labels=None):
        # Dataset and optional training labels
        self.data = data
        self.labels = labels

        if labels is None:
            print('No labels provided. Feature importance based methods are not available.')

        self.base_features = list(data.columns)
        self.one_hot_features = None

        # Dataframes recording information about features to remove
        self.record_single_unique = None
        self.record_collinear = None
        self.record_zero_importance = None
        self.record_low_importance = None

        self.unique_stats = None
        self.corr_matrix = None
        self.feature_importances = None

        # Dictionary to hold removal operations
        self.ops = {}

        self.one_hot_correlated = False

    def identify_single_unique(self):
        """Finds features with only a single unique value. NaNs do not count as a unique value. """

        # Calculate the unique counts in each column
        unique_counts = self.data.nunique()
        self.unique_stats = pd.DataFrame(unique_counts).rename(columns={'index': 'feature', 0: 'nunique'})
        self.unique_stats = self.unique_stats.sort_values('nunique', ascending=True)

        # Find the columns with only one unique count
        record_single_unique = pd.DataFrame(unique_counts[unique_counts == 1]).reset_index().rename(columns={'index': 'feature', 0: 'nunique'})

        to_drop = list(record_single_unique['feature'])

        self.record_single_unique = record_single_unique
        self.ops['single_unique'] = to_drop

        print('%d features with a single unique value.\n' % len(self.ops['single_unique']))

    def identify_collinear(self, correlation_threshold, one_hot=False):
        """
        Finds collinear features based on the correlation coefficient between features.
        For each pair of features with a correlation coefficient greather than `correlation_threshold`,
        only one of the pair is identified for removal.

        Using code adapted from:
            https://chrisalbon.com/machine_learning/feature_selection/drop_highly_correlated_features/

        Parameters
        --------

        correlation_threshold : float between 0 and 1
            Value of the Pearson correlation cofficient for identifying correlation features

        one_hot : boolean, default = False
            Whether to one-hot encode the features before calculating the correlation coefficients

        """

        self.correlation_threshold = correlation_threshold
        self.one_hot_correlated = one_hot
        
         # Calculate the correlations between every column
        if one_hot:
            
            # One hot encoding
            features = pd.get_dummies(self.data)
            self.one_hot_features = [column for column in features.columns if column not in self.base_features]

            # Add one hot encoded data to original data
            self.data_all = pd.concat([features[self.one_hot_features], self.data], axis = 1)
            
            corr_matrix = pd.get_dummies(features).corr()

        else:
            corr_matrix = self.data.corr()
        
        self.corr_matrix = corr_matrix
    
        # Extract the upper triangle of the correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k = 1).astype(np.bool))
        
        # Select the features with correlations above the threshold
        # Need to use the absolute value
        to_drop = [column for column in upper.columns if any(upper[column].abs() > correlation_threshold)]

        # Dataframe to hold correlated pairs
        record_collinear = pd.DataFrame(columns = ['drop_feature', 'corr_feature', 'corr_value'])

        # Iterate through the columns to drop to record pairs of correlated features
        for column in to_drop:

            # Find the correlated features
            corr_features = list(upper.index[upper[column].abs() > correlation_threshold])

            # Find the correlated values
            corr_values = list(upper[column][upper[column].abs() > correlation_threshold])
            drop_features = [column for _ in range(len(corr_features))]    

            # Record the information (need a temp df for now)
            temp_df = pd.DataFrame.from_dict({'drop_feature': drop_features,
                                             'corr_feature': corr_features,
                                             'corr_value': corr_values})

            # Add to dataframe
            record_collinear = record_collinear.append(temp_df, ignore_index = True)

        self.record_collinear = record_collinear
        self.ops['collinear'] = to_drop
        
        print('%d features with a correlation magnitude greater than %0.2f.\n' % (len(self.ops['collinear']), self.correlation_threshold))

    def identify_zero_importance(self, task, method="xgboost", eval_metric=None, n_iterations=10, early_stopping=True):
        """
        
        Identify the features with zero importance according to a gradient boosting machine.
        The gbm can be trained with early stopping using a validation set to prevent overfitting. 
        The feature importances are averaged over `n_iterations` to reduce variance. 
        
        Uses the LightGBM implementation (http://lightgbm.readthedocs.io/en/latest/index.html) or the XGBoost

        Parameters 
        --------

        eval_metric : string
            Evaluation metric to use for the gradient boosting machine for early stopping. Must be
            provided if `early_stopping` is True

        task : string
            The machine learning task, either 'classification' or 'regression'

        n_iterations : int, default = 10
            Number of iterations to train the gradient boosting machine

        early_stopping : boolean, default = True
            Whether or not to use early stopping with a validation set when training


        Notes
        --------

        - Features are one-hot encoded to handle the categorical variables before training.
        - The gbm is not optimized for any particular task and might need some hyperparameter tuning
        - Feature importances, including zero importance features, can change across runs
        - LightGBM doesn't works on Mac OS Sierra

        """

        if early_stopping and eval_metric is None:
            raise ValueError("eval metric must be provided with early stopping. Examples include 'auc' for classification or \
                'l2' for regression.")

        if self.labels is None:
            raise ValueError("No training labels provided.")

        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        features = np.array(features)
        labels = np.array(self.labels).reshape((-1, ))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        print('Training Gradient Boosting Model\n')

        # Iterate through each fold
        for i in range(0, n_iterations):

            if task == 'classification':
                if method == "lightgbm":
                    model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
                elif method == "xgboost":
                    model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05)
                else:
                    raise ValueError("Invalid method. Use 'lightgbm' or 'xgboost'.")

            elif task == 'regression':
                if method == "lightgbm":
                    model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
                elif method == "xgboost":
                    model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
                else:
                    raise ValueError("Invalid method. Use 'lightgbm' or 'xgboost'.")

            else:
                raise ValueError('Task must be either "classification" or "regression"')

            print('Iteration {}\n'.format(i))

            # If training using early stopping need a validation set
            if early_stopping:

                train_features, valid_features, train_labels, valid_labels = train_test_split(features, labels, test_size=0.15)

                # Train the model with early stopping
                model.fit(train_features, train_labels, eval_metric=eval_metric,
                          eval_set=[(valid_features, valid_labels)],
                          early_stopping_rounds=100)

            else:
                model.fit(features, labels)

            # Record the feature importances
            feature_importance_values += model.feature_importances_ / n_iterations

        feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})

        # Sort features according to importance
        feature_importances = feature_importances.sort_values('importance', ascending=False).reset_index(drop=True)

        # Normalize the feature importances to add up to one
        feature_importances['normalized_importance'] = feature_importances['importance'] / feature_importances['importance'].sum()
        feature_importances['cumulative_importance'] = np.cumsum(feature_importances['normalized_importance'])

        # Extract the features with zero importance
        record_zero_importance = feature_importances[feature_importances['importance'] == 0.0]

        to_drop = list(record_zero_importance['feature'])

        self.feature_importances = feature_importances
        self.record_zero_importance = record_zero_importance
        self.ops['zero_importance'] = to_drop

        print('\n%d features with zero importance after one-hot encoding.\n' % len(self.ops['zero_importance']))
    
    
    def identify_low_importance(self, cumulative_importance):
        """
        Finds the lowest importance features not needed to account for `cumulative_importance` fraction
        of the total feature importance from the gradient boosting machine. As an example, if cumulative
        importance is set to 0.95, this will retain only the most important features needed to
        reach 95% of the total feature importance. The identified features are those not needed.

        Parameters
        --------
        cumulative_importance : float between 0 and 1
            The fraction of cumulative importance to account for

        """

        self.cumulative_importance = cumulative_importance

        # The feature importances need to be calculated before running
        if self.feature_importances is None:
            raise NotImplementedError("""Feature importances have not yet been determined.
                                         Call the `identify_zero_importance` method first.""")

        # Make sure most important features are on top
        self.feature_importances = self.feature_importances.sort_values('cumulative_importance')

        # Identify the features not needed to reach the cumulative_importance
        record_low_importance = self.feature_importances[self.feature_importances['cumulative_importance'] > cumulative_importance]

        to_drop = list(record_low_importance['feature'])

        self.record_low_importance = record_low_importance
        self.ops['low_importance'] = to_drop

        print('%d features required for cumulative importance of %0.2f after one hot encoding.' % (len(self.feature_importances) - len(self.record_low_importance),
                                                                                                   self.cumulative_importance))
        print('%d features do not contribute to cumulative importance of %0.2f.\n' % (len(self.ops['low_importance']),
                                                                                      self.cumulative_importance))

    
    def ga_selector(self, task, method="xgboost", ga_params = {}):
        """Feature selection with genetic algorithm.
            
            Parameters
            ----------
            
            task (string)   >>> classification or regression.
            method (string) >>> A supervised learning estimator.
            
            ga_params (dict) >>> can contain:
                cv : int, cross-validation generator or an iterable, optional
                Determines the cross-validation splitting strategy.
                Possible inputs for cv are:
                
                - None, to use the default 3-fold cross-validation,
                - integer, to specify the number of folds.
                - An object to be used as a cross-validation generator.
                - An iterable yielding train/test splits.
                
                For integer/None inputs, if ``y`` is binary or multiclass,
                :class:`StratifiedKFold` used. If the estimator is a classifier
                or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.
                
                Refer :ref:`User Guide <cross_validation>` for the various
                cross-validation strategies that can be used here.
                
                scoring : string, callable or None, optional, default: None
                A string (see model evaluation documentation) or
                a scorer callable object / function with signature
                ``scorer(estimator, X, y)``.
                
                fit_params : dict, optional
                Parameters to pass to the fit method.
                
                verbose : int, default=0
                Controls verbosity of output.
                
                n_jobs : int, default 1
                Number of cores to run in parallel.
                Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
                to number of cores.
                
                n_population : int, default=300
                Number of population for the genetic algorithm.
                
                crossover_proba : float, default=0.5
                Probability of crossover for the genetic algorithm.
                
                mutation_proba : float, default=0.2
                Probability of mutation for the genetic algorithm.
                
                n_generations : int, default=40
                Number of generations for the genetic algorithm.
                
                crossover_independent_proba : float, default=0.1
                Independent probability of crossover for the genetic algorithm.
                
                mutation_independent_proba : float, default=0.05
                Independent probability of mutation for the genetic algorithm.
                
                tournament_size : int, default=3
                Tournament size for the genetic algorithm.
                
                caching : boolean, default=False
                If True, scores of the genetic algorithm are cached.
            """

        if self.labels is None:
            raise ValueError("No training labels provided.")

        # One hot encoding
        features = pd.get_dummies(self.data)
        self.one_hot_features = [column for column in features.columns if column not in self.base_features]

        # Add one hot encoded data to original data
        self.data_all = pd.concat([features[self.one_hot_features], self.data], axis=1)

        # Extract feature names
        feature_names = list(features.columns)

        # Convert to np array
        #features = np.array(features)
        labels = np.array(self.labels).reshape((-1, ))

        # Empty array for feature importances
        feature_importance_values = np.zeros(len(feature_names))

        if task == 'classification':
            if method == "lightgbm":
                model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05)
            elif method == "xgboost":
                model = xgb.XGBClassifier(n_estimators=1000, learning_rate=0.05)
            else:
                raise ValueError("Invalid method. Use 'lightgbm' or 'xgboost'.")

        elif task == 'regression':
            if method == "lightgbm":
                model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05)
            elif method == "xgboost":
                model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05)
            else:
                raise ValueError("Invalid method. Use 'lightgbm' or 'xgboost'.")

        else:
            raise ValueError('Task must be either "classification" or "regression"')

        selector = GeneticSelectionCV(model, **ga_params)
        selector = selector.fit(features, y=self.labels)

        to_drop = list(features.iloc[:, ~selector.support_].columns.values)
        self.ops['ga_selection'] = to_drop

    
    def remove(self, methods, keep_one_hot=True):
        """
        Remove the features from the data according to the specified methods.

        Parameters
        --------
            methods : 'all' or list of methods
                If methods == 'all', any methods that have identified features will be used
                Otherwise, only the specified methods will be used.
                Can be one of ['missing', 'single_unique', 'collinear', 'zero_importance', 'low_importance']
            keep_one_hot : boolean, default = True
                Whether or not to keep one-hot encoded features

        Return
        --------
            data : dataframe
                Dataframe with identified features removed


        Notes
        --------
            - If feature importances are used, the one-hot encoded columns will be added to the data
                (and then may be removed)
            - Check the features that will be removed before transforming data!

        """

        features_to_drop = []

        if methods == 'all':

            # Need to use one-hot encoded data as well
            data = self.data_all

            print('{} methods have been run\n'.format(list(self.ops.keys())))

            # Find the unique features to drop
            features_to_drop = set(list(chain(*list(self.ops.values()))))

        else:
            # Need to use one-hot encoded data as well
            if 'zero_importance' in methods or 'low_importance' in methods or 'ga_selection' in methods or self.one_hot_correlated:
                data = self.data_all

            else:
                data = self.data

            # Iterate through the specified methods
            for method in methods:

                # Check to make sure the method has been run
                if method not in self.ops.keys():
                    raise NotImplementedError('%s method has not been run' % method)

                # Append the features identified for removal
                else:
                    features_to_drop.append(self.ops[method])

            # Find the unique features to drop
            features_to_drop = set(list(chain(*features_to_drop)))

        features_to_drop = list(features_to_drop)

        if not keep_one_hot:

            if self.one_hot_features is None:
                print('Data has not been one-hot encoded')
            else:

                features_to_drop = list(set(features_to_drop) | set(self.one_hot_features))

        # Remove the features and return the data
        data = data.drop(features_to_drop, axis=1)
        self.removed_features = features_to_drop

        if not keep_one_hot:
            print('Removed %d features including one-hot features.' % len(features_to_drop))
        else:
            print('Removed %d features.' % len(features_to_drop))

        return data

    def plot_unique(self, figsize=(7, 3), dpi=200, features_lim=None):
        """Histogram of number of unique values in each feature"""
        if self.record_single_unique is None:
            raise NotImplementedError('Unique values have not been calculated. Run `identify_single_unique`')

        # Histogram of number of unique values
        if features_lim:
            data = self.unique_stats[:features_lim]
        else:
            data = self.unique_stats

        feature_engineering_plot._plot_unique_value(data, figsize)

    def plot_collinear(self, plot_all=False, figsize=(10, 6)):
        """
        Heatmap of the correlation values. If plot_all = True plots all the correlations otherwise
        plots only those features that have a correlation above the threshold

        Notes
        --------
            - Not all of the plotted correlations are above the threshold because this plots
            all the variables that have been idenfitied as having even one correlation above the threshold
            - The features on the x-axis are those that will be removed. The features on the y-axis
            are the correlated features with those on the x-axis

        Code adapted from https://seaborn.pydata.org/examples/many_pairwise_correlations.html
        """

        if self.record_collinear is None:
            raise NotImplementedError('Collinear features have not been idenfitied. Run `identify_collinear`.')

        if plot_all:
            corr_matrix_plot = self.corr_matrix
            title = 'All Correlations'

        else:
            # Identify the correlations that were above the threshold
            # columns (x-axis) are features to drop and rows (y_axis) are correlated pairs
            corr_matrix_plot = self.corr_matrix.loc[list(set(self.record_collinear['corr_feature'])),
                                                    list(set(self.record_collinear['drop_feature']))]

            title = "Correlations Above Threshold"

        feature_engineering_plot._plot_collinear(corr_matrix_plot, title, figsize)

    def plot_feature_importances(self, plot_n=15, threshold=None, figsize=(2000, 800)):
        """
        Plots `plot_n` most important features and the cumulative importance of features.
        If `threshold` is provided, prints the number of features needed to reach `threshold` cumulative importance.

        Parameters
        --------

        plot_n : int, default = 15
            Number of most important features to plot. Defaults to 15 or the maximum number of features
            whichever is smaller

        threshold : float, between 0 and 1 default = None
            Threshold for printing information about cumulative importances

        """

        if self.record_zero_importance is None:
            raise NotImplementedError('Feature importances have not been determined. Run `idenfity_zero_importance`')

        # Need to adjust number of features if greater than the features in the data
        plot_n = self.feature_importances.shape[0] - 1 if plot_n > self.feature_importances.shape[0] else plot_n

        feature_engineering_plot._plot_features_importance(self.feature_importances, plot_n, threshold, figsize)

