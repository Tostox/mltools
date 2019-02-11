import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

class StackingModels():

    def __init__(self, algorithms):
        self.algorithms = algorithms 
        
    def _cv_predict(self, clf, x_train, y_train, x_test, kf):
    
        ntrain = x_train.shape[0]
        oof_train = np.zeros((ntrain,))
        ntest = x_test.shape[0]
        oof_test = np.zeros((ntest,))
        for train_index, test_index in kf.split(x_train):
            x_tr, x_te = x_train[train_index], x_train[test_index]
            y_tr, y_te = y_train[train_index], y_train[test_index]
            clf.fit(x_tr, y_tr)
            oof_train[test_index] = clf.predict(x_te)
        #refit on the entire training set
        clf.fit(x_train, y_train)
        #predict on the test set
        oof_test[:] = clf.predict(x_test)
        out = [oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)]
        return out
    
    def fit(self, x_train, y_train, test, n_folds = 10, seed = 123):

        kf = KFold(n_splits= n_folds, random_state=seed)
    
        out_train = pd.DataFrame()
        out_test = pd.DataFrame()
        for model in self.algorithms:
            print("Fitting {}".format(str(model)))
            model_prediction = self._cv_predict(model, x_train, y_train, test, kf)
            out_train = pd.concat([out_train, pd.DataFrame(model_prediction[0])], axis=1)
            out_test = pd.concat([out_test, pd.DataFrame(model_prediction[1])], axis=1)

        col_names = [str(model).split('(')[0] for model in self.algorithms]
        out_train.columns = col_names
        out_test.columns = col_names
        return out_train, out_test














