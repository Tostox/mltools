def create_model_dict():
    """ This function create a dict that contain all available methods.

        Returns:
        --------
            dict.
    """
    model_dict = {
    "MultinomialNB": {
        "path": "from sklearn.naive_bayes import MultinomialNB",
        "init": "MultinomialNB()"
        },
    "GaussianNB": {
        "path": "from sklearn.naive_bayes import GaussianNB",
        "init": "GaussianNB()"
        },
    "SVM": {
        "path": "from sklearn.svm import SVC",
        "init": "SVC()"
        },
    "SGD_SVM": {
        "path": "from sklearn.linear_model import SGDClassifier",
        "init": "SGDClassifier()"
        },
    "SGD_regressor": {
        "path": "from sklearn.linear_model import SGDRegressor",
        "init": "SGDRegressor()"
        },
    "SVM_regressor": {
        "path": "from sklearn.svm import SVR",
        "init": "SVR()"
        },
    "KNN": {
        "path": "from sklearn.neighbors import KNeighborsClassifier",
        "init": "KNeighborsClassifier()"
        },
    "KNN_regressor": {
        "path": "from sklearn.neighbors import KNeighborsRegressor",
        "init": "KNeighborsRegressor()"
        },
    "DT": {
        "path": "from sklearn.tree import DecisionTreeClassifier",
        "init": "DecisionTreeClassifier()"
        },
    "DT_regressor": {
        "path": "from sklearn.tree import DecisionTreeRegressor",
        "init": "DecisionTreeRegressor()"
        },
    "RandomForest": {
        "path": "from sklearn.ensemble import RandomForestClassifier",
        "init": "RandomForestClassifier()"
        },
    "RandomForest_regressor": {
        "path": "from sklearn.ensemble import RandomForestRegressor",
        "init": "RandomForestRegressor()"
        },
    "LinearRegression": {
        "path": "from sklearn.linear_model import LinearRegression",
        "init": "LinearRegression()"
        },
    "NonNegativeLinearRegression": {
        "path": "from civismlext import NonNegativeLinearRegression",
        "init": "NonNegativeLinearRegression()"
        },
    "LogisticRegression": {
        "path": "from sklearn.linear_model import LogisticRegression",
        "init": "LogisticRegression()"
        },
    "LassoRegression": {
        "path": "from sklearn.linear_model import Lasso",
        "init": "Lasso()"
        },
    "RidgeRegression": {
        "path": "from sklearn.linear_model import Ridge",
        "init": "Ridge()"
        },
    "ElasticNet": {
        "path": "from sklearn.linear_model import ElasticNet",
        "init": "ElasticNet()"
        },
    "LinearGAM": {
        "path": "from pygam import LinearGAM",
        "init": "LinearGAM()"
        },
    "PoissonGAM": {
        "path": "from pygam import PoissonGAM",
        "init": "PoissonGAM()"
        },
    "GammaGAM": {
        "path": "from pygam import GammaGAM",
        "init": "GammaGAM()"
        },
    "MARS": {
        "path": "from pyearth import Earth",
        "init": "Earth()"
        },
    "GradientBoosting": {
        "path": "from sklearn.ensemble import GradientBoostingClassifier",
        "init": "GradientBoostingClassifier()"
        },
    "GradientBoosting_regressor": {
        "path": "from sklearn.ensemble import GradientBoostingRegressor",
        "init": "GradientBoostingRegressor()"
        },
    "AdaBoost": {
        "path": "from sklearn.ensemble import AdaBoostClassifier",
        "init": "AdaBoostClassifier()"
        },
    "AdaBoost_regressor": {
        "path": "from sklearn.ensemble import AdaBoostRegressor",
        "init": "AdaBoostRegressor()"
        },
    "LightGBM": {
        "path": "import lightgbm as lgb",
        "init": "lgb.LGBMClassifier()"
        },
    "LightGBM_regressor": {
        "path": "import lightgbm as lgb",
        "init": "lgb.LGBMRegressor()"
        },
    "XGBoost": {
        "path": "import xgboost as xgb",
        "init": "xgb.XGBClassifier()"
        },
    "XGBoost_regressor": {
        "path": "import xgboost as xgb",
        "init": "xgb.XGBRegressor()"
        }
    }

    return model_dict

def create_score_dict():
    """ This function create a dict that contain all available methods.

        Returns:
        --------
            dict.
    """

    score_dict = {
        "accuracy": "make_scorer(accuracy_score)",
        "auc": "make_scorer(roc_auc_score)",
        "f1": "make_scorer(f1_score)",
        "f1_multiclass": "make_scorer(f1_score, average='weighted')",
        "mcc": "make_scorer(matthews_corrcoef)",
        "precision": "make_scorer(precision_score)",
        "precision_multiclass": "make_scorer(precision_score, average='weighted')",
        "recall": "make_scorer(recall_score)",
        "recall_multiclass": "make_scorer(recall_score, average='weighted')",
        "mse": "make_scorer(mean_squared_error)",
        "rmse": "make_scorer(residual_mean_squared_error)",
        "mae": "make_scorer(mean_absolute_error)",
        "medae": "make_scorer(median_absolute_error)",
        "msle": "make_scorer(mean_squared_log_error)",
        "r2": "make_scorer(r2_score)"
        }

    return score_dict
