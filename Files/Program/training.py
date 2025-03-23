import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score
from deap import base, creator, tools, algorithms
plt.rcParams['font.family'] = 'Times New Roman'
def warn(*args, **kwargs): pass
warnings.warn = warn
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier


def train(args):

    ### Load Dataset ###
    D = pd.read_csv(args['TrainDataPath'])
    subfold = args['subfold']

    ### Splitting dataset into X, Y ###
    X_train = D.iloc[:, :-1].values
    Y_train = D.iloc[:, -1].values
    scale = StandardScaler()
    X_train = scale.fit_transform(X_train)

    if args['model']['XGB'] == 1:
        best_params = optParams(X_train, Y_train)
        model = XGBClassifier(best_params=best_params)
        # model = XGBClassifier()
        print('XGBoost ', end='')
        model.fit(X_train, Y_train)

    import joblib
    with open('../Files/'+subfold+'/dumpModel.pkl', 'wb') as File:
        joblib.dump(model, File)



def optParams(X,y):
    model = XGBClassifier()
    param_space = {
        'learning_rate': [0.1, 0.01, 0.001],
        'max_depth': range(3, 10),
        'n_estimators': range(100, 1000, 100),
        'subsample': np.arange(0.5, 1.0, 0.1),
        'colsample_bytree': np.arange(0.5, 1.0, 0.1),
        'reg_alpha': [0, 0.1, 1],
        'reg_lambda': [0, 0.1, 1]
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_space,
        scoring='neg_mean_squared_error',
        cv=5,
    )
    grid_search.fit(X, y)

    print("Best Parameters: ", grid_search.best_params_)
    print("Best Score: ", grid_search.best_score_)
    return grid_search.best_params_, grid_search.best_score_
    # ###############################################################

