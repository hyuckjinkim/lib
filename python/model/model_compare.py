# 작성이 오래되서 수정필요

import time
from tqdm import tqdm

import pandas as pd
import numpy as np

from scipy.stats import stats

import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
# from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score,StratifiedKFold,train_test_split
from sklearn.metrics import f1_score,make_scorer,r2_score,log_loss
# from sklearn.compose import ColumnTransformer

# from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeClassifier, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, ExtraTreesRegressor, AdaBoostRegressor, AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import seaborn as sns

def AUTOML_COMP(
    train,
    target,
    objective,
    iterations=100,
    scoring=None,
    test=None,
    ignore_features=None,
    n_splits=10,
    fit_model=False,
    seed=0
):

    # copy
    X_train = train.drop([target], axis=1)
    y_train = train[target]
    if test is not None:
        X_test = test.drop([target], axis=1)
        y_test = test[target]

    # remove needless features
    if ignore_features is not None:
        features = list(set(X_train.columns)-set(ignore_features))
    else:
        features = list(X_train.columns)

    # 모형적합
    start_time = time.time()

    # binary / regression
    models = []
    if objective == 'binary':

        if scoring is None:
            scoring = 'accuracy'

        models.append(('LR', LogisticRegression(
            solver='liblinear', multi_class='ovr', random_state=seed)))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier(random_state=seed)))
        models.append(('NB', GaussianNB()))  # Gaussian Naive Bayes
        models.append(('SVM', SVC(gamma='auto', random_state=seed)))
        models.append(('RFC', RandomForestClassifier(random_state=seed)))
        models.append(('XGBC', XGBClassifier(
            iterations=iterations, verbosity=0, random_state=seed)))
        models.append(('LGBMC', LGBMClassifier(random_state=seed)))
        models.append(('AdaC', AdaBoostClassifier(random_state=seed)))
        models.append(('Cat', CatBoostClassifier(
            iterations=iterations, silent=True, random_state=seed)))

    elif objective == 'regression':

        if scoring is None:
            scoring = 'neg_mean_squared_error'

        models.append(('LR', LinearRegression()))
        models.append(('RIDGE', RidgeClassifier()))
        models.append(('LASSO', Lasso(random_state=seed)))
        models.append(('KNN', KNeighborsRegressor()))
        models.append(('CART', DecisionTreeRegressor(random_state=seed)))
        models.append(('EN', ElasticNet(random_state=seed)))
        models.append(('SVM', SVR()))
        models.append(('RFR', RandomForestRegressor(random_state=seed)))
        models.append(('XGBR', XGBRegressor(
            iterations=iterations, verbosity=0, random_state=seed)))
        models.append(('LGBMR', LGBMRegressor(random_state=seed)))
        models.append(('AdaR', AdaBoostRegressor(random_state=seed)))
        models.append(('Cat', CatBoostRegressor(
            iterations=iterations, silent=True, random_state=seed)))

    # kfold cross validation
    if fit_model:

        results = []
        names = []
        msgs = []

        pbar = tqdm(models)
        for name, model in pbar:
            pbar.set_description(f'fitting... ({name})')

            kfold = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
            cv_results = cross_val_score(model,
                                         X=X_train,
                                         y=y_train,
                                         cv=kfold,
                                         scoring=scoring,
                                         verbose=0)
            results.append(cv_results)
            names.append(name)
            msgs.append((name, cv_results.mean(), cv_results.std()))

        end_time = time.time()
        running_time = (end_time - start_time)/60
        running_time = f'{running_time:.1f} Mins'

    ret = {}

    ret['X_train'] = X_train
    ret['y_train'] = y_train

    if fit_model:
        ret['run_time'] = running_time
        ret['message'] = msgs
        ret['model'] = models
        ret['cv_result'] = results

    if test is not None:
        ret['X_test'] = X_test

    return(ret)

def AUTOML_COMP_PLOT(
    object,
    title='Algorithm Comparision', 
    logscale=True,
    n_best_fun=np.min, 
    n_best=12, 
    sorting=False
    ):
        
    cv_res_df = pd.DataFrame(
        np.transpose(object['cv_result']),
        columns=[name for name,model in object['model']]
    )
    
    n_best_col = cv_res_df.apply(n_best_fun).sort_values(ascending=False)[:n_best].index
    if sorting:
        cv_res_df = cv_res_df[n_best_col]
    else:
        cv_res_df = cv_res_df[pd.Series(cv_res_df.columns).isin(n_best_col)]

    fig,ax = plt.subplots(figsize=(15, 5))
    if logscale:
        ax.set_yscale('symlog')
    sns.boxplot(data=cv_res_df)
    sns.swarmplot(data=cv_res_df, marker='*', s=7, color='crimson')
    plt.title(title, fontsize=20)
    plt.show()