"""
shap package를 통해서 활용 할 수 있는 함수와 클래스를 제공한다.

함수 목록
1. `get_importance`
    주어진 model 및 X에 대해서 shap importance를 반환한다.
"""

import pandas as pd
import numpy as np
import shap
import catboost

def get_importance(model: catboost.core.CatBoostRegressor|catboost.core.CatBoostClassifier,
                   X: pd.DataFrame,
                   normalize: bool = False) -> pd.DataFrame:
    """
    주어진 model 및 X에 대해서 shap importance를 반환한다.
    
    Args:
        model: 학습이 완료된 모델 객체.
        X (pd.DataFrame): shap importance를 계산 할 데이터셋.
        normalize (bool, optional): shap importance를 normalize 할 것인지 여부. default=False.
        
    Raises:
        model의 type이 아래의 지원목록에 해당하지 않는 경우.
            catboost:
                `catboost.core.CatBoostRegressor`, `catboost.core.CatBoostClassifier`
    
    Returns:
        pd.DataFrame: 
    """
    
    enable_model_type = [
        catboost.core.CatBoostRegressor,
        catboost.core.CatBoostClassifier,
    ]
    not_enable_error_text = f"{type(model)} is not currently enabled. Enable model type: {enable_model_type}."
    assert type(model) in enable_model_type, not_enable_error_text

    # get shap values
    shap_values = shap.TreeExplainer(model).shap_values(X)
    #shap.summary_plot(shap_values, X, plot_type="bar")

    # get model informations
    # (1) catboost
    if type(model) in [catboost.core.CatBoostRegressor,catboost.core.CatBoostClassifier]:
        best_iteration = model.best_iteration_
    else:
        raise ValueError(not_enable_error_text)
    
    # make the dataframe
    feature_names = X.columns
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_mean_df = np.abs(shap_df.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(feature_names, shap_mean_df)), columns=['feature','shap_feature_importance'])
    shap_importance = shap_importance.sort_values('shap_feature_importance',ascending=False).reset_index(drop=True)
    if normalize:
        shap_importance['shap_feature_importance'] /= shap_importance['shap_feature_importance'].sum()
    
    # additional information
    shap_importance['best_iteration'] = best_iteration
    # shap_importance['n_train'] = len(model.X_train)
    # shap_importance['n_val'] = len(model.X_val)

    return shap_importance