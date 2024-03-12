import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

#----------------------------------------------------------------------------#
# > 설명 : mse, mae, mape, smape를 산출하는 함수
# > 상세
#    - mape, smape의 경우에는 예측률 확인을 위해서 0~100의 범위로 제한
#    - 아래의 산출식으로는 mape, smape > 0 이므로, min(100,res)는 제외해도 될 것으로 보임
#----------------------------------------------------------------------------#
def mse(true, pred):
    return np.mean((pred-true)**2)

def mae(true, pred):
    return np.mean(np.abs(pred-true))

def mape(true, pred):
    res = np.mean(np.abs(pred-true)/true)*100
    res = min(100,res)
    res = max(0,res)
    if (np.isnan(res)) or (np.isinf(res)):
        return 100
    else:
        return res
    
def smape(true, pred):
    res = np.mean(np.abs(pred-true) / (np.abs(pred)+np.abs(true))) * 200
    res = min(100,res)
    res = max(0,res)
    if (np.isnan(res)) or (np.isinf(res)):
        return 100
    else:
        return res

# MASE 계산 함수
def mase(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    naive_forecast = y_true[:-1]
    naive_actual = y_true[1:]
    mae_naive = mean_absolute_error(naive_actual, naive_forecast)
    return mae / mae_naive

def NormalizedRMSE(y_true,y_pred,method):
    assert method in ['sd','mean','maxmin','iqr'], \
        print("method must be one of ['sd','mean','maxmin','iqr']")
    
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    
    if method=='sd':
        denominator = np.std(y_true)
    elif method=='mean':
        denominator = np.mean(y_true)
    elif method=='maxmin':
        denominator = np.max(y_true) - np.min(y_true)
    elif method=='iqr':
        denominator = np.quantile(y_true,0.75) - np.quantile(y_true,0.25)
    
    rmse = mean_squared_error(y_true=y_true,y_pred=y_pred)**0.5
    nrmse = rmse / denominator
    
    return nrmse

# scale이 서로 다른(mean, sd 등) 여러가지 segment들로부터 산출된 score(RMSE,MAPE 등)들에 대해서
# score들을 서로 비교 할 수 있게 normalize하는 함수
def normalize_scoring(scores,feature_range=[0,1]):
    assert isinstance(feature_range,list) & (len(feature_range)==2), \
        "feature_range must be list type and length 2"
    
    # scaling minimum to maximum
    minimum, maximum = feature_range[0]**2, feature_range[1]**2
    normalized_scores = minimum + ((scores-min(scores)) / (max(scores)-min(scores))) * (maximum-minimum)
    normalized_scores = np.sqrt(normalized_scores)
    
    return normalized_scores