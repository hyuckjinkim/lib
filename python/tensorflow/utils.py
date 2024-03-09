import numpy as np
import tensorflow as tf

# loss function for Keras
def huber_loss(true, pred):
    threshold = 1
    error = true - pred
    is_small_error = tf.abs(error) <= threshold
    small_error_loss = tf.square(error) / 2
    big_error_loss = threshold * (tf.abs(error) - (0.5 * threshold))
    return tf.where(is_small_error, small_error_loss, big_error_loss)

# Early Stopping function for DL
def Callback_EarlyStopping(LossList, min_delta=0.1, patience=20):
    
    # (1) patience : No early stopping for 2*patience epochs 
    es_patience = np.where(len(LossList)//patience < 2,False,True)
    
    # (2) min_delta
    if min_delta is not None:
    
        # Mean loss for last patience epochs and second-last patience epochs
        mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
        mean_recent = np.mean(LossList[::-1][:patience]) #last
        # you can use relative or absolute change
        delta_abs = np.abs(mean_recent - mean_previous) #abs change
        delta_abs = np.abs(delta_abs / mean_previous)  # relative change
        
        es_min_delta = np.where(delta_abs < min_delta,True,False)
    else:
        es_min_delta = False
        
    final_es = es_patience | es_min_delta
        
    return final_es


## optuna 적용 시, best callback에서 object를 저장함으로써, best model에 대해 한번 더 피팅하는 과정을 스킵 -> 속도향상
# from tensorflow.python.ops.gen_dataset_ops import dataset_cardinality_eager_fallback
def best_callback(study, trial):
    # # object 함수에 아래를 적용해주면 됨
    # trial.set_user_attr(key="Value",value=value)
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="Value",value=trial.user_attrs["Value"])