import os
import random
import numpy as np
import tensorflow as tf

#----------------------------------------------------------------------------#
# > 설명 : 난수생성, 모델링 등에서 결과값의 변동이 없도록 하기위해 seed를 fix하는 함수
#----------------------------------------------------------------------------#
def seed_everything(seed: int=0):
    
    # (참조) https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras

    # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
    os.environ['PYTHONHASHSEED']=str(seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    random.seed(seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    np.random.seed(seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    tf.random.set_seed(seed)
    # for later versions: 
    # tf.compat.v1.set_random_seed(seed_value)

    # 5. Configure a new global `tensorflow` session
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)