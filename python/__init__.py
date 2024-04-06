"""
python을 사용 할 때 도움이 될 수 있는 함수와 클래스를 제공한다.

함수 목록
1. `prt`
    f-string과 carriage return을 같이 사용하기 위한 print 함수.
2. `displays`
    데이터프레임의 head와 tail을 원하는 상황에 맞게 보여준다.
3. setdiff`
    x의 값들 중 y에 포함되지 않는 값들을 보여준다 (R의 setdiff와 동일한 결과를 보여준다).
4. `gc_collect_all`
    gc.collect()를 0이 될때까지 수행한다.
5. `str2bool`
    argparse 사용 시, type=bool로 하게되면 string으로 읽어들이는 문제를 해결하기위한 사용자정의함수.

클래스 목록
1. `color`
    print 함수를 사용할 때, 색상을 설정하는 코드를 모아둔 클래스.
"""

import sys
import numpy as np
import pandas as pd
from IPython.display import display
import gc
import argparse

def prt(text: str) -> None:
    """
    f-string과 carriage return을 같이 사용하기 위한 print 함수.
    
    Example:
        ```python
        for i in range(1000):
            prt('\r{}'.format(str(i+1).zfill(4)))
        ```
    
    Args:
        text (str): 텍스트.
    """
    
    sys.stdout.write(text)
    sys.stdout.flush()
    
    return None

class color:
    """
    print 함수를 사용할 때, 색상을 설정하는 코드를 모아둔 클래스.
    
    Example:
        ```python
        from lib.python import color
        print(f'{color.BOLD}{color.BLUE}text...{color.END}')
        ```
    
    """
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def displays(data: pd.DataFrame,
             n: int = 5,
             sort_columns: list = None,
             select_columns: list = None,
             check_last_data: bool = False) -> None:
    """
    데이터프레임의 head와 tail을 원하는 상황에 맞게 보여준다.
    
    Args:
        data (pd.DataFrame): 데이터프레임.
        n (int, optional): head, tail의 수. default=5.
        sort_columns (list, optional): sorting을 원하는 컬럼명 리스트. default=None.
        select_columns (list, optional): 선택을 원하는 컬럼명 리스트. default=None.
        check_last_data (bool, optional): 마지막 데이터를 확인할건지 여부. default=False.
        
    Returns:
        None.
    """
    d = data.copy()
    
    if sort_columns is not None:
        d = d.sort_values(sort_columns)
        
    if select_columns is None:
        select_columns = d.columns
    
    print(f'> head({n})')
    display(d[select_columns].head())
    if (check_last_data) & (1<=d.shape[0]<=n):
        display(np.array(d[select_columns].tail().iloc[-1,:]))
        
    print('')
    
    if d.shape[0]>n:
        
        print(f'> tail({n})')
        display(d[select_columns].tail())
        if check_last_data:
            display(np.array(d[select_columns].tail().iloc[-1,:]))
            
    return None

def setdiff(x: list, y: list) -> list:
    """
    x의 값들 중 y에 포함되지 않는 값들을 보여준다 (R의 setdiff와 동일한 결과를 보여준다).
    
    Args:
        x (list): 확인하고자하는 리스트.
        y (list): 제외하고자하는 리스트.
    
    Returns:
        list: x의 값들 중 y가 제외된 리스트.
    """
    #result = list(set(x)-set(y)) #순서가 뒤바뀜
    result = [x_value for x_value in x if x_value not in y]
    return result

def gc_collect_all(verbose: bool = True) -> None:
    """
    gc.collect()를 0이 될때까지 수행한다.
    
    Args:
        verbose (bool, optional): 진행현황을 출력할 것인지 여부. default=True.
    
    Returns:
        None.
    """
    while True:
        gc_collected = gc.collect()
        if gc_collected==0:
            break
        else:
            if verbose:
                print('gc_collect: {}'.format(gc_collected)) 
    return None
                
def str2bool(v: str) -> bool:
    """
    argparse 사용 시, type=bool로 하게되면 string으로 읽어들이는 문제를 해결하기위한 사용자 정의 함수.
    
    Args:
        v (str): input argument.
        
    Raises:
        input argument가 아래에 해당되지 않는 경우.
            `True: ('yes', 'true', 't', 'y', '1'), False: ('no', 'false', 'f', 'n', '0')`
    
    Returns:
        bool: input argument가 가리키는 True/False 값.
    """
    
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')