import sys
import numpy as np
import pandas as pd

#----------------------------------------------------------------------------#
# > 설명 : f-string과 carriage return을 같이 사용하기 위해서 만듦
# > 예시 : for i in range(1000): prt('\r{}'.format(str(i+1).zfill(4)))
#----------------------------------------------------------------------------#
def prt(x):
    sys.stdout.write(x)
    sys.stdout.flush()

#----------------------------------------------------------------------------#
# > 설명 : print 함수를 사용할 때, 색상을 설정하는 함수
# > 예시 : print(f'{color.BOLD}{color.BLUE}text...{color.END}')
#----------------------------------------------------------------------------#
class color:
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

#----------------------------------------------------------------------------#
# > 설명 : dataframe의 head와 tail을 원하는 상황에 맞게 보여주는 함수
# > 상세
#    - n : head, tail의 데이터 수
#    - sort_var : sorting을 원하는 컬럼명 리스트
#    - select_var : 선택을 원하는 컬럼명 리스트
#    - check_last_data : 마지막 데이터를 확인할건지 여부
#----------------------------------------------------------------------------#
def displays(data,n=5,sort_var=None,select_var=None,check_last_data=False):
    d = data.copy()
    
    if sort_var is not None:
        d = d.sort_values(sort_var)
        
    if select_var is None:
        select_var = d.columns
    
    print(f'> head({n})')
    display(d[select_var].head())
    if (check_last_data) & (1<=d.shape[0]<=n):
        display(np.array(d[select_var].tail().iloc[-1,:]))
        
    print('')
    
    if d.shape[0]>n:
        
        print(f'> tail({n})')
        display(d[select_var].tail())
        if check_last_data:
            display(np.array(d[select_var].tail().iloc[-1,:]))

#----------------------------------------------------------------------------#
# > 설명 : R의 setdiff와 동일한 결과를 보여주는 함수
#----------------------------------------------------------------------------#
def setdiff(x, y):
    #result = list(set(x)-set(y)) #순서가 뒤바뀜
    result = [x_value for x_value in x if x_value not in y]
    return result

#----------------------------------------------------------------------------#
# > 설명 : gc.collect()를 0이 될때까지 수행
#----------------------------------------------------------------------------#
import gc
def gc_collect_all(verbose=True):
    while True:
        gc_collected = gc.collect()
        if gc_collected==0:
            break
        else:
            if verbose:
                print('gc_collect: {}'.format(gc_collected))