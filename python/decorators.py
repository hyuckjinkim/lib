import time
from functools import wraps
import numpy as np

def retry_with_delay(retry_count: int = 3,
                     delay_seconds: float|list|tuple = 1,
                     verbose: int = 2,
                     verbose_period: int = 1):
    
    if isinstance(delay_seconds,list|tuple):
        if len(delay_seconds)!=2:
            raise ValueError('If delay_seconds is tuple or list, the length must be 2.')
        min_, max_ = delay_seconds[0], delay_seconds[1]
        delay_seconds = np.random.uniform(min_,max_)
        
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count):
                try:
                    # 함수 호출
                    return func(*args, **kwargs)
                except Exception as e:
                    # progress
                    if (attempt+1) % verbose_period == 0:
                        if verbose==0:
                            pass
                        elif verbose==1:
                            print(f"Retry {attempt + 1}/{retry_count}")
                        elif verbose>1:
                            print(f"Retry {attempt + 1}/{retry_count}, Error: {e}")
                    # 지정된 시간만큼 대기
                    time.sleep(delay_seconds)
            # 모든 재시도 실패 후 예외를 다시 발생
            raise Exception(f"Failed after {retry_count} attempts")
        return wrapper
    return decorator