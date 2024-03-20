import time
from functools import wraps

def retry_with_delay(retry_count: int = 3,
                     delay_seconds: float = 1,
                     verbose: int = 2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(retry_count):
                try:
                    # 함수 호출
                    return func(*args, **kwargs)
                except Exception as e:
                    if verbose==1:
                        print(f"Retry {attempt + 1}/{retry_count}")
                    elif verbose>1:
                        print(f"Retry {attempt + 1}/{retry_count}, Error: {e}")
                    # 지정된 시간만큼 대기
                    time.sleep(delay_seconds)
            # 모든 재시도 실패 후 예외를 다시 발생
            raise Exception(f"Failed after {retry_count} attempts")
        return wrapper
    return decorator