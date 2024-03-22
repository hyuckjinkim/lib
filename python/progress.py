"""
작업의 progress를 관리하는 함수와 클래스를 제공한다.

함수 목록
1. `tqdm_by_verbose`
    verbose 인자에 따라서 tqdm을 넣을건지 아닌지를 선택해주는 함수.
2. `print_by_verbose`
    verbose 인자에 따라서 print를 할건지 안할건지를 선택해주는 함수.

클래스 목록
1. `ProgressManager`
    작업의 progress를 관리하는 클래스.
"""

# installed libraries
import logging
from tqdm.auto import tqdm

def tqdm_by_verbose(x, verbose: bool, **kwargs):
    """
    verbose 인자에 따라서 tqdm을 넣을건지 아닌지를 선택해주는 함수.
    
    Args:
        x (any): tqdm()에 넣을 객체.
        verbose (bool): tqdm progress bar를 출력 할 것 인지를 설정.
        **kwargs: tqdm()에 들어갈 추가 인자.
    
    Returns:
        verbose가 True인 경우에는 tqdm(x)가 반환되고, verbose=False인 경우에는 x값 그대로 반환된다.
    """
    if verbose:
        return tqdm(x,**kwargs)
    else:
        return x
    
def print_by_verbose(text: str, verbose: bool, logger: logging.RootLogger = None) -> None:
    """
    verbose 인자에 따라서 print를 할건지 안할건지를 선택해주는 함수.
    
    Args:
        x (str): 출력 할 텍스트.
        verbose (bool): 출력 할 것 인지를 설정.
        logger (logging.RootLogger, optional): logger 객체. logger 객체가 전달 된 경우에는 verbose와 관계없이 logger.info()가 실행된다. default=None.
    
    Returns:
        None.
    """
    
    if logger is None:
        if verbose:
            print(text)
    else:
        logger.info(text)

class ProgressManager:
    """
    작업의 progress를 관리하는 클래스.
    
    Attributes:
        job_names (list): 작업목록의 주제로 이루어진 List.
    """
    
    def __init__(self, job_names) -> None:
        """ProgressManager 클래스의 생성자.
        
        Args:
            job_names (list): 작업목록의 주제로 이루어진 List.
        """
        
        self.job_names = job_names
        self.job_iters = iter(range(len(job_names)))
        self.total_iters = len(job_names)
        
    def __call__(self):
        """
        객체를 호출할 때 실행되는 메서드로, 입력된 작업목록에 대한 progress text를 출력한다.
        
        * 출력예시
            > [05/10] 작업목록 5번
        """
        
        self.iter = next(self.job_iters)
        iter_zfill = str(self.iter+1).zfill(len(str(self.total_iters)))
        
        lines = '-'*80
        self.main_progress = '> [{}/{}] {}'.format(iter_zfill,self.total_iters,self.job_names[self.iter])
        self.progress = f'\n{lines}\n{self.main_progress}\n{lines}'
        print(self.progress)