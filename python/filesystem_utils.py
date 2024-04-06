"""
파일 시스템과 관련하여 유틸리티 함수와 클래스를 제공한다.

함수 목록
1. `to_pickle`
    pickle로 파일을 저장한다.
2. `read_pickle`
    pickle로 파일을 불러온다.
3. `mkdir`
    주어진 경로에 대해서 디렉토리를 생성한다.
"""

import pickle
import os

def to_pickle(data: object, path: str) -> None:
    """
    pickle로 파일을 저장한다.
    
    Args:
        data (object): 저장 할 데이터.
        path (str): 저장 경로.
        
    Returns:
        None.
    """
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path: str) -> object:
    """
    pickle로 파일을 불러온다.
    
    Args:
        path (str): 저장 경로.
    
    Returns:
        object: 저장된 파일 데이터.
    """
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def mkdir(paths: str|list) -> None:
    """
    주어진 경로에 대해서 디렉토리를 생성한다.
    
    Args:
        paths (str|list): 생성 할 디렉토리 경로 리스트.
        
    Returns:
        None.
    """
    if isinstance(paths,str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            print('directory created: {}'.format(path))