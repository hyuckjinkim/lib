"""
파일 시스템과 관련하여 유틸리티 함수와 클래스를 제공한다.

함수 목록
1. `to_pickle`
    pickle로 파일을 저장한다.
2. `read_pickle`
    pickle로 파일을 불러온다.
3. `mkdir`
    주어진 경로에 대해서 디렉토리를 생성한다.
4. `get_script_extension`
    실행 중인 스크립트의 파일 확장자를 반환한다.
5. `delete_file`
    파일을 하나를 삭제한다.
6. `delete_files_with_pattern`
    입력된 pattern을 통해 파일을 삭제한다.
7. `load_from_google_drive`
    Google Drive 다운로드 링크로 부터 파일을 다운로드 받는다.
"""

import pickle
import os
import sys
import datetime
import glob
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def get_script_extension():
    try:
        # Jupyter Notebook인지 확인
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return '.ipynb'
        elif shell == 'TerminalInteractiveShell':
            return '.py'
    except NameError:
        # Jupyter가 아닌 경우, 스크립트 확장자를 확인
        script_path = os.path.abspath(sys.argv[0])
        _, extension = os.path.splitext(script_path)
        return extension
    return None

def delete_file(file_path: str) -> tuple:
    """
    파일을 하나를 삭제한다.

    Args:
        file_path (str): 삭제 할 파일경로.
    
    Returns:
        tuple: 
            첫 번째 요소는 파일 경로 (str)이며, 두 번째 요소는 파일 삭제 성공 여부 (bool)이다.
            - (file_path, True): 파일 삭제 성공.
            - (file_path, False): 파일 삭제 실패.
    """

    try:
        os.remove(file_path)
        return file_path, True
    except Exception as e:
        return file_path, False

def delete_files_with_pattern(pattern: str,
                              root_dir: str = None,
                              return_file_status: bool = False,
                              max_workers: int = os.cpu_count()) -> None | dict[list]:
    """
    입력된 pattern을 통해 파일을 삭제한다.

    Args:
        pattern (str): 삭제 할 파일경로의 패턴.
        return_file_status (str, optional): 파일 삭제가 성공적으로 이루어졌는지 정보를 반환할지 여부. default=False.
        max_workers (int, optional): 파일 삭제 시 사용 될 worker의 수. default=os.cpu_count().

    Returns:
        return_file_status가 False인 경우 None을 반환하고, True인 경우 'successed'와 'failed'의 key와 파일경로의 values로 이루어진 dict[list] 반환한다.

    Examples:
        ```file_status = delete_files_with_pattern(pattern='./**/._*', return_file_status=True)```
    """

    if root_dir is not None:
        pattern = os.path.join(root_dir, pattern)

    # 1. find files to delete
    start_time = datetime.datetime.now()
    print(f'\n> (1) Find files to delete')
    print(f'> Start   : {str(start_time)[:19]}')
    files_to_delete = glob.glob(pattern, recursive=True)
    end_time = datetime.datetime.now()
    run_time = (end_time-start_time).seconds / 60
    print(f'> End     : {str(end_time)[:19]}')
    print(f'> Elasped : {run_time:.2f} minutes')

    # 2. delete files
    print('\n> (2) Deleting files')
    extension = get_script_extension()
    assert extension in ['.py', '.ipynb'], "extension must be one of ['.py', '.ipynb']"

    # (1) if ipynb file : tqdm으로 처리
    if extension=='.ipynb':
        files_not_deleted = []
        with tqdm(total=len(files_to_delete)) as pbar:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(delete_file, file_path) for file_path in files_to_delete]
                for future in as_completed(futures):
                    file_path, success = future.result()
                    pbar.set_postfix(file=file_path)
                    pbar.update(1)
                    if not success:
                        print(f"Error deleting {file_path}")
                        files_not_deleted.append(file_path)

    # (2) if py file : tqdm 없이 자체적으로 만든 progress bar로 처리.
    elif extension=='.py':
        files_not_deleted = []
        total_files = len(files_to_delete)
        current = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(delete_file, file_path) for file_path in files_to_delete]
            for future in as_completed(futures):
                iter_start_time = datetime.datetime.now()
                
                file_path, success = future.result()
                
                iter_end_time = datetime.datetime.now()
                elapsed_time = (iter_end_time - start_time).total_seconds()
                current += 1
                percent = (current / total_files) * 100
                remaining_files = total_files - current
                estimated_time_remaining = (elapsed_time / current) * remaining_files
                
                progress_bar = (f"\r> Deleting Files: {current:,} / {total_files:,} ({percent:.2f}%)"
                                f", Elapsed: {elapsed_time:.2f}s"
                                f", ETA: {estimated_time_remaining:.2f}s")
                
                print(progress_bar, end='', flush=True)

    # return file status
    if return_file_status:
        file_status = {
            'successed' : list(set(files_to_delete)-set(files_not_deleted)),
            'failed' : files_not_deleted,
        }
        return file_status
    else:
        return {}

def load_from_google_drive(file_id: str):
    """
    Google Drive 다운로드 링크로 부터 파일을 다운로드 받는다.

    참조: https://dacon.io/competitions/official/235862/codeshare/4059

    Args:
        file_id (str): Google Drive 다운로드 링크의 파일 ID. 링크 URL 주소가 https://drive.google.com/file/d/abcdefgABCDEFG1234567/view와 같다면, file_id='abcdefgABCDEFG1234567'. 

    Returns:
        data/train.csv, data/test.csv, data/sample_submission.csv와 같이 다운로드 받는다.
    """
    from gdrivedataset import loader
    return loader.load_from_google_drive(file_id)