#Configparser 라이브러리를 사용하여 config 파일을 읽어옵니다.
import os
from configparser import ConfigParser

def _get_current_file_path() -> str:
    """현재 실행 중인 파일이 속한 폴더의 경로를 나타냅니다.

    Returns:
        str: 폴더 경로
    """
    current_path = os.path.abspath(__file__)
    current_folder_path = os.path.dirname(current_path)
    
    return current_folder_path

def get_config(config_file_name : str, section: str = None, key: str = None) -> str:
    """config 파일을 읽어서 반환합니다.
    
    Args:
        config_file_name(str) : config 파일의 이름
        section(str) : ini 파일의 섹일. default=None.
        key(str) : ini 파일의 키. default=None.

    Returns:
        config 파일의 내용.
        
        section이 입력되지 않거나 section까지만 입력되는 경우:  str
        key까지 입력되는 경우: configparser
    """
    
    current_folder_path = _get_current_file_path()
    
    config = ConfigParser()
    config.read(f'{current_folder_path}/{config_file_name}')
    
    # 입력된 section,key 기준으로 마지막 level까지 return
    if (section is None) and (key is None):
        return config
    elif (section is not None) and (key is None):
        return config[section]
    elif (section is not None) and (key is not None):
        return config[section][key]
    elif (section is None) and (key is not None):
        raise ValueError("key는 입력되었으나 section이 입력되지 않았습니다.")