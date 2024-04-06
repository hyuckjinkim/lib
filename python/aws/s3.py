""" AWS S3 관련 함수들을 모아놓은 모듈"""

# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

from lib.python.config import get_config

from io import BytesIO

import boto3
import pandas as pd
import pickle
# import s3fs

athena_config = get_config(config_file_name="aws.ini", section="ATHENA")
AWS_ACCESS_KEY_ID = athena_config["access_key"]
AWS_SECRET_ACCESS_KEY = athena_config["secret_key"]

def get_s3_client() -> boto3.client:
    """S3 Client 객체를 얻는 함수

    Returns:
        botocore.client: S3 Client
    """
    s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY_ID, aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
    return s3_client

def get_object_list(bucket: str, key: str, start_after_key: str = '', exclude_str: str = '', return_current_key: bool = True) -> list[dict]:
    """object 리스트를 불러온다.

    **필터 옵션 설명**

    s3 `abc` 버킷에 `test` key 하위에 다음과 같은 key가 있을 경우

    - `test1.csv`
    - `test2.csv`
    - `test3.csv`
    - `test4.csv`

    아래와 같이 파라미터 입력

    - `bucket='abc'`
    - `key='test/te'`
    - `start_after_key='abc/test1.csv'`
    - `exclude_str='3'`

    결과

    - test2.csv
    - test4.csv

    Args:
        bucket (str): s3 버켓
        key (str): 조회할 key, startwith 필터링 가능
        start_after_key (str): 조회 범위 시작 필터 key. 조회된 key 들 중 필터 from을 지정한다. bucket명을 제외한 key full path 기입해야하며, 이 파라미터값 이후의 key부터 필터링된다.
        exclude_str: 제외할 key를 지정하는 문자열. 이 파라미터에 지정한 값은 delimiter로 분류되면서 조회되는 key에서는 제외된다.
        return_current_key (bool): 현재 key Contents 출력 여부. False일 경우 하위 key에 대한 Contents만 출력된다.

    Raises:
        Exception: 경로가 없다고 판단되면 에러 발생

    Returns:
        list[dict]: object list
    """
    s3_client = get_s3_client()

    kwargs = {
        'Bucket' : bucket,
        'Prefix' : key,
        'StartAfter' : start_after_key,
        'Delimiter' : exclude_str,
    }

    try:
        contents = []
        obj_list = s3_client.list_objects_v2(**kwargs)
        contents = obj_list["Contents"]

        # 파일이 1,000개가 넘으면 1,000개까지만 가져와짐 -> ContinuationToken을 통해서 가져와야함
        while obj_list['IsTruncated']:
            obj_list = s3_client.list_objects_v2(**kwargs, ContinuationToken=obj_list['NextContinuationToken'])
            contents += obj_list["Contents"] 
        
        # 파티션 key가 입력되었을 때, 파티션 key Contents 출력 여부 
        if not return_current_key:
            child_key_contents = [] 
            for content in contents:
                if content['Key'] != f'{key}/':
                    child_key_contents.append(content)
            return child_key_contents
        return contents
    except KeyError:
        raise KeyError(f"Check the S3 path '{bucket}/{key}' and filter is correct.")
    except Exception as e:
        raise Exception(f"ERROR - {e}")

def get_str_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    데이터프레임의 모든 컬럼의 타입을 str으로 변경

    Args:
        dataframe (pd.DataFrame): target dataframe

    Returns:
        pd.DataFrame: str 로 변경된 데이터프레임
    """
    
    # pandas chained assignment가 이루어지지 않도록 copy 생성
    df = dataframe.copy()
        
    for column in df.columns:
        df[column] = df[column].astype(str)
    
    return df

def exist_key(s3_bucket: str, s3_prefix_key: str) -> bool:
    """입력된 s3_prefix_key에 파일이 존재하는지 여부를 반환한다.

    Args:
        s3_bucket (str): s3 버켓
        s3_prefix_key (str): prefix 가 포함된 Key
    
    Raises:
        get_object_list에서 KeyError 이외의 에러가 발생 한 경우.
    
    Return:
        bool: 파일이 존재하는지 여부.
    """
    
    try:
        get_object_list(s3_bucket, s3_prefix_key)
        return True
    except KeyError as e:
        return False
    except Exception as e:
        raise Exception(f"ERROR - {e}")

def upload_object(input_object: object,
                  s3_bucket: str,
                  s3_prefix_key: str,
                  overwrite: bool = True,
                  convert_to_str: bool = False) -> None:
    """파일을 s3에 업로드한다.

    Args:
        input_object (object): target object.
        s3_bucket (str): s3 버켓.
        s3_prefix_key (str): prefix 가 포함된 Key.
        overwrite (bool, optional): 덮어쓰기 여부. default=True.
        convert_to_str (bool, optional): 데이터프레임의 모든 컬럼의 타입을 str으로 변경할지 여부. default=False.
        
    Return:
        None
    """

    # 확장자명
    file_format = s3_prefix_key.split('.')[-1]
    
    # s3 client를 가져온다.
    s3_client = get_s3_client()
    
    # input이 DataFrame인 경우, string type으로 converting
    if convert_to_str:
        input_object = get_str_dataframe(input_object)
    
    # (1) 파일이 존재하는데 덮어쓰기를 하지 않는 경우에는 저장을 하지않는다.
    if (not overwrite) and exist_key(s3_bucket, s3_prefix_key):
        print(f"Upload failed! '{s3_bucket}/{s3_prefix_key}' already exists.")
        return None
        
    # (2) 확장자명이 parquet 또는 csv인 경우.
    elif file_format in ['parquet','csv']:
        out_buffer = BytesIO()
        if file_format=='parquet':
            input_object.to_parquet(out_buffer, index=False)
        elif file_format=='csv':
            input_object.to_csv(out_buffer, index=False)
        body = out_buffer.getvalue()
    
    # (3) 확장자명이 pickle인 경우.
    elif file_format in ['pickle','pkl']:
        body = pickle.dumps(input_object)
        
    # (4) 확장자명이 json인 경우.
    elif file_format in ["json"]:
        body = input_object
    
    s3_client.put_object(Bucket=s3_bucket, Key=s3_prefix_key, Body=body)
        
    return None

def read_object(s3_bucket: str, s3_prefix_key: str) -> list[dict] | object:
    """S3의 파일을 가져온다.
    
    Args:
        s3_bucket (str): s3 버켓.
        s3_prefix_key (str): prefix 가 포함된 Key.
        
    Returns:
        parquet,csv인 경우 dict[list]를, pickle,pkl인 경우 object를 가져온다.
        
        example.
        
        dict[list]: ``[{key: value}, {key: value}, {key: value}, ... , {key: value}]``
        object: ``해당 파일에 해당하는 class 등의 객체``
    """

    # 확장자명
    file_format = s3_prefix_key.split('.')[-1]
    
    # s3 client로 부터 object를 가져온다.
    s3_client = get_s3_client()
    obj = s3_client.get_object(Bucket=s3_bucket, Key=s3_prefix_key)
    obj_body = obj["Body"].read()

    # (1) 확장자명이 parquet 또는 csv인 경우.
    if file_format in ['parquet','csv']:
        byte = BytesIO(obj_body)
        if file_format=='parquet':
            data = pd.read_parquet(byte)
        elif file_format=='csv':
            data = pd.read_csv(byte)
        data = data.to_dict("records")

    # (2) 확장자명이 pickle인 경우.
    elif file_format in ['pickle','pkl']:
        data = pickle.loads(obj_body)
        
    # (3) 확장자명이 json인 경우.
    elif file_format in ["json"]:
        data = obj_body.decode()
    
    return data

# def s3_glob(path: str) -> list[str]:
#     """glob-matching으로 S3내에 있는 파일을 찾는다.
    
#     Args:
#         path (str): S3 경로 (glob 형식).
    
#     Returns:
#         list[str]: glob-matching으로 얻어진 S3 경로
        
#     Reference:
#         s3fs.S3FileSystem 참조: https://s3fs.readthedocs.io/en/latest/api.html#s3fs.core.S3FileSystem.glob
#     """
    
#     prefix = 's3://'
    
#     s3 = s3fs.S3FileSystem(anon=False, key=AWS_ACCESS_KEY_ID, secret=AWS_SECRET_ACCESS_KEY)
#     obj_list = s3.glob(path)
#     obj_list = [prefix+obj for obj in obj_list]
    
#     return obj_list