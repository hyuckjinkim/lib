# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

from lib.python.config import get_config

import boto3

# AWS 키 정보
athena_config = get_config(config_file_name="aws.ini",section="DATA-ATHENA")
AWS_ACCESS_KEY_ID = athena_config["access_key"]
AWS_SECRET_ACCESS_KEY = athena_config["secret_key"]
AWS_REGION = athena_config["region"]

def get_boto3_session() -> boto3.session.Session:
    """Boto3 Session 객체를 얻는 함수
    Returns:
        boto3.session.Session: Boto3 Session
    """
    boto3_session = boto3.Session( 
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_REGION,
    )
    return boto3_session