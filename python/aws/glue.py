""" AWS Glue 관련 함수를 정의한 모듈"""

# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

from lib.python.config import get_config

import boto3
import botocore

# AWS 키 정보
athena_config = get_config(config_file_name="aws.ini", section="ATHENA")
AWS_ACCESS_KEY_ID = athena_config["access_key"]
AWS_SECRET_ACCESS_KEY = athena_config["secret_key"]
AWS_REGION = athena_config["region"]

def get_glue_client() -> botocore.client:
    """Glue Client 객체를 얻는 함수

    Returns:
        botocore.client: Glue Client
    """
    glue_client = boto3.client('glue',
                               aws_access_key_id=AWS_ACCESS_KEY_ID,
                               aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                               region_name=AWS_REGION)
    return glue_client


def start_crawler(crawler_name: str) -> None:
    """Glue Crawler 를 실행하는 함수

    Args:
        crawler_name (str): Glue Crawler 이름

    Returns:
        None
    """
    glue_client = get_glue_client()
    glue_client.start_crawler(Name=crawler_name)
    return None