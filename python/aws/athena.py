# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

from lib.python.config import get_config
from lib.python.aws import get_boto3_session

import awswrangler as wr
import time
from datetime import timedelta
import boto3

def run_athena_query(query: str, verbose: bool = True):
    """Athena 쿼리를 실행한다.
    
    Args:
        query (str): SQL query.
        verbose (bool, optional): print the output or not.

    Returns:
        pandas.DataFrame
    """

    # time check
    start_time = time.time()

    # connect boto3
    boto3_session = get_boto3_session()

    if verbose:
        print("* Query started !")

    try:
        # (argment참조) https://aws-sdk-pandas.readthedocs.io/en/stable/stubs/awswrangler.athena.read_sql_query.html
        df = wr.athena.read_sql_query(
            query,
            database=None,
            boto3_session=boto3_session,
            ctas_approach=None,
            workgroup="primary",
            keep_files=False,
        )
        if verbose:
            print("* Query ended !")

    except Exception as e:
        print("Error: {}".format(e))
        raise Exception(e)

    finally:
        if verbose:
            end_time = time.time() - start_time
            print('* Query Time : ', str(timedelta(seconds=end_time)))

    return df


if __name__ == "__main__":
    print("AWS Athena Library")
