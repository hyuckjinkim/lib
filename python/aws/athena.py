import pytz
import awswrangler as wr
import datetime
import time
import boto3

# # 예시
# query = """
# SELECT column1, column2
# FROM database
# ;
# """

# df = run_athena_query(
#     query=query,
#     database=None, 
#     s3_output=None,
#     verbose=True,
# )
# df = df.apply(lambda x: np.array(x))

def run_athena_query(
    query,
    database,
    s3_output=None,
    boto3_session=None,
    categories=None,
    chunksize=None,
    ctas_approach=None,
    profile=None,
    workgroup="primary",
    region_name=region_name, #"ap-northeast-2"
    keep_files=False,
    max_cache_seconds=0,
    verbose=False,
):
    
    """
    Args:
        - query: SQL query.
        - database (str): AWS Glue/Athena database name - It is only the original database from where the query will be launched. You can still using and mixing several databases writing the full table name within the sql (e.g. database.table).
        - ctas_approach (bool): Wraps the query using a CTAS, and read the resulted parquet data on S3. If false, read the regular CSV on S3.
        - categories (List[str], optional): List of columns names that should be returned as pandas.Categorical. Recommended for memory restricted environments.
        - chunksize (Union[int, bool], optional): If passed will split the data in a Iterable of DataFrames (Memory friendly). If True wrangler will iterate on the data by files in the most efficient way without guarantee of chunksize. If an INTEGER is passed Wrangler will iterate on the data by number of rows igual the received INTEGER.
        - s3_output (str, optional): Amazon S3 path.
        - workgroup (str, optional): Athena workgroup.
        - keep_files (bool): Should Wrangler delete or keep the staging files produced by Athena? default is False
        - profile (str, optional): aws account profile. if boto3_session profile will be ignored.
        - boto3_session (boto3.Session(), optional): Boto3 Session. The default boto3 session will be used if boto3_session receive None. if profilename is provided a session will automatically be created.
        - max_cache_seconds (int): Wrangler can look up in Athena’s history if this query has been run before. If so, and its completion time is less than max_cache_seconds before now, wrangler skips query execution and just returns the same results as last time. If reading cached data fails for any reason, execution falls back to the usual query run path. by default is = 0
    Returns:
        - Pandas DataFrame
    """
    
    # 함수 시작/종료시간 display -> 한국시간으로 표시
    KST = pytz.timezone('Asia/Seoul')
    
    # 함수 시작시간
    start_time = str(datetime.datetime.now(KST))[:19]
    
    # test for boto3 session and profile.
    if (boto3_session == None) & (profile != None):
        boto3_session = boto3.Session(profile_name=profile, region_name=region_name)

    if verbose:
        print("> Query Job Start...!")

    try:
        df = wr.athena.read_sql_query(
            query,
            database=database,
            boto3_session=boto3_session,
            categories=categories,
            chunksize=chunksize,
            ctas_approach=ctas_approach,
            s3_output=s3_output,
            workgroup=workgroup,
            keep_files=keep_files,
            max_cache_seconds=max_cache_seconds,
        )
        if verbose:
            print("> Query Job Success...!")

    except Exception as e:
        print("Error: {}".format(e))
        raise Exception(e)
    
    finally:
        # 함수 종료시간
        end_time = str(datetime.datetime.now(KST))[:19]

        fmt = '%Y-%m-%d %H:%M:%S'
        run_time = datetime.datetime.strptime(end_time,fmt) - datetime.datetime.strptime(start_time,fmt)

        if verbose:
            print(f'> Query Job Info.')
            print(f'  -   Shape : NROW({df.shape[0]:,}), NCOL({df.shape[1]:,})')
            print(f'  -   Start : {start_time}')
            print(f'  -     End : {end_time}')
            print(f'  - Running : {run_time}')
        
    return df