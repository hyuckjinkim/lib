import tableauserverclient as TSC
import pandas as pd

from data_lib import config
tableau_config = config.get_config(config_file_name="tableau.ini", section="TableauServer")
TABLEAU_ACCESS_TOKEN_ID = tableau_config["access_key"]
TABLEAU_ACCESS_TOKEN_PW = tableau_config["secret_key"]
SERVER_URL = tableau_config["server_url"]
SITE_ID = tableau_config["site_id"]

def get_server() -> TSC.Server:
    """
    태블로서버 서버 객체를 가져온다.
    
    Returns:
        tableauserverclient.Server: 태블로 서버 객체
    """
    return TSC.Server(SERVER_URL, use_server_version=True)
    
def get_auth() -> TSC.PersonalAccessTokenAuth:
    """
    태블로서버 인증 객체를 가져온다.
    
    Returns:
        tableauserverclient.PersonalAccessTokenAuth: 태블로 인증 객체
    """
    return TSC.PersonalAccessTokenAuth(TABLEAU_ACCESS_TOKEN_ID, TABLEAU_ACCESS_TOKEN_PW, site_id=SITE_ID)

def get_request_options(pagesize: int = 1000) -> TSC.RequestOptions:
    """
    태블로서버 Request Option에 대한 객체를 가져온다.
    
    Args:
        pagesize (int, optional): 태블로서버에서 가져오는 페이지 사이즈. default=1000 (최대 페이지 크기).
    
    Returns:
        tableauserverclient.RequestOptions: 태블로 Request Option에 대한 객체
    """
    return TSC.RequestOptions(pagesize=pagesize)

def get_datasources() -> pd.DataFrame:
    """
    태블로서버에 올라가있는 모든 데이터원본을 가져온다.
    
    Returns:
        pandas.DataFrame: 태블로서버의 모든 데이터원본에 대한 데이터프레임.
                          columns = ['name','id']
                            - name : 데이터원본명
                            - id : 해당 데이터원본에 할당된 id
    """
    
    server = get_server()
    tableau_auth = get_auth()
    request_options = get_request_options()
    with server.auth.sign_in(tableau_auth):
        all_datasources, pagination_item = server.datasources.get(request_options)
    
    sources = []
    for datas in all_datasources:
        sources.append([datas.name, datas.id])

    datasource_data = pd.DataFrame(sources, columns=['name', 'id'])
        
    return datasource_data

def datasource_refresh(source_names: list, verbose: bool = True) -> None:
    """
    태블로 원본데이터 새로고침.
    
    Args:
        source_names (list): 원본데이터 이름들로 작성된 리스트.
        verbose (bool, optional): 출력물을 print 할 것인지 여부.
        
    Returns:
        None
    """
    
    server = get_server()
    tableau_auth = get_auth()
    datasource_data = get_datasources()

    with server.auth.sign_in(tableau_auth):
        target_data = datasource_data[datasource_data.name.isin(source_names)]
        
        if verbose:
            print('target_name :', target_data.name.values)
            print('target_id   :', target_data.id.values)

        for target_id in target_data.id.values :
            data = server.datasources.get_by_id(target_id)
            server.datasources.refresh(data)