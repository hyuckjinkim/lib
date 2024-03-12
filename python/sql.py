"""
sql과 관련된 함수와 클래스를 제공한다.

함수 목록
1. `get_sql_query`
    .sql파일을 parsing하여 string으로 가져온다.
"""

# installed libraries
import sqlparse

def get_sql_query(path: str) -> str|list:
    """
    .sql파일을 parsing하여 string으로 가져온다.
    
    Args:
        path (str): .sql 파일의 경로.
        
    Raises:
        path의 파일명이 .sql로 이루어지지 않은 경우.
        
    Return:
        str|list: 해당 경로의 sql쿼리가 하나인 경우 str을, 여러개인 경우에는 list를 반환한다.
    """
    
    file_path = path.split('/')[-1]
    assert file_path.find('.sql')>=0, \
        "file path must be .sql file"

    # Read the content of the SQL file
    with open(path, 'r') as file:
        sql_content = file.read()

    # Parse the content using sqlparse
    parsed_statements = sqlparse.parse(sql_content)

    # Process each parsed statement
    statement_list = []
    for statement in parsed_statements:
        # Format the statement in a readable way with proper indentation and uppercase keywords
        formatted_statement = sqlparse.format(str(statement), reindent=False, keyword_case='upper')
        statement_list.append(formatted_statement)
        
    if len(statement_list)==1:
        return statement_list[0]
    else:
        return statement_list