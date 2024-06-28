import pandas as pd

def set_display_option(n: int = None) -> None:
    """
    pd.set_option을 통해 display 시 보여지는 rows, columns의 maximum 값을 세팅한다.
    
    Args:
        n (int):display 시 보여지는 rows, columns의 maximum 값. default=None.
        
    Returns:
        None.
    """
    pd.set_option('display.max_rows', n)
    pd.set_option('display.max_columns', n)
    pd.set_option('display.width', n)