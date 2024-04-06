"""
데이터를 준비하는 과정과 관련된 함수와 클래스를 제공한다.

인자 목록
1. `global_assignment_text`
    dictionary에 있는 값을 global 값으로 적용한다.

함수 목록
1. `reduce_mem_usage`
    수치형 컬럼을 최적의 타입으로 변경함으로써 데이터프레임의 용량을 낮춘다.
2. `df_to_json`
    입력된 pd.DataFrame을 json 형식의 str로 변환한다.
3. `is_null_case`
    전달된 문자값이 null case인지 확인한다.
4. `strnull_to_null`
    string으로 저장된 데이터에 대해서 string null 값을 np.nan으로 바꾼다.
5. `data_retype`
    데이터의 변수들의 타입을 재설정한다.
6. `all_subsets`
    주어진 리스트로 이루어진 집합에 대한 모든 부분집합을 생성한다.
7. `only_in_test`
    선택한 그룹 컬럼 리스트에 대해서, 테스트 데이터셋에만 존재하는지 여부를 반환한다.
8. `delete_unique_columns`
    주어진 데이터프레임에서 unique한 컬럼을 제거한다.
9. `add_lag_variable`
    lag 값을 추가한다.

클래스 목록.
1. `CategoricalQuantileCalculator`
    카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 산출하는 클래스.
2. `GroupScaler`
    그룹 변수 별로 Scaling을 적용하는 클래스.
3. `OneHotEncoder`
    One Hot Encoding을 제공하는 클래스.
4. `InteractionTerm`
    상호작용항을 추가하는 클래스.
5. `TargetTransform`
    타겟 컬럼에 다음의 변환을 적용한다: `'identity','log','sqrt','loglog'`
6. `OutlierDetect`
    outlier를 판별하여 제거하는 클래스. Q1,Q3를 기준으로 1.5 IQR보다 멀리있는 값들을 Outlier로 판단한다.
7. `FourierTransform`
    푸리에변환 및 역푸리에변환을 제공하는 클래스.
"""

import pandas as pd
import numpy as np
from pytimekr import pytimekr
import json
import itertools
from itertools import chain, combinations
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
)
from copy import deepcopy
from tqdm import tqdm, trange
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

## dictionary에 있는 값을 global 값으로 적용한다.
# (!주의) py파일에서 사용하면 적용되지않음 -> ipynb파일로 들고가서 사용해야함
global_assignment_text = """
def global_assignment(dictionary):
    for k,v in dictionary.items():
        exec("globals()['{}']=dictionary['{}']".format(k,k))
"""

def reduce_mem_usage(props: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    수치형 컬럼을 최적의 타입으로 변경함으로써 데이터프레임의 용량을 낮춘다.
    
    Args:
        props (pd.DataFrame): 데이터프레임.
        verbose (bool, optional): 진행현황을 출력할 것인지 여부. default=True.
        
    Returns:
        pd.DataFrame: 데이터프레임.
    """
    
    # Byte -> MB : 2^20
    asis_mem_usg = props.memory_usage().sum() / (2**20)
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    
    num_cols_loc = [False if (dtype in [object]) or (str(dtype).find('datetime')>=0) else True for dtype in props.dtypes]
    num_cols = props.columns[num_cols_loc]
    i=0
    total=len(num_cols)
    for col in num_cols:
        i+=1
        asis_dtype = props[col].dtype

        # make variables for Int, max and min
        IsInt = False
        mx = props[col].max()
        mn = props[col].min()

        # Integer does not support NA, therefore, NA needs to be filled
        if not np.isfinite(props[col]).all(): 
            NAlist.append(col)
            props[col].fillna(mn-1,inplace=True)  

        # test if column can be converted to an integer
        asint = props[col].fillna(0).astype(np.int64)
        result = (props[col] - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True

        # Make Integer/unsigned Integer datatypes
        to = None
        if IsInt:
            if mn >= 0:
                if mx < 255:
                    to = np.uint8
                elif mx < 65535:
                    to = np.uint16
                elif mx < 4294967295:
                    to = np.uint32
                else:
                    to = np.uint64
            else:
                if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                    to = np.int8
                elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                    to = np.int16
                elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                    to = np.int32
                elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                    to = np.int64
                    
        # Make float datatypes 32 bit
        else:
            to = np.float32
            
        try:
            props[col] = props[col].astype(to)
        except Exception as e:
            if verbose:
                print('convert failed')
                print(e)

        tobe_dtype = props[col].dtype
        if verbose:
            text = '[{}/{}] {}: {} -> {}'.format(
                str(i).zfill(len(str(total))),total,col,asis_dtype,tobe_dtype)
            print(text)
    
    # na가 있으면 min-1로 넣었으므로, 이 값들을 다시 nan으로 변경
    for col in NAlist:
        props[col] = props[col].replace(props[col].min(),np.nan)
        
    tobe_mem_usg = props.memory_usage().sum() / (2**20)
    reduced_mem  = 100*tobe_mem_usg/asis_mem_usg
    
    if verbose:
        print('Memory reduced by {:.2f}% ({:.2f} MB → {:.2f} MB)'.format(reduced_mem,asis_mem_usg,tobe_mem_usg))
        
    return props, (asis_mem_usg, tobe_mem_usg, reduced_mem)

def df_to_json(df: pd.DataFrame) -> str:
    """
    입력된 pd.DataFrame을 json 형식의 str로 변환한다.
    
    Args:
        df (pd.DataFrame): json 형식의 str로 변환 할 데이터프레임.
    
    Returns:
        str: json 형식의 문자열.
    """
    
    df_dict = {}
    for index in df.index:
        df_dict[index] = df.iloc[index,:].to_dict()
    df_json = json.dumps(df_dict, ensure_ascii=False, indent=3)
    return df_json

def is_null_case(x: str):
    """
    전달된 문자값이 null case인지 확인한다.
    
    **Null으로 인식되는 값들**
    1. 다음의 null 객체인 경우 : `pd.NA, None, np.nan`
    2. 대문자 변환 시 다음의 경우 : `'NAN', 'NONE', '<NA>'`
    3. 다음의 모듈로 제공되는 null 객체인 경우 : `pd._libs.missing.NAType`
    
    Args:
        x (str): null case인지 확인 할 객체.
        
    Returns:
        bool: 전달된 문자값이 null case인지 여부.
    """
    
    object_null_cases = [pd.NA, None, np.nan]
    string_upper_null_cases = ['NAN', 'NONE', '<NA>']
    module_null_cases = [pd._libs.missing.NAType]
    
    # (1) object or string_upper null cases
    if len(set([x])-set(object_null_cases))==0:
        return True
    elif x.upper() in string_upper_null_cases:
        return True
    
    # (2) module null cases
    for module_null_case in module_null_cases:
        if isinstance(x,module_null_case):
            return True
    
    # not null case
    return False

def strnull_to_null(data: pd.DataFrame, dropna: bool = True, ignore_columns: str|list = []) -> pd.DataFrame:
    """
    string으로 저장된 데이터에 대해서 string null 값을 np.nan으로 바꾼다.
    
    **Null으로 인식되는 값들**
    1. 다음의 null 객체인 경우 : `pd.NA, None, np.nan`
    2. 대문자 변환 시 다음의 경우 : `'NAN','NONE'`
    3. 다음의 모듈로 제공되는 null 객체인 경우 : `pd._libs.missing.NAType`
    
    Args:
        data (pandas.DataFrame): 데이터프레임.
        dropna (bool, optional): null값들을 제거할지 여부. default=True.
        ignore_columns (str|list, optional): 제외 할 컬럼들. default=[].
        
    Returns:
        pandas.DataFrame: 데이터프레임.
    """
    d = data.copy()
    
    # ignore_columns가 str인 경우 list로 변경
    if isinstance(ignore_columns,str):
        ignore_columns = [ignore_columns]
    
    # null 처리를 할 타겟컬럼 리스트.
    target_columns = list(set(d.columns)-set(ignore_columns))
    
    # str null -> np.nan
    for col in target_columns:
        d[col] = [np.nan if is_null_case(x) else x for x in d[col]]
        
    # dropna
    if dropna:
        d.dropna(inplace=True)
        
    return d

def data_retype(data: pd.DataFrame,
                str_cols: list,
                dummy_cols: list,
                int_cols: list,
                float_cols: list) -> pd.DataFrame:
    """
    데이터의 변수들의 타입을 재설정한다.
    
    Args:
        data (pandas.DataFrame): 체크 할 데이터셋.
        str_cols (list|np.array): string 변수 목록.
        dummy_cols (list|np.array): dummy 변수 목록.
        int_cols (list|np.array): integer 변수 목록.
        float_cols (list|np.array): float 변수 목록.
        
    Raises:
        data의 컬럼과 입력한 컬럼들이 매칭되지 않을 때.
        
    Returns:
        pandas.DataFrame: 변수들의 타입이 재설정된 데이터프레임.
    """
    
    pd.set_option('mode.chained_assignment',None)
    d = data.copy()
    
    # 설정한 컬럼의 개수가 맞는지 확인
    loaded_columns = d.columns
    setting_columns = str_cols + dummy_cols + int_cols + float_cols
    
    check_1 = len(loaded_columns) == len(setting_columns)
    check_2 = len(set(loaded_columns)-set(setting_columns)) == 0
    check_3 = len(set(setting_columns)-set(loaded_columns)) == 0
    assert check_1 & check_2 & check_3, \
        f"The number of columns in loaded data does not match the number of columns in the specified settings. (check_1,check_2,check_3)={check_1,check_2,check_3}"
    
    # # astype(float)와 astype(int)를 비교해서 모두 값이 같으면, int_cols로 할당
    # check_float2int_dict = check_float2int(d,float_cols)
    # float2int_cols = [col for col,not_equal_len in check_float2int_dict.items() if not_equal_len==0]
    # if len(float2int_cols)>0:
    #     float_cols = [col for col in float_cols if col not in float2int_cols]
    #     int_cols = int_cols + float2int_cols
    
    ## check
    # list(set(data.columns)-set(str_cols+dummy_cols+int_cols+float_cols))
    # len(data.columns), len(str_cols+dummy_cols+int_cols+float_cols)

    int_dict   = {key:'integer' for key in dummy_cols+int_cols}
    float_dict = {key:'float' for key in float_cols}
    numeric_dict  = {**int_dict,**float_dict}
    
    for col in str_cols:
        d[col] = d[col].astype('string') # 'string'이 아니라 'str' 또는 str로 하게되면, np.nan이 'nan'으로 변환됨
        
    for col,dtype in numeric_dict.items():
        d[col] = pd.to_numeric(d[col], errors='coerce', downcast=dtype)
    
    return d

def get_holiday(years: list[int]) -> list:
    """
    한국의 휴일인 날짜를 가져온다.
    
    Args:
        years (list[int]): 휴일인 날짜를 가져올 년도.
        
    Returns:
        list: 휴일인 날짜 리스트.
    """
    kr_holidays = []
    for year in years:
        holidays = pytimekr.holidays(year=year)
        kr_holidays += holidays
    return kr_holidays

def all_subsets(element_set: list) -> list:
    """
    주어진 리스트로 이루어진 집합에 대한 모든 부분집합을 생성한다.
    
    Args:
        element_set (list): 집합 리스트.
    
    Returns:
        list: 주어진 리스트로 이루어진 집합에 대한 모든 부분집합.
    """
    return list(chain(*map(lambda x: combinations(element_set, x), range(0, len(element_set)+1))))

def only_in_test(train: pd.DataFrame, test: pd.DataFrame, group: str|list) -> bool:
    """
    선택한 그룹 컬럼 리스트에 대해서, 테스트 데이터셋에만 존재하는지 여부를 반환한다.
    
    Args:
        train (pd.DataFrame): 학습 데이터셋.
        test (pd.DataFrame): 테스트 데이터셋.
        group (str|list): 테스트 데이터셋에만 존재하는지 확인하는 컬럼 리스트.
        
    Returns:
        bool: 테스트 데이터셋에만 존재하는지 여부.
    """
    tr_uniques = ['_'.join(str(x)) for x in train[group].drop_duplicates().values]
    te_uniques = ['_'.join(str(x)) for x in test [group].drop_duplicates().values]
    test_only = list(set(te_uniques)-set(tr_uniques))
    _has_test_only = False if len(test_only)==0 else True
    return _has_test_only

def delete_unique_columns(data: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    주어진 데이터프레임에서 unique한 컬럼을 제거한다.
    
    Args:
        data (pd.DataFrame): 데이터프레임.
        verbose (bool, optional): 진행현황을 출력할 것인지 여부. default=True.
        
    Returns:
        pd.DataFrame: unique한 컬럼이 제거된 데이터프레임.
    """
    d = data.copy()
    unique_info = d.apply(lambda x: x.nunique())
    unique_cols = unique_info[unique_info==1].index.tolist()
    if verbose:
        print('> unique_columns: {}'.format(unique_cols))
    return d.drop(columns=unique_cols)

def add_lag_variable(train_data: pd.DataFrame,
                     test_data: pd.DataFrame,
                     segment_feature: str,
                     num_features: list,
                     lag_range: list = range(1,6),
                     bfill: bool = False):
    """
    lag 값을 추가한다.
    
    Args:
        train_data (pd.DataFrame)
        test_data (pd.DataFrame)
        segment_feature (str)
        num_features (list)
        lag_range (list, optional): default=range(1,6).
        bfill (bool, optional): default=False.
    
    Example:
        ```python
        train, test = add_lag_variable(
            train_data=train,
            test_data=test,
            lag_range=range(1,5+1),
            bfill=False,
        )
        ```
    """
    train = train_data.copy()
    test  = test_data .copy()

    train_list = []
    test_list  = []

    segment_list = train[segment_feature].unique()
    pbar = tqdm(segment_list)
    for segment in pbar:

        tr_d = train[train[segment_feature]==segment].assign(group='train')
        te_d = test [test [segment_feature]==segment].assign(group='test')
        data = pd.concat([tr_d,te_d],axis=0)

        for i,target in enumerate(num_features):
            pbar.set_description('num_features: [{}/{}]'.format(i+1,len(num_features)))

            for k in lag_range:
                lag_feature = f'{target}_lag{k}'
                if bfill:
                    data[lag_feature] = data[target].shift(k).fillna(method='bfill')
                else:
                    data[lag_feature] = data[target].shift(k)
                    data.dropna(subset=[lag_feature],inplace=True)

        train_list.append(data[data['group']=='train'].drop('group',axis=1))
        test_list .append(data[data['group']=='test' ].drop('group',axis=1))

    train = pd.concat(train_list,axis=0).sort_index()
    test  = pd.concat(test_list ,axis=0).sort_index()
    
    if not bfill:
        print('> train: {:,} deleted ({:,} -> {:,})'.format(len(train_data)-len(train),len(train_data),len(train)))
        print('> test : {:,} deleted ({:,} -> {:,})'.format(len(test_data )-len(test ),len(test_data ),len(test )))
    
    return train, test

class CategoricalQuantileCalculator:
    """
    카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 산출하는 클래스.
    
    Example:
        ```python
        calculator = CategoricalQuantileCalculator()
        calculator.fit(
            data=train_df,
            test_data=test_df,
            target_feature='Sales',
            cat_features=['Brand','Company'],
            subset_depth=3,
        )
        train_df2 = calculator.transform(train_df)
        test_df2  = calculator.transform(test_df)
        ```
    """
    
    def __init__(self, quantiles: list = [25,50,75], add_avg: bool = True):
        """
        CategoricalQuantileCalculator의 생성자.
        
        Args:
            quantiles (list, optional): quantile 리스트. default=[25,50,75].
            add_avg (bool, optional): quantile 이외에 평균값을 추가할지 여부. default=True.
        """
        self.quantiles = quantiles
        self.add_avg = add_avg
    
    def _get_quantile(self, x: list, col: str) -> pd.DataFrame:
        """
        주어진 값에 대한 quantile을 구한다.
        
        Args:
            x (list): quantile을 구할 값.
            col (str): 해당 값의 컬럼명.
            
        Returns:
            pd.DataFrame: 주어진 값에 대한 quantile로 이루어진 데이터프레임.
        """
        x = np.array(x).flatten()
        x = x[pd.notnull(x)]

        agg_df = pd.DataFrame(index=[0])
        if self.add_avg:
            agg_df[f'{col}_Avg'] = np.mean(x)
        for q in self.quantiles:
            agg_df[f'{col}_Q{q}'] = np.quantile(x,q/100)

        return agg_df
    
    def fit(self,
            data: pd.DataFrame,
            test_data: pd.DataFrame,
            target_feature: str,
            cat_features: list = [],
            subset_depth: int = 1,
            verbose: bool = True) -> None:
        """
        카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 계산한다.
        
        Args:
            data (pd.DataFrame): quantile을 계산 할 학습 데이터셋.
            test_data (pd.DataFrame): 테스트 데이터셋으로, test에만 있는 subset을 제거 할 때 사용된다.
            target_feature (str): 타겟 컬럼명.
            cat_features (list, optional): 카테고리형 컬럼 리스트. default=[].
            subset_depth (int, optional): 카테고리형 컬럼의 조합의 최대 depth. default=1.
            verbose (bool, optional): 진행현황을 출력할 것인지 여부. default=True.
        
        Raises:
            카테고리형 컬럼 리스트의 개수가 subset_depth보다 같거나 큰 경우.
        
        Returns:
            None.
        """
        
        assert len(cat_features)>=subset_depth, \
            'len(cat_features) >= subset_depth'
        
        self.data           = data
        self.test_data      = test_data
        self.target_feature = target_feature
        self.cat_features   = cat_features
        self.subset_depth   = subset_depth
        self.verbose        = verbose
        
        # 카테고리 변수에 따른 가격의 Quantile값
        all_subset_list = all_subsets(self.cat_features)
        all_subset_list = [subset for subset in all_subset_list if (len(subset)<=subset_depth) & (len(subset)>=1)]
        if verbose:
            print(f'> Get quantiles of target by categorical features (depth={subset_depth})')
            pbar = tqdm(all_subset_list)
        else:
            pbar = all_subset_list
        
        self.agg_dict = {}
        for subset in pbar:
            subset = list(subset)
            if only_in_test(data,test_data,subset):
                pass
            else:
                if verbose:
                    pbar.set_description('Subset: {}'.format(' + '.join(subset)))
                subset_name = '&'.join(subset)
                agg_fn = data\
                    .groupby(subset)[self.target_feature]\
                    .apply(lambda x: self._get_quantile(x,subset_name))\
                    .reset_index()
                drop_cols = [col for col in agg_fn if col.find('level_')>=0]
                agg_fn.drop(columns=drop_cols,inplace=True)
                if agg_fn.isnull().sum().sum()>0:
                    if verbose:
                        print('> Null Detectd: {} -> Passed'.format('+'.join(subset)))
                else:
                    self.agg_dict[subset_name] = agg_fn
        
    def transform(self, data: pd.DataFrame, prefix: str = '') -> pd.DataFrame:
        """
        fit()을 통해 계산된 카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 data에 합쳐준다.
        
        Args:
            data (pd.DataFrame): 카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 합쳐줄 데이터프레임.
            prefix (str, optional): 컬럼명의 prefix. default=''.
            
        Returns:
            pd.DataFrame: 카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile이 합쳐진 데이터프레임.
        """
        # 카테고리 변수에 따른 가격의 Quantile값
        for key,agg_df in self.agg_dict.items():
            keys = key.split('&')
            if prefix!='':
                rename_dict = {col:f'{prefix}{col}' for col in agg_df.columns if col not in keys}
                agg_df = agg_df.rename(columns=rename_dict)
            data = pd.merge(data,agg_df,how='left',on=keys)
        return data
    
    def fit_transform(self,
                      data: pd.DataFrame,
                      test_data: pd.DataFrame,
                      target_feature: str,
                      cat_features: list = [],
                      subset_depth: int = 1,
                      verbose: bool = True,
                      prefix: str = '') -> None:
        """
        카테고리형 컬럼의 조합에 따른 타겟 컬럼의 quantile을 계산하여 data에 합쳐준다.
        
        Args:
            data (pd.DataFrame): quantile을 계산 할 학습 데이터셋.
            test_data (pd.DataFrame): 테스트 데이터셋으로, test에만 있는 subset을 제거 할 때 사용된다.
            target_feature (str): 타겟 컬럼명.
            cat_features (list, optional): 카테고리형 컬럼 리스트. default=[].
            subset_depth (int, optional): 카테고리형 컬럼의 조합의 최대 depth. default=1.
            verbose (bool, optional): 진행현황을 출력할 것인지 여부. default=True.
            prefix (str, optional): 컬럼명의 prefix. default=''.
        
        Raises:
            카테고리형 컬럼 리스트의 개수가 subset_depth보다 같거나 큰 경우.
        
        Returns:
            None.
        """
        self.fit(data,test_data,target_feature,cat_features,subset_depth)
        return self.transform(data,prefix)

class GroupScaler:
    """
    
    
    Example:
        ```python
        scaler = GroupScaler(scaler=MinMaxScaler())
        scaler.fit(
            data=train_df,
            segment_feature=segment_feature,
            num_features=num_features,
        )
        train_df2 = scaler.transform(train_df)
        test_df2  = scaler.transform(test_df)
        ```
    """
    
    def __init__(self, scaler=StandardScaler()):
        self.scaler = scaler
        self.enable_scalers = [
            StandardScaler, MinMaxScaler, RobustScaler, 
            MaxAbsScaler, QuantileTransformer, PowerTransformer,
        ]
        assert any(isinstance(scaler,s) for s in self.enable_scalers), \
            "scaler must be one of scalers in sklearn.preprocessing"
    
    def fit(self,data,group,num_features):
        not_num_features = [dtype for dtype in data[num_features].dtypes if dtype not in [int,float]]
        assert len(not_num_features)==0, \
            "not numerical features: {}".format(not_num_features)
        if len(group)==1:
            group = group[0]
            
        self.group = group
        self.num_features = num_features
        
        self.scalers = {}
        self.group_size = len(data[group].drop_duplicates())
        
        pbar = tqdm(data.groupby(self.group),total=self.group_size)
        for i,(grp,d) in enumerate(pbar):
            grp_str = '_'.join([str(g) for g in grp])
            self.scalers[grp_str] = {}
            for feature in self.num_features:
                pbar.set_description('[Fit]')
                #pbar.set_description('[Fit] Group: {}({}/{})'.format(grp,i+1,self.group_size))
                scaler = deepcopy(self.scaler)
                scaler.fit(np.array(d[feature]).reshape(-1,1))
                self.scalers[grp_str][feature] = scaler
                
    def transform(self,data):
        data_list = []
        pbar = tqdm(data.groupby(self.group),total=self.group_size)
        for i,(grp,d) in enumerate(pbar):
            grp_str = '_'.join([str(g) for g in grp])
            for feature in self.num_features:
                pbar.set_description('[Transform]')
                #pbar.set_description('[Transform] Group: {}({}/{})'.format(grp,i+1,self.group_size))
                if feature in data.columns:
                    d[feature] = self.scalers[grp_str][feature].transform(np.array(d[feature]).reshape(-1,1))
            data_list.append(d)
        transform_data = pd.concat(data_list,axis=0)
        transform_data = transform_data.loc[data.index]
        return transform_data
    
    def fit_transform(self,data,group,num_features):
        self.fit(data,group,num_features)
        return self.transform(data)
    
    def inverse_transform(self,data,num_features):
        data_list = []
        pbar = data.groupby(self.group)
        for i,(grp,d) in enumerate(pbar):
            grp_str = '_'.join([str(g) for g in grp])
            for feature in self.num_features:
                #pbar.set_description('[Inverse Transform]')
                #pbar.set_description('[Transform] Group: {}({}/{})'.format(grp,i+1,self.group_size))
                if feature in data.columns:
                    d[feature] = self.scalers[grp_str][feature].inverse_transform(np.array(d[feature]).reshape(-1,1))
            data_list.append(d)
        inv_transform_data = pd.concat(data_list,axis=0)
        inv_transform_data = inv_transform_data.loc[data.index]
        return inv_transform_data

class OneHotEncoder:
    """
    One Hot Encoding을 제공하는 클래스.
    
    Example:
        ```python
        ohe = OneHotEncoder()
        ohe.fit(X, fixed_cat_features, remove_first=False)
        X_oh = ohe.transform(X)
        ```
    """
    def __init__(self):
        pass
    
    def fit(self,data,columns,remove_first=True):
        self.transform_list = []
        self.remove_first = remove_first
        for col in columns:
            try:
                value_list = sorted(data[col].unique())
            except:
                value_list = data[col].unique()
            for i,value in enumerate(value_list):
                if (i==0) & (remove_first):
                    pass
                else:
                    self.transform_list.append([col,value])
        
    def transform(self,data):
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        new_data = data.copy()
        for col,value in self.transform_list:
            new_data[f'{col}_{value}'] = np.where(new_data[col]==value,1,0)
        drop_columns = pd.unique(np.array(self.transform_list)[:,0])
        new_data.drop(columns=drop_columns,inplace=True)
        return new_data
    
    def fit_transform(self,data,columns,remove_first=True):
        self.fit(data,columns,remove_first)
        return self.transform(data)

def _get_abs_corr(x,y):
    return np.abs(np.corrcoef(x,y))[0,1]

class InteractionTerm:
    """
    상호작용항을 추가하는 클래스.
    
    Example:
        ```python
        interaction_maker = InteractionTerm()
        interaction_maker.fit(
            data=df,
            num_features=num_features,
            corr_cutoff=0.7,
        )
        df = interaction_maker.transform(df)
        ```
    """
    
    def __init__(self):
        pass
    
    def fit(self,data,num_features,corr_cutoff=0.7):
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        
        d = data.copy()
        self.interaction_list = []
        for i in trange(len(num_features),desc='fitting...'):
            for j in range(len(num_features)):
                if i>j:
                    col_i = num_features[i]
                    col_j = num_features[j]
                    
                    # 상관계수가 cutoff보다 큰 경우에는 interaction을 생성하지 않음
                    if (_get_abs_corr(d[col_i]*d[col_j],d[col_i])>=corr_cutoff) | (_get_abs_corr(d[col_i]*d[col_j],d[col_j])>=corr_cutoff):
                        pass
                    else:
                        self.interaction_list.append(f'{col_i}*{col_j}')
    
    def transform(self,data):
        d = data.copy()
        print('> the number of interaction term:',len(self.interaction_list))
        for interaction in self.interaction_list:
            col_i,col_j = interaction.split('*')
            d[interaction] = d[col_i]*d[col_j]
        return d
    
    def fit_transform(self,data,num_features,corr_cutoff=0.7):
        self.fit(data,num_features,corr_cutoff)
        return self.transform(data)

def _identity(x):
    return x

def _loglog(x):
    return np.log(np.log(x))

def _expexp(x):
    return np.exp(np.exp(x))

class TargetTransform:
    """
    타겟 컬럼에 다음의 변환을 적용한다: `'identity','log','sqrt','loglog'`
    
    Example:
        ```python
        target_transform = TargetTransform(func='log')
        transformed = target_transform.fit_transform(target=target, verbose=False)
        ```
    """
    
    def __init__(self,func='identity',offset=None):
        assert func in ['identity','log','sqrt','loglog'], \
            print("func must be one of ['identity','log','sqrt','loglog']")
        self.func = func
        self.inv_func = {'log':'exp','sqrt':'square'}.get(func,'Unknown')
        self.offset = offset
        
        if self.func=='identity':
            self.transform_fn = _identity
            self.inverse_transform_fn = _identity
        elif self.func=='loglog':
            self.transform_fn = _loglog
            self.inverse_transform_fn = _expexp
        else:
            self.transform_fn = eval('np.{}'.format(self.func))
            self.inverse_transform_fn = eval('np.{}'.format(self.inv_func))
        
    def _get_offset(self,x):
        x = np.array(x)
        if min(x)>0:
            offset = 0
        elif min(x)==0:
            offset = 1e-3
        else:
            offset = -min(x)+1e-3
        return offset
        
    def fit(self,target):
        target = np.array(target)
        is_2d_array = len(target.shape) == 1
        if is_2d_array==1:
            target = target.reshape(-1,1)
            
        if self.func=='identity':
            self.offsets = np.zeros(target.shape[1])
        else:   
            self.offsets = []
            for i in range(target.shape[1]):
                x = np.array(target)[:,i]
                if self.offset is None:
                    offset = self._get_offset(x)
                else:
                    offset = self.offset
                self.offsets.append(offset)
    
    def transform(self,target):
        target = np.array(target)
        is_2d_array = len(target.shape) == 1
        if is_2d_array==1:
            target = target.reshape(-1,1)
            
        res = []
        for i in range(target.shape[1]):
            x = np.array(target)[:,i]
            x = self.transform_fn(x+self.offsets[i])
            res.append(x)
            
        return np.array(res).T
    
    def fit_transform(self,target):
        self.fit(target)
        return self.transform(target)
    
    def inverse_transform(self,target):
        target = np.array(target)
        is_2d_array = len(target.shape) == 1
        if is_2d_array==1:
            target = target.reshape(-1,1)
        
        res = []
        for i in range(target.shape[1]):
            x = np.array(target)[:,i]
            x = self.inverse_transform_fn(x) - self.offsets[i]
            res.append(x)
        return np.array(res).T

class OutlierDetect:
    """
    outlier를 판별하여 제거하는 클래스. Q1,Q3를 기준으로 1.5 IQR보다 멀리있는 값들을 Outlier로 판단한다.
    
    Example:
        ```python
        outlier_detector = OutlierDetect(target_feature='target', group='segment')
        new_data = outlier_detector.fit_transform(data=data, whis=1.5, max_example=4)
        outlier_detector.outlier_boundary
        ```
    """
    
    def __init__(self,target_feature,group=None):
        self.target_feature = target_feature
        self.group = group
    
    def _get_outlier(self,data,target_feature,whis):
        target = data[target_feature]
        q1,q2,q3 = target.quantile([0.25,0.50,0.75]).values
        outlier_lower = q1 - whis*(q3-q1)
        outlier_upper = q3 + whis*(q3-q1)
        outlier_boundary = [
            outlier_lower,
            q1,
            q2,
            q3,
            outlier_upper,
            np.where((target<outlier_lower)|(target>outlier_upper),1,0).sum(),
            np.where((target<outlier_lower)|(target>outlier_upper),1,0).sum() / len(target),
            len(target),
        ]
        return outlier_boundary
    
    def fit(self,data,whis=1.5):
        assert self.target_feature in data.columns, \
            "No {} column in the data".format(self.target_feature)
        self.data = data
        
        if self.group is None:
            outlier_boundary = self._get_outlier(self.data,self.target_feature,whis)
            self.outlier_boundary = pd.DataFrame(
                [outlier_boundary],
                columns=['outlier_lower','Q1','Q2','Q3','outlier_upper','n_outlier','p_outlier','n_total'],
            )
        else:
            self.group_list = self.data[self.group].unique()
            outlier_boundary_list = []
            for group in self.group_list:
                d = self.data[self.data[self.group]==group]
                outlier_boundary = [group]+self._get_outlier(d,self.target_feature,whis)
                outlier_boundary_list.append(outlier_boundary)
            self.outlier_boundary = pd.DataFrame(
                outlier_boundary_list,
                columns=['group','outlier_lower','Q1','Q2','Q3','outlier_upper',
                         'n_outlier','p_outlier','n_total'],
            )
    
    def transform(self,data,max_example=4,verbose=True):
        example = []
        if self.group is None:
            if len(self.outlier_boundary)!=1:
                raise ValueError('length of self.outlier_boundary_df must be 1')
            else:
                d = data.copy()
                outlier_lower = self.outlier_boundary.outlier_lower.values[0]
                outlier_upper = self.outlier_boundary.outlier_upper.values[0]
                outlier_in = (d[self.target_feature]>=outlier_lower)&(d[self.target_feature]<=outlier_upper)
                new_data = d[outlier_in]
                ex = d[self.target_feature][~outlier_in]
                if len(ex)>0:
                    ex = sorted([round(e,3) for e in ex])
                    if len(ex)>max_example:
                        ex = ex[:2]+['...']+ex[-2:]
                else:
                    ex = ''
                example.append(ex)
        else:
            check_1 = list(set(self.group_list)-set(data[self.group].unique()))
            check_2 = list(set(data[self.group].unique())-set(self.group_list))
            if (len(check_1)==0) & (len(check_2)==0):
                pass
            elif (len(check_1)>0) & (len(check_2)==0):
                pass
            else:
                raise ValueError('Unknown group values')
            
            new_data = []
            for group in self.group_list:
                d = data[data[self.group]==group]
                outlier_d = self.outlier_boundary[self.outlier_boundary['group']==group]
                if len(outlier_d)!=1:
                    raise ValueError('length of self.outlier_boundary_df must be 1')
                else:
                    outlier_lower = outlier_d.outlier_lower.values[0]
                    outlier_upper = outlier_d.outlier_upper.values[0]
                    outlier_out = (d[self.target_feature]>=outlier_lower)&(d[self.target_feature]<=outlier_upper)
                    new_d = d[outlier_out]
                    new_data.append(new_d)
                    ex = d[self.target_feature][~outlier_out]
                    if len(ex)>0:
                        ex = sorted([round(e,3) for e in ex])
                        if len(ex)>max_example:
                            ex = ex[:2]+['...']+ex[-2:]
                    else:
                        ex = ''
                    example.append(ex)
            new_data = pd.concat(new_data,axis=0)
        
        self.outlier_boundary['example'] = example
        self.example = example
        
        if verbose:
            print("> {:,}'s outliers deleted ({:,}->{:,})".format(len(data)-len(new_data),len(data),len(new_data)))
            
        return new_data
    
    def fit_transform(self,data,whis=1.5,max_example=4,verbose=True):
        self.fit(data,whis)
        return self.transform(data=data,verbose=verbose)

class FourierTransform:
    """
    푸리에변환 및 역푸리에변환을 제공하는 클래스.
    
    Limitation:
        inverse_transform 시, imaginary_part를 가져와야 정확하게 역변환 됨. 따라서, 새로운 데이터에 대해서 inverse_transform이 불가능함.
        
    Example:
        ```python
        ft = FourierTransform()
        ft.fit(target)
        fourier_transformed = ft.transform(target)
        ft.inverse_transform(fourier_transformed)
        ```
    """
    
    def fit(self,x):
        x = np.array(x)
        y = np.fft.fft(x) / len(x)
        self.real_part      = np.array([np.real(_) for _ in y])
        self.imaginary_part = np.array([_-np.real(_) for _ in y])
    
    def transform(self,x):
        x = np.array(x)
        y = np.fft.fft(x) / len(x)
        y = np.array([np.real(_) for _ in y])
        return y
    def inverse_transform(self,x):
        x = np.array(x)
        x = x + self.imaginary_part
        y = np.fft.ifft(x) * len(x)
        y = np.array([np.real(_) for _ in y])
        return y