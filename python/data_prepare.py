import pandas as pd
import numpy as np
from base import color, prt

#---------------------------------------------------------------------------------------------#
# > 설명 : numeric 컬럼을 최적의 타입으로 변경함으로써 data size를 낮추는 함수
# > 참조 : https://www.kaggle.com/code/arjanso/reducing-dataframe-memory-size-by-65
# > 예시 : df, _ = reduce_mem_usage(df,verbose=True)
#---------------------------------------------------------------------------------------------#
def reduce_mem_usage(props,verbose=False):
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

#---------------------------------------------------------------------------------------------#
# > 설명 : column type을 설정하는 class
# > 예시 :
#         type_controller = TypeController(
#             target_feature=CFG.TARGET,
#             cat_features=['hour'],
#             unuse_features=['year','month','day'],
#             segment_feature=CFG.SEGMENT,
#         )
#         type_controller.fit(
#             data=train_df,
#             global_assignment=True,
#             verbose=True,
#         )
#         train_df = type_controller.transform(train_df)
#         test_df  = type_controller.transform(test_df)
#---------------------------------------------------------------------------------------------#
# (!주의) py파일에서 사용하면 적용되지않음
# -> ipynb파일로 들고가서 사용해야함
def global_assignment(dictionary):
    for k,v in dictionary.items():
        exec("globals()['{}']=dictionary['{}']".format(k,k))

class TypeController:
    def __init__(self,target_feature,cat_features=None,unuse_features=None,segment_feature=None):
        assert type(target_feature).__name__ in ['str'], \
            "target_feature must be 'str'"
        assert type(cat_features).__name__ in ['NoneType','list'], \
            "cat_feature must be 'None' or 'list'"
        assert type(unuse_features).__name__ in ['NoneType','list'], \
            "unuse_features must be 'None' or 'list'"
        assert type(segment_feature).__name__ in ['NoneType','str'], \
            "unuse_features must be 'None' or 'str'"
        
        self.target_feature     = target_feature
        self.fixed_cat_features = [] if cat_features    is None else cat_features
        self.unuse_features     = [] if unuse_features  is None else unuse_features
        self.segment_feature    = segment_feature
    
    def _check_dummy(self,data,col):
        return (data[col].nunique()==2) & ((sorted(data[col].unique()) == [0,1]) | (sorted(data[col].unique()) == ['0','1']))
    
    def _check_str(self,data,col):
        try:
            data[col].astype(float)
            dtype = 'float'
        except:
            dtype = 'nan'
        return dtype=='nan'
    
    def get_feature_type(self):
        feature_list = ['target_feature','unuse_features','dummy_features','cat_features','num_features','segment_feature']
        
        feature_dict = {}
        feature_dict['target_feature'] = self.target_feature
        feature_dict['unuse_features'] = self.unuse_features
        feature_dict['dummy_features'] = self.dummy_features
        feature_dict['cat_features'] = self.cat_features
        feature_dict['num_features'] = self.num_features
        feature_dict['segment_feature'] = self.segment_feature

        return feature_dict
    
    def fit(self,data) -> None:
        self.cat_features   = []
        self.dummy_features = []
        self.num_features   = []
        
        for col in data.columns:
            if col==self.target_feature:
                pass
            elif col in self.unuse_features:
                pass
            elif col==self.segment_feature:
                pass
            elif col in self.fixed_cat_features:
                self.cat_features.append(col)
            elif self._check_dummy(data,col):
                self.dummy_features.append(col)
            elif self._check_str(data,col):
                self.cat_features.append(col)
            else:
                self.num_features.append(col)

    def transform(self,data):
        d = data.copy()
        
        # # (1) unuse_features
        # for col in self.unuse_features:
        #     if col in d.columns:
        #         d.drop(col,axis=1,inplace=True)
        
        # (2) segment_feature
        if self.segment_feature is not None:
            d[self.segment_feature] = d[self.segment_feature].astype(str)
        
        # (3) dummy_features
        d[self.dummy_features] = d[self.dummy_features].astype(int)
        
        # (4) cat_features
        d[self.cat_features] = d[self.cat_features].astype(object)
        
        # (5) num_features
        d[self.num_features] = d[self.num_features].astype(float)
        
        return d

#---------------------------------------------------------------------------------------------#
# > 설명 : get korea holidays
# > 예시 : get_holiday(year_list=[2022,2023])
#---------------------------------------------------------------------------------------------#
from pytimekr import pytimekr
def get_holiday(year_list):
    kr_holidays = []
    for year in year_list:
        holidays = pytimekr.holidays(year=year)
        kr_holidays += holidays
    return kr_holidays

#---------------------------------------------------------------------------------------------#
# > 설명 : subset_depth를 설정하여 categorical feature의 조합에 따른 target feature의 quantile 산출
# > 예시 :
#         calculator = CategoricalQuantileCalculator()
#         calculator.fit(
#             data=train_df,
#             test_data=test_df,
#             target_feature=target_feature,
#             cat_features=cat_features,
#             subset_depth=CFG.SUBSET_DEPTH,
#         )
#         train_df2 = calculator.transform(train_df)
#         test_df2  = calculator.transform(test_df)
#---------------------------------------------------------------------------------------------#
import itertools
from itertools import chain, combinations

def all_subsets(ss):
    return list(chain(*map(lambda x: combinations(ss, x), range(0, len(ss)+1))))

def has_test_only_value(train,test,group):
    tr_uniques = ['_'.join(str(x)) for x in train[group].drop_duplicates().values]
    te_uniques = ['_'.join(str(x)) for x in test [group].drop_duplicates().values]
    test_only = list(set(te_uniques)-set(tr_uniques))
    _has_test_only = False if len(test_only)==0 else True
    return _has_test_only

class CategoricalQuantileCalculator:
    def __init__(self,quantiles=[25,50,75],add_avg=True):
        self.quantiles = quantiles
        self.add_avg = add_avg
    
    def _get_quantile(self,x,col):
        x = np.array(x).flatten()
        x = x[pd.notnull(x)]

        agg_df = pd.DataFrame(index=[0])
        if self.add_avg:
            agg_df[f'{col}_Avg'] = np.mean(x)
        for q in self.quantiles:
            agg_df[f'{col}_Q{q}'] = np.quantile(x,q/100)

        return agg_df
    
    # test_data는 test에만 있는 group을 판별 시 사용됨
    # -> 함수 : has_test_only_value(data,test_data,subset)
    def fit(self,data,test_data,target_feature,cat_features=[],subset_depth=1,verbose=True):
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
            if has_test_only_value(data,test_data,subset):
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
        
    def transform(self,data,prefix=''):
        # 카테고리 변수에 따른 가격의 Quantile값
        for key,agg_df in self.agg_dict.items():
            keys = key.split('&')
            if prefix!='':
                rename_dict = {col:f'{prefix}{col}' for col in agg_df.columns if col not in keys}
                agg_df = agg_df.rename(columns=rename_dict)
            data = pd.merge(data,agg_df,how='left',on=keys)
        return data
    
    def fit_transform(self,data,test_data,target_feature,cat_features=[],subset_depth=1,prefix=''):
        self.fit(data,test_data,target_feature,cat_features,subset_depth)
        return self.transform(data,prefix)
    
    
#---------------------------------------------------------------------------------------------#
# > 설명 : 그룹변수 별로 scaling 적용
# > 예시 :
#         scaler = GroupScaler(scaler=MinMaxScaler())
#         scaler.fit(
#             data=train_df,
#             segment_feature=segment_feature,
#             num_features=num_features,
#         )
#         train_df2 = scaler.transform(train_df)
#         test_df2  = scaler.transform(test_df)
#---------------------------------------------------------------------------------------------#
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, QuantileTransformer, PowerTransformer,
)
from copy import deepcopy
from tqdm import tqdm

class GroupScaler:
    def __init__(self,scaler=StandardScaler()):
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

#---------------------------------------------------------------------------------------------#
# > 설명 : delete unique columns
# > 예시 : delete_unique_columns(df,verbose=False)
#---------------------------------------------------------------------------------------------#
def delete_unique_columns(data,verbose=True):
    d = data.copy()
    unique_info = d.apply(lambda x: x.nunique())
    unique_cols = unique_info[unique_info==1].index.tolist()
    if verbose:
        print('> unique_columns: {}'.format(unique_cols))
    return d.drop(columns=unique_cols)


#---------------------------------------------------------------------------------------------#
# > 설명 : onehot encoding
# > 예시 :
#         ohe = OneHotEncoder()
#         ohe.fit(X,fixed_cat_features,remove_first=False)
#         X_oh = ohe.transform(X)
#---------------------------------------------------------------------------------------------#
import pandas as pd
import warnings

class OneHotEncoder:
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

#---------------------------------------------------------------------------------------------#
# > 설명 : create the interaction term
# > 예시 :
#         interaction_maker = InteractionTerm()
#         interaction_maker.fit(
#             data=df,
#             num_features=num_features,
#             corr_cutoff=0.7,
#         )
#         df = interaction_maker.transform(df)
#---------------------------------------------------------------------------------------------#
import warnings
from tqdm import trange

def get_abs_corr(x,y):
    return np.abs(np.corrcoef(x,y))[0,1]

class InteractionTerm:
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
                    if (get_abs_corr(d[col_i]*d[col_j],d[col_i])>=corr_cutoff) | (get_abs_corr(d[col_i]*d[col_j],d[col_j])>=corr_cutoff):
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

#---------------------------------------------------------------------------------------------#
# 설명 : target을 log/sqrt변환
# 예시 : 
#         target_transform = TargetTransform(func='log')
#         transformed = target_transform.fit_transform(
#             target=target,
#             verbose=False,
#         )
#---------------------------------------------------------------------------------------------#
import numpy as np

def identity(x):
    return x

def loglog(x):
    return np.log(np.log(x))

def expexp(x):
    return np.exp(np.exp(x))

class TargetTransform:
    def __init__(self,func='identity',offset=None):
        assert func in ['identity','log','sqrt','loglog'], \
            print("func must be one of ['identity','log','sqrt','loglog']")
        self.func = func
        self.inv_func = {'log':'exp','sqrt':'square'}.get(func,'Unknown')
        self.offset = offset
        
        if self.func=='identity':
            self.transform_fn = identity
            self.inverse_transform_fn = identity
        elif self.func=='loglog':
            self.transform_fn = loglog
            self.inverse_transform_fn = expexp
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

#---------------------------------------------------------------------------------------------#
# > 설명 : Q1,Q3를 기준으로 1.5 IQR보다 멀리있는 값들을 Outlier로 판단하여 제거함
# > 예시 : 
#         outlier_detector = OutlierDetect(
#             target_feature='target',
#             group='segment',
#         )
#         new_data = outlier_detector.fit_transform(
#             data=data,
#             whis=1.5,
#             max_example=4,
#         )
#         outlier_detector.outlier_boundary
#---------------------------------------------------------------------------------------------#
class OutlierDetect:
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

#---------------------------------------------------------------------------------------------#
# > 설명 : 푸리에변환 및 역푸리에변환
# > 한계 : inverse_transform 시, imaginary_part를 가져와야 정확하게 역변환 됨.
#         따라서, 새로운 데이터에 대해서 inverse_transform이 불가능
# > 예시 :
#         ft = FourierTransform()
#         ft.fit(target)
#         fourier_transformed = ft.transform(target)
#         ft.inverse_transform(fourier_transformed)
#---------------------------------------------------------------------------------------------#
import numpy as np
import warnings

class FourierTransform:
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

#---------------------------------------------------------------------------------------------#
# > 설명 : add lag variables
# > 예시 :
#         train, test = AddLagVariable(
#             train_data=train,
#             test_data=test,
#             lag_range=range(1,5+1),
#             bfill=False,
#         )
#---------------------------------------------------------------------------------------------#
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

def AddLagVariable(train_data,test_data,lag_range=range(1,6),bfill=False):
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