#------------------------------------------------------------------------------------------------------------#
# [1] 키워드 오적재 케이스를 파악하는 함수
#------------------------------------------------------------------------------------------------------------#
def keyword_error_case(keyword):

    # 특수문자, only number 제거
    def is_special_char(k):
        # [1] 특수문자가 포함된 경우
        k_1 = re.sub('[^ㄱ-ㅎㅣ^가-힣+|^a-zA-Z0-9|^ |^-|^.|^%|^MLT-|^CLT-]','',k) # 정상 케이스 빼고 제거
        k_1 = re.sub('\^','',k_1)               # ^ 제외
        k_1 = re.sub('\|','',k_1)               # | 제외
        k_1 = re.sub('[ㄱ-ㅎ|ㅏ-ㅣ]','',k_1)      # 자음만 있는 경우 제외
        
        return k_1!=k
    
    def is_only_number(k):
        # [2] 숫자 + 대쉬(-,_)만 포함된 경우
        k_2 = re.sub(r'[0-9|-|_]','',k)
        
        return k_2==''
    
    keyword = [str(x) for x in keyword]
    
    _case = [
        # (1) nvMid가 포함된 경우 : 값이 밀려들어가거나 해서 잘못 적재된 걸로 예상됨
        'case1' if x.find('nvMid')>=0 else
        
        # (2) \t 값으로 들어간 경우
        'case2' if (x=='\t') else
        
        # (3) dict 형태로 들어가있는 경우 (2,3번째는 키보드 자판으로 입력이 안됨(특수문자로 보임), 복붙해야 인식됨)
        'case3' if (x.find("': '")>=0) or (x.find("': F")>=0) or (x.find("': T")>=0) else
        
        # (4) 따옴표만 들어가있는 경우
        'case4' if (x=='""') or (x=='"') or (x=="''") or (x=="'") else
        
        # (5) ^가 들어간 경우
        'case5' if x.find('^')>=0 else
        
        # (6) `/`가 들어간 경우
        'case6' if x.find("`/`")>=0 else
        
        # (7) |가 들어간 경우 (/가 포함된 경우 제외 -> case10에서 가져옴)
        'case7' if (x.find('|')>=0) & (x.find('/')<0) else
        
        # (8) 숫자/가 들어간 경우
        'case8' if x.find('0/')>=0 or x.find('1/')>=0 or x.find('2/')>=0 or x.find('3/')>=0 or x.find('4/')>=0 or\
                   x.find('5/')>=0 or x.find('6/')>=0 or x.find('7/')>=0 or x.find('8/')>=0 or x.find('9/')>=0 else
        
        # (9) 숫자_가 들어간 경우
        'case9' if x.find('0_')>=0 or x.find('1_')>=0 or x.find('2_')>=0 or x.find('3_')>=0 or x.find('4_')>=0 or\
                   x.find('5_')>=0 or x.find('6_')>=0 or x.find('7_')>=0 or x.find('8_')>=0 or x.find('9_')>=0 else
        
        # (10) /가 들어간 경우
        'case10' if x.find('/')>=0 else
        
        # (11) 0.0, 1.0, ..., 10.0인 경우
        'case11' if (x=='0.0') or (x=='1.0') or (x=='2.0') or (x=='3.0') or (x=='4.0') or\
                    (x=='5.0') or (x=='6.0') or (x=='7.0') or (x=='8.0') or (x=='9.0') else
        
        # (12) 이외의 특수문자
        'case12' if is_special_char(x) else
        
        # (13) only number
        'case13' if is_only_number(x) else
        
        # 정상으로 예상되는 것들
        'others'
        for x in keyword
    ]
    _case = pd.Series(_case)
    _res = _case.value_counts().sort_index()
    
    #------------------------------------------------------------------------------------------------------------#
    # case group과 table을 return
    #------------------------------------------------------------------------------------------------------------#
    # _case : case1 ~ case11(오적재 그룹), others(정상으로 예상되는 그룹)
    # _res  : 오적재 케이스 Freq.
    return np.array(_case),_res

#------------------------------------------------------------------------------------------------------------#
# [2] 키워드 전처리 케이스 파악하는 함수
#------------------------------------------------------------------------------------------------------------#
def check_1(df):

    # displays(chk,sort_var='key_len')

    print(f'>\t기존 키워드 개수 : {df.keyword.astype(str).nunique():,}')

    quote_cnt = df[df["keyword"]!=df["keyword_new"]]["keyword_new"].nunique()
    print(f'>\t따옴표 수정 필요한 키워드 건수 : {quote_cnt:,}')
    if quote_cnt>0:
        print(f'>\t  수정 전 키워드 건수 : {df.keyword.nunique():,}')
        print(f'>\t  수정 후 키워드 건수 : {df.keyword_new.nunique():,}')
        print(f'>\t  수정 전/후 차이 : {df.keyword.nunique() - df.keyword_new.nunique():,}')

        if (df.keyword.nunique() - df.keyword_new.nunique()) != 0:
        
            key = df[df['keyword']!=df['keyword_new']]['keyword_new'].unique()
            chk = df[df['keyword_new'].isin(key)]
            cnt_info = chk.keyword_new.value_counts().value_counts()

            print(f'>\t    - 기존에 존재하던 키워드와 중복 O : {cnt_info[cnt_info.index >1].values[0]:,}')
            print(f'>\t    - 기존에 존재하던 키워드와 중복 X : {cnt_info[cnt_info.index==1].values[0]:,}')
    
            check_count = cnt_info.sum() -\
                df[df["keyword"]!=df["keyword_new"]]["keyword_new"].nunique()
            check = '정상' if check_count==0 else '에러'

            col = color.BLUE if check_count==0 else color.RED
            
            print(f'>\t    - 합계                      : {cnt_info.sum():,} (키워드 건수 확인 결과 : {col}{check}{color.END})')
    
#------------------------------------------------------------------------------------------------------------#
# [3] 전체 전처리 후 건수 변동 파악
#------------------------------------------------------------------------------------------------------------#
def check_2(df,keyword_df,final_df):
    
    print(f'>\t건수 변동 : {df.shape[0]:,}건 -> {final_df.shape[0]:,}건 ({df.shape[0]-final_df.shape[0]:,}건 제거)')
    print(f'>\t키워드 변동 : {df.keyword.nunique():,}건 -> {final_df.keyword.nunique():,}건', 
          f'({df.keyword.nunique()-final_df.keyword.nunique():,}건 제거)')

    key = keyword_df['keyword_tobe'][keyword_df["keyword_asis"]!=keyword_df["keyword_tobe"]]
    chk = keyword_df[keyword_df['keyword_tobe'].isin(key)]['keyword_tobe']

    print(f'>\t따옴표 변동건 : {chk.value_counts().value_counts().sum():,}건',
          f'(기존 전체 {keyword_df.shape[0]:,}건)')
    
#------------------------------------------------------------------------------------------------------------#
# [4] 최종 키워드 전처리 함수 (건수 프린트 포함)
#------------------------------------------------------------------------------------------------------------#
# > Args :
#   - df : 전처리를 하기 이전의 데이터
#          함수 맨 앞부분에서 keyword의 unique를 들고와서 작업하기 때문에, 시간이 오래 걸리지 않음
#          (해당 키워드의 정상/비정상 여부를 확인하기 때문에, unique를 들고옴)
#          이후, 유의한 키워드만을 선택해서 최종 데이터셋을 return
#
# > Returns :
#  (1) keyword_df
#      - keyword_asis : 기존 키워드
#      - keyword_tobe : 따옴표 제거 후 키워드
#      - keyword_error_case : keyword_error_case에서 정의된 "확인 된" 오적재 케이스를 확인 (쿼리참조)
#      - keyword_special_char : 특수문자가 포함되었는지 여부
#      - keyword_only_number : 문자열이 모두 숫자인지 여부
#
#  (2) final_keyword_df : 전처리 후 키워드를 저장, 따옴표 포함된 키워드를 최종 데이터셋에 적용하기 위해 생성
#      - keyword_asis : 따옴표 포함된 키워드를 포함한 기존 키워드
#      - keyword_tobe : 따옴표를 수정한 키워드
#
#  (3) final_df : 전처리 모두 반영한 최종 데이터셋
#------------------------------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd

import datetime
import time
import os,sys
import re
import pytz

def KeywordPreprocessing(df):
    
    # 함수 시작/종료시간 display -> 한국시간으로 표시
    KST = pytz.timezone('Asia/Seoul')
    
    # 함수 시작시간
    start_time = str(datetime.datetime.now(KST))[:19]
    
    # 키워드를 unique로 가져와서 사용 (속도개선)
    d = pd.DataFrame({
        'keyword' : df.keyword.unique()
    })
    
    #------------------------------------------------------------------------------------------------------------#
    # (1) None 제거
    #------------------------------------------------------------------------------------------------------------#
    print('-'*100)
    print('> (1/8) None 제거 - 건수 :', d[d.keyword.astype(str)=='None'].shape[0])
    if d[d.keyword.astype(str)=='None'].shape[0]>0:
        d = d[d.keyword.astype(str)!='None']
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (2) Null 제거
    #------------------------------------------------------------------------------------------------------------#
    print('> (2/8) Null 제거 - 건수 :', d[(d.keyword=='') | (d.keyword.isnull())].shape[0])
    if d[(d.keyword=='') | (d.keyword.isnull())].shape[0]>0:
        d = d[(d.keyword!='') & (~d.keyword.isnull())]
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (3) 따옴표 제거 (정상인 건에서, 따옴표가 붙은 건 제거)
    #------------------------------------------------------------------------------------------------------------#
    keyword_list = d.keyword.unique()

    # 따옴표 제거 룰 : 첫번째와 마지막이 따옴표인 경우 제거
    # (why?) 쌍따옴표가 있는 키워드는 의미없는 키워드라고 생각하고, 바로 쌍따옴표를 제거해버리면 되지만,
    #        추후에는 문제가 될 수도 있을거라 판단되기 때문에, 보수적으로 따옴표를 제거하도록 하였음.
    key_new = [k.replace('"','') if (k.find('"')==0) & (k.find('"',1)==len(k)-1) else
               k.replace("'",'') if (k.find("'")==0) & (k.find("'",1)==len(k)-1) else
               k for k in keyword_list]
    key_new = np.array(key_new)

    old = keyword_list[keyword_list!=key_new] # 수정해야하는 대상의 수정 전 키워드
    new = key_new[keyword_list!=key_new]      # 수정해야하는 대상의 수정 후 키워드

    start_time_x = time.time()
    if len(new)>0:
        print(f'> (3/8) 따옴표 제거 필요 : {len(new):,} / {len(keyword_list):,}')
        # 키워드 별로 따옴표 제거한 변수 추가
        total = len(new)
        new_df_1 = d[d.keyword.isin(new)]
        new_df_1['keyword_new'] = new_df_1['keyword']
        new_df_2 = []

        for iter in range(len(new)):
            prt(f'\r>       따옴표 제거 : {iter+1:,} / {total:,} ({iter/total*100:.1f}%)    ')

            sub = d[d.keyword==old[iter]]
            sub['keyword_new'] = new[iter]
            new_df_2.append(sub)

        new_df_2 = pd.concat(new_df_2,axis=0)
        new_df = pd.concat([
            new_df_1,
            new_df_2,
        ],axis=0)
        
        end_time_x = time.time()
        run_time_x = (end_time_x-start_time_x)/60
        
        prt(f'\r>       따옴표 제거 : {iter:,} / {total:,} ({iter/total*100:.1f}%), Runtime : {run_time_x:.2f} Min\n')
        
    else:
        new_df = d.copy()
        new_df['keyword_new'] = new_df['keyword']
        print(f'>       따옴표 제거 : 따옴표 제거 대상 없음')
    
    print('-'*100)
        

    #------------------------------------------------------------------------------------------------------------#
    # (4) 예측된 오적재 케이스 확인
    #------------------------------------------------------------------------------------------------------------#
    print(f'> (4/8) 예측된 오적재 케이스 확인 (케이스 상세설명은 쿼리 참조)')
    case,res = keyword_error_case(new_df['keyword_new'].values)
    new_df['keyword_error_case'] = case
    for iter in range(len(res)):
        print(f'>\t{res.index[iter]} : {res[iter]:,}')
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (5) 최종 오적재 케이스 별 건수 파악
    #------------------------------------------------------------------------------------------------------------#
    print(f'> (5/8) 최종 오적재 케이스 별 건수 파악')
    check_1(new_df[new_df['keyword_error_case'] == 'others'])
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (6) 최종 키워드 전처리 (이슈있는 키워드 제거)
    #------------------------------------------------------------------------------------------------------------#
    print(f'> (6/8) 최종 키워드 전처리')
    print('-'*100)
    keyword_df = new_df[
        (new_df['keyword_error_case'] == 'others')
    ]\
        [['keyword','keyword_new']].\
        rename(columns={'keyword'    :'keyword_asis',
                        'keyword_new':'keyword_tobe'})
    
    #------------------------------------------------------------------------------------------------------------#
    # (7) 전처리 후, 최종 데이터 셋 생성
    #------------------------------------------------------------------------------------------------------------#
    df2 = df[df.keyword.isin(keyword_df['keyword_asis'])]

    keyword_list = keyword_df['keyword_asis'][keyword_df['keyword_asis']!=keyword_df['keyword_tobe']]
    
    if len(keyword_list)>0:
        df_1 = df2[ df.keyword.isin(keyword_list)]
        df_2 = df2[~df.keyword.isin(keyword_list)]

        start_time_x = time.time()
        final_df_1 = []
        final_df_2 = []
        iter = 0
        total = len(keyword_list)
        for k in keyword_list:
            iter += 1
            prt(f'\r> (7/8) 전처리 후, 최종 데이터셋 생성 : {iter:,} / {total:,} ({iter/total*100:.1f}%)  ')

            sub_df = df_1[df_1.keyword==k]
            key_df = keyword_df[keyword_df['keyword_asis']==k]

            sub_df['keyword'] = key_df['keyword_tobe'].values[0]

            final_df_1.append(sub_df)

        final_df_1 = pd.concat(final_df_1,axis=0)
        final_df_2 = df_2

        final_df = pd.concat([
            final_df_1,
            final_df_2,
        ],axis=0).sort_values('keyword')

        end_time_x = time.time()
        run_time_x = (end_time_x-start_time_x)/60
        prt(f'\r> (7/8) 전처리 후, 최종 데이터셋 생성 : {iter:,} / {total:,} ({iter/total*100:.1f}%), Runtime : {run_time_x:.2f} Min\n')
    else:
        prt(f'> (7/8) 전처리 후, 최종 데이터셋 생성 : 따옴표 변경사항 없으므로, 이외의 오적재 케이스만 삭제\n')
        final_df = df2.copy()
    print('-'*100)
    
    #------------------------------------------------------------------------------------------------------------#
    # (8) 전체 전처리 후, 건수 변동 파악
    #------------------------------------------------------------------------------------------------------------#
    print(f'> (8/8) 전체 전처리 후, 건수 변동 파악')
    check_2(df,keyword_df,final_df)
    print('-'*100)
    
    res = (
        new_df,      # keyword_error_case 확인
        keyword_df,  # 따옴표 제거전/후 비교
        final_df,    # 최종 데이터셋
    )
    
    # 함수 종료시간
    end_time = str(datetime.datetime.now(KST))[:19]
    
    fmt = '%Y-%m-%d %H:%M:%S'
    run_time = datetime.datetime.strptime(end_time,fmt) - datetime.datetime.strptime(start_time,fmt)
    
    print('')
    print('-'*100)
    print(f'> start time : {start_time}')
    print(f'>   end time : {end_time}')
    print(f'>   run time : {run_time}')
    print('-'*100)

    return res

# # Example
# keyword_df, final_keyword_df, final_df = KeywordPreprocessing(df)