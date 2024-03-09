#----------------------------------------------------------------------------#
# > 설명 : pickle load/save를 쉽게하기위해서 만든 함수.
#         pandas의 read_csv, to_csv와 비슷하도로 구성.
#----------------------------------------------------------------------------#
import pickle

def to_pickle(data,path):
    with open(path, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    
def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

#----------------------------------------------------------------------------#
# > 설명 : 폴더생성 함수
#----------------------------------------------------------------------------#
import os
def mkdir(paths)->None:
    if isinstance(paths,str):
        paths = [paths]
    for path in paths:
        if not os.path.exists(path):
            os.mkdir(path)
            print('directory created: {}'.format(path))