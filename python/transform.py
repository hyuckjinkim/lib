import numpy as np
import pandas as pd

#----------------------------------------------------------------------------#
# > 설명 : 양수의 값에 대해서, log변환 및 재변환하는 함수
# > 상세
#    - 이러한 변환을 하는 근거에 대해서 정확하게 확인을 하지 못함
#    - lower_bound는 양의 값에 대한 변환이므로 0
#    - upper_bound는 왜 100을 곱해주는지 잘 모르겠음
# > 예시
#     max_number = 10**2
#     x = np.array(sorted([-x for x in range(1,max_number)] + [0] + [x for x in range(1,max_number)]))
#     y = positive_transformation(x)[0]
#     plt.plot(x,y)
#----------------------------------------------------------------------------#
def positive_transformation(x,bound=None):
    if bound is None:
        adj = -0.1
        lower_bound = 0 + adj
        upper_bound = max(x)*100 - adj
    else:
        lower_bound = bound[0]
        upper_bound = bound[1]
    return np.log((x-lower_bound)/(upper_bound-x)), lower_bound, upper_bound
    
def inverse_positive_transformation(x, lower_bound, upper_bound):
    upper_bound = upper_bound
    lower_bound = lower_bound
    return (upper_bound-lower_bound)*np.exp(x)/(1+np.exp(x))+lower_bound

#----------------------------------------------------------------------------#
# > 설명 : smoothing line function
# > 예시
#     smooth(y,box_pts=30)
#----------------------------------------------------------------------------#
# https://stackoverflow.com/questions/20618804/how-to-smooth-a-curve-in-the-right-way
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

#----------------------------------------------------------------------------#
# > 설명 : data point들에 대한 미분값 계산
# > 예시
#     smooth(y,box_pts=30)
#----------------------------------------------------------------------------#
def calculate_differential(x,y,step=1,offset=1e-3,is_timeseries=False):
    data = pd.DataFrame({'x':x,'y':y})
    if is_time_series:
        data.sort_values('x',inplace=True)
        dx = (data.x - data.x.shift(step)).dt.days.values[step:]
        dy = (data.y - data.y.shift(step)).values[step:]
    else:
        dx = (data.x - data.x.shift(step)).values[step:]
        dy = (data.y - data.y.shift(step)).values[step:]
    return dy/(dx+offset)