import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#----------------------------------------------------------------------------#
# > 설명 : R의 abline과 동일한 결과를 보여주는 함수
# > 상세
#    - intercept : y절편으로, 직선함수 y=a+bx의 a에 해당
#    - slope : 기울기로, 직선함수 y=a+bx의 b에 해당
#    - linewidth : 결과물인 직선의 두께 설정
#    - linestyle : 결과물인 직선의 선유형 설정
#    - color : 결과물인 직선의 색상 설정
#----------------------------------------------------------------------------#
def abline(intercept,slope,linewidth=2,linestyle='--',color='red',ax=None):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    if ax is None:
        plt.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color)
    else:
        ax.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color)

#----------------------------------------------------------------------------#
# > 설명 : plt.legend 설정 시, sorting 순서를 조절하는 함수
#----------------------------------------------------------------------------#
def get_sorted_handles(ax,ref_labels):
    handles, labels = ax.get_legend_handles_labels()
    sorted_loc = [np.where(np.array(labels)==l)[0][0] for l in ref_labels]
    new_handles = np.array(handles)[sorted_loc].tolist()
    return new_handles

def actual_prediction_scatterplot(y_true,y_pred,title=None):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if title is None:
        rmse = np.mean((y_true-y_pred)**2)**0.5
        title = 'RMSE={:.3f}'.format(rmse)
    
    offset = 0.05
    xylim = (np.min([y_true,y_pred])*(1-offset),np.max([y_true,y_pred])*(1+offset))
    
    plt.figure(figsize=(15,7))
    sns.scatterplot(x=y_true,y=y_pred)
    abline(0,1)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    plt.xlim(xylim)
    plt.ylim(xylim)
    plt.grid()
    plt.title(title)
    plt.show()