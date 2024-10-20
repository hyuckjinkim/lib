"""
visualization 작업과 관련된 함수와 클래스를 제공한다.

함수 목록
1. `abline`
    기울기 및 절편으로 이루어진 선을 표시한다.
    
클래스 목록
1. `MatplotlibFontManager`
    MatplotlibFontManager의 생성자로, matplotlib의 font를 관리한다.
    
함수 목록
1. `abline`
    기울기 및 절편으로 이루어진 선을 표시한다.
2. `get_sorted_handles`
    plt.legend 설정 시, sorting 순서를 조절한다.
3. `actual_prediction_scatterplot`
    Actual vs Prediction에 대한 산점도를 그린다.
"""

# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

GIT_DIR = 'G:/My Drive/Storage/Github/hyuckjinkim'
FONT_DIR = os.path.join(GIT_DIR, 'tools/NanumFont')
KOREAN_FONT_PATH = os.path.join(GIT_DIR, 'tools/NanumFont/NanumGothic.ttf')

class MatplotlibFontManager:
    """
    matplotlib의 font를 관리하기위한 클래스.
    
    예시
    ```python
    from lib.python.graph import MatplotlibFontManager
    fm = MatplotlibFontManager()
    fm.set_korean_font(check=False)
    ```
    """
    
    def __init__(self, font_path: str = None):
        """
        MatplotlibFontManager의 생성자로, matplotlib의 font를 관리한다.
        
        Args:
            font_path (str): 폰트 경로. default=None으로, None인 경우에 'domains/tools/NanumFont/NanumGothic.ttf'의 폰트를 가져온다.
        """
        
        self.font_dir = FONT_DIR
        self.font_path = font_path
        if font_path is None:
            self.font_path = KOREAN_FONT_PATH
        
    def get_font_file_paths(self) -> list:
        """data-mlops/domains/tools에 저장된 ttf 파일의 경로를 반환한다."""
        return glob.glob(FONT_DIR+'/*.ttf')

    def set_korean_font(self, check: bool = False) -> None:
        """한국어 폰트를 적용한다.
        
        Args:
            check (bool, optional): 한국어 폰트가 잘 적용되었는지 확인한다. default=False.
        """
        fm.fontManager.addfont(self.font_path)
        font_name = fm.FontProperties(fname=self.font_path).get_name()
        plt.rc("font", family=font_name)
        plt.rc('axes', unicode_minus=False)
        
        if check:
            plt.figure(figsize=(5,2))
            plt.plot()
            plt.title('한국어 폰트 체크 확인')
            plt.show()
            
        return None

def abline(intercept: float,
           slope: float,
           linewidth: int = 2,
           linestyle: str = 'dashed',
           color: str = 'red',
           ax: matplotlib.axes.Axes = None,
           **kwargs) -> None:
    """
    기울기 및 절편으로 이루어진 선을 표시한다.
    
    Args:
        intercept (float): y축 절편 값.
        slope (float): 기울기 값.
        linewidth (int, optional): 선 굵기. default=2.
        linestyle (str, optional): 선 스타일. default='dashed'.
        color (str, optional): 선 색상. default='red'.
        ax (matplotlib.axes.Axes, optional): 선을 plot할 Axes 객체. default=None.
        
    Returns:
        None.
    """
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    if ax is None:
        plt.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color, **kwargs)
    else:
        ax.plot(x_vals, y_vals, linestyle=linestyle, linewidth=linewidth, color=color, **kwargs)

def get_sorted_handles(ax, ref_labels):
    """plt.legend 설정 시, sorting 순서를 조절한다."""
    handles, labels = ax.get_legend_handles_labels()
    sorted_loc = [np.where(np.array(labels)==l)[0][0] for l in ref_labels]
    new_handles = np.array(handles)[sorted_loc].tolist()
    return new_handles

def actual_prediction_scatterplot(y_true: list, y_pred: list, title: str = None):
    """
    Actual vs Prediction에 대한 산점도를 그린다.
    
    Args:
        y_true (list): Actual 값의 리스트.
        y_pred (list): Prediction 값의 리스트.
        title (str, optional): 타이틀로 설정 할 텍스트. default=None.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if title is None:
        rmse = np.mean((y_true-y_pred)**2)**0.5
        title = 'RMSE={:.3f}'.format(rmse)
    
    #offset = 0.05
    #xylim = (np.min([y_true,y_pred])*(1-offset),np.max([y_true,y_pred])*(1+offset))
    
    plt.figure(figsize=(15,7))
    sns.scatterplot(x=y_true,y=y_pred)
    abline(0,1)
    plt.xlabel('Actual')
    plt.ylabel('Prediction')
    #plt.xlim(xylim)
    #plt.ylim(xylim)
    plt.grid()
    plt.title(title)
    plt.show()