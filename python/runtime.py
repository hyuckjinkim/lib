import datetime
import pytz

class runtime:
    
    def __init__(self):
        self.kst = pytz.timezone('Asia/Seoul')
        self.fmt = '%Y-%m-%d %H:%M:%S'
    
    def current_time(self):
        # 함수 시작/종료시간 display -> 한국시간으로 표시
        start_time = str(datetime.datetime.now(self.kst))[:19]
        return start_time
    
    def run_time(self,start,end,verbose=True):
        run_time = datetime.datetime.strptime(end,self.fmt) - datetime.datetime.strptime(start,self.fmt)

        if verbose:
            print('')
            print('-'*100)
            print(f'> start time : {start}')
            print(f'>   end time : {end}')
            print(f'>   run time : {run_time}')
            print('-'*100)
            return run_time
        elif verbose:
            return run_time
    
# start_time = runtime().current_time()
# end_time   = runtime().current_time()
# runtime().run_time(start_time,end_time)