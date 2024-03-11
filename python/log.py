import os, sys
from io import StringIO
import logging

class PrintStream(StringIO):
    """Logging 사용 시, 표준 출력 캡처를 위한 클래스."""
    def add_logger(self, logger: logging.RootLogger) -> None:
        """
        self 객체에 logger를 추가한다.
        
        Args:
            logger (logging.RootLogger): logger 객체.
            
        Returns:
            None.
        """
        self.logger = logger
    
    def write(self, buf) -> None:
        """print 출력을 로그에 추가한다."""
        self.logger.info(buf.strip())

def get_logger(name: str = None, save_path: str = 'log_saved.log') -> logging.RootLogger:
    """
    log를 관리하기 위한 logger를 가져온다.
    
    Args:
        name (str): logger의 name. default=None.
        save_path (str): 출력물의 저장위치. default=None.
    
    Raises:
        save_path가 '.log'의 형태가 아닌 경우.
    
    Returns:
        logging.RootLogger: logger 객체.
    """
    
    # raise error
    assert save_path.split('/')[-1].find('.log')>0, "The 'save_path' must be a string ending with '.log'."
    
    # log 폴더 생성
    if not os.path.exists(save_path):
        save_dir = os.path.join(*save_path.split('/')[:-1])
        os.makedirs(save_dir, exist_ok=True)
    
    # logger 생성
    if name is None:
        name = 'root'
    logger = logging.getLogger(name)
    
    logger.setLevel(logging.INFO)
    #logger.setLevel(logging.DEBUG)
    
    # asctime - 시간정보
    # levelname - logging level
    # funcName = log가 기록된 함수
    # lineno - log가 기록된 line
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(funcName)s:%(lineno)d] - %(message)s")
    console = logging.StreamHandler()
    
    # 파일 저장위치 설정
    file_handler_info = logging.FileHandler(filename=save_path)
    #file_handler_debug = logging.FileHandler(filename=save_path)

    # 콘솔 출력 핸들러 설정
    console.setLevel(logging.DEBUG)
    file_handler_info.setLevel(logging.INFO)
    #file_handler_debug.setLevel(logging.DEBUG)

    # 파일 출력 핸들러 설정
    console.setFormatter(formatter)
    file_handler_info.setFormatter(formatter)
    #file_handler_debug.setFormatter(formatter)

    # 핸들러 추가    
    logger.addHandler(console)
    logger.addHandler(file_handler_info)
    #logger.addHandler(file_handler_debug)

    # print 출력을 파일로 리다이렉트
    print_streamer = PrintStream()
    print_streamer.add_logger(logger)
    sys.stdout = print_streamer

    return logger