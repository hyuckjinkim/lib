"""
로그를 확인에 필요한 함수와 클래스를 제공한다.

클래스 목록
1. `PrintStream`
    Logging 사용 시, 표준 출력 캡처를 위한 클래스.
    
함수 목록
1. `get_logger`
    log를 관리하기 위한 logger를 가져온다.
2. `trace`
    특정 함수의 진입 여부 로깅을 남기는 함수로, 데코레이터로 사용한다.
"""

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

def get_logger(logger_name: str = None,
               module_name: str = None,
               level: str = 'DEBUG',
               flag: str|None = None,
               save_path: str = None) -> logging.Logger:
    """
    log를 관리하기 위한 logger를 가져온다.
    
    Args:
        logger_name (str, optional): logger의 name. default=None.
        module_name (str, optional): module의 name. default=None.
        flag (str | None, optional): 로거 유형. default=None.
        save_path (str, optional): 출력물의 저장위치. default=None.
    
    Raises:
        level이 ['DEBUG','INFO']에 속하지 않는 경우.
        flag가 ['deco',None]에 속하지 않는 경우.
        save_path가 '.log'의 형태가 아닌 경우.
    
    Returns:
        logging.Logger: logger 객체.
    """
    
    # raise error
    assert level in ['DEBUG','INFO'], "level must be one of ['DEBUG','INFO']"
    assert flag in ['deco',None], "flag must be one of ['deco',None]."
    if save_path is not None:
        assert save_path.split('/')[-1].find('.log')>0, "The 'save_path' must be a string ending with '.log'."
        # log 폴더 생성
        if not os.path.exists(save_path):
            save_dir = os.path.join(*save_path.split('/')[:-1])
            os.makedirs(save_dir, exist_ok=True)
        # log 파일 삭제
        os.system(f"rm -rf {save_path}")
    
    if level=='INFO':
        logger_level = logging.INFO
    elif level=='DEBUG':
        logger_level = logging.DEBUG
    
    # logger 생성
    logger = logging.getLogger(logger_name)
    
    # logging 모듈은 getLogger를 통해 새로운 logger를 받아올 때, 같은 이름일 경우 기존 logger를 그대로 return하는 싱글턴 패턴을 사용한다.
    # 같은 logger에 handler를 계속 붙이기 때문에 한 번 logging해도 여러 개의 handler가 작동하여 log가 여러 개 찍힌다.
    # 이 점을 방지하기 위해 기존에 handler가 있는지 확인한다.
    if len(logger.handlers) > 0:
        return logger
    
    # logger 레벨 설정
    logger.setLevel(logger_level)
    
    # asctime: 시간정보, levelname: logging level, funcName: log가 기록된 함수, lineno: log가 기록된 line
    if flag=='deco':
        formatter = logging.Formatter(f'[%(asctime)s {module_name.split(".")[-1]}.py][%(levelname)s] %(message)s')
    else:
        formatter = logging.Formatter('[%(asctime)s %(filename)s:%(lineno)d][%(levelname)s] %(message)s')
    
    # 핸들러 추가
    handlers = [logging.StreamHandler()]
    if save_path is not None:
        handler = logging.FileHandler(filename=save_path)
        handlers.append(handler)

    # 핸들러 설정
    for handler in handlers:
        handler.setLevel(logger_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # print 출력을 파일로 리다이렉트
    print_streamer = PrintStream()
    print_streamer.add_logger(logger)
    sys.stdout = print_streamer

    return logger

def trace(func):
    """
    특정 함수의 진입 여부 로깅을 남기는 함수로, 데코레이터로 사용한다.

    .. code-block:: python
    
        함수의 데코레이터로 '@trace'를 붙여 사용한다.
            ex)
            from data_lib.util.function_logging import trace

            @trace
            def example_func():
                pass


        로깅 출력 포맷
            ex)
            [2023-06-22 15:08:38,446 create_chunk_list.py][INFO] Entered to 'create_chunk_list()'
            [2023-06-22 15:08:38,447 create_chunk_list.py][INFO] Exited 'create_chunk_list()'
    
    """
    logger = get_logger(func.__name__, func.__module__, 'DEBUG', 'deco')
    
    def wrapper(self, *args, **kwargs):
        logger.info("Entered to '{}()'".format(func.__name__))
        function_response = func(self, *args, **kwargs)
        logger.info("Exited from '{}()'".format(func.__name__))
        return function_response
    return wrapper