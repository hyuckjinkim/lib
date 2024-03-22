from abc import ABCMeta,abstractmethod

import requests


class BaseExtractor(metaclass=ABCMeta):
    """크롤링을 위한 추상화 클래스"""
    
    def set_request_config(self, **kwargs):        
        """request에 필요한 config 를 저장하는 함수
        """
        # variable
        if 'url' in kwargs:
            self.__url = kwargs['url']
        if 'cookies' in kwargs:
            self.__cookies = kwargs['cookies']
        if 'headers' in kwargs:
            self.__headers = kwargs['headers']
        if 'params' in kwargs:
            self.__params = kwargs['params']
        if 'json'  in kwargs:
            self.__json = kwargs['json']
    
    
    def request(self, method: str = "GET") -> str:
        """요청받은 결과를 문자열 형태로 반환

        Returns:
            str: Parsing 이 필요한 문자열
        """

        if method == "GET":
            resp = requests.get(
                self.__url, params=self.__params, headers=self.__headers, cookies=self.__cookies
            )
            return resp
        elif method == "POST":
            resp = requests.post(self.__url, json=self.__json)
            return resp

    @abstractmethod
    def crawl(self, **kwargs) -> list[dict]:
        """Queue 에 들어온 메시지로 크롤링을 진행하하는 함수로, extractor 의 진입함수이다.
            - request, parse 함수가 실행된다.

        Returns:
            list[dict]: 크롤링 결과
        """
        pass