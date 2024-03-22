"""
Selenium의 Chrome과 관련된 함수와 클래스를 제공한다.

클래스 목록
1. `BaseChromeWebdriver`
    Selenium Chrome Webdrive를 초기세팅하는 클래스.
"""

import time

import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.ie.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.remote.webelement import WebElement

from webdriver_manager.chrome import ChromeDriverManager

from seleniumwire import webdriver as wired_webdriver

class BaseChromeWebdriver:
    """Selenium Chrome Webdrive를 초기세팅하는 클래스."""
    def __init__(self,
                 wired: bool = False,
                 chromedriver_path: str = None,
                 headless: bool = False,
                 timeout: int = None) -> None:
        """
        BaseChromeWebdriver의 생성자로, Selenium Chrome Webdrive를 초기세팅한다.
        
        **참조**
        1. chrome drive 수기로 설치하는 방법
            * 구글크롬 > 설정 > Chrome 정보에서 크롬버전 확인 (url: chrome://settings/help)
            * 다음 링크에서 버전에 맞는 크롬드라이버 설치 : https://chromedriver.chromium.org/downloads
        2. selector 얻는 방법
            * (Mac 기준) option + command + i로 개발자설정을 연다
            * 개발자설정 좌측상단의 화살표버튼을 통해 크롤링하고자하는 객체를 클릭하여 '요소'를 확인한다
            * 요소에 우클릭 > 복사 > selector 복사를 통해 selector를 얻는다
        
        Args:
            wired (bool, optional): selenium-wire를 사용 할 것인지 여부. default=False.
            chromedrive_path (str, optional): 크롬드라이버가 설치된 경로. None인 경우 `ChromeDriverManager().install()`을 통해서 설치한다. default=None.
            headless (bool, optional): 기본 options를 세팅 시, headless를 설정할지 여부. default=False.
            timeout (int, optional): timeout을 설정할지 여부. None이 아닌 경우 `WebDriverWait(driver, timeout).until(EC.presence_of_element_located(By.CSS_SELECTOR, selector))`의 형태로 설정한다. default=None.
            
        Returns:
            None.
        """
        
        self.timeout = timeout
        
        # 크롬드라이버 설치
        if chromedriver_path is None:
            print('** Install ChromeDriver **')
            
            # chromedriver 설치하여 copy해서 가져오기
            # (참조) https://stackoverflow.com/questions/78093580/chrome-driver-version-122-0-6261-95
            chromedriver_path = ChromeDriverManager().install()
            # !cp {chromedriver_path} 'tools/chromedriver'
        
        options = self._get_options(headless)
        service = ChromeService(chromedriver_path)
        if wired:
            self.driver = wired_webdriver.Chrome(service=service, options=options, keep_alive=False)
        else:
            self.driver = webdriver.Chrome(service=service, options=options, keep_alive=False)
        #self.driver.implicitly_wait(10)
    
    def _get_options(self, headless: bool = True) -> selenium.webdriver.chrome.options.Options:
        """기본 options를 세팅한다."""
        # 압축해제한 웹드라이버의 경로와 파일명 지정
        options = webdriver.ChromeOptions()
        if headless:
            options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")
        options.page_load_strategy = 'none'
        return options
    
    def get(self, url: str) -> None:
        """
        chrome webdriver를 통해 url로 접속한다.
        
        Args:
            url (str): 접속하고자하는 url 정보.
            
        Returns:
            None.
        """
        self.driver.get(url=url)
        
    def close(self) -> None:
        """chrome webdriver를 종료한다."""
        self.driver.close()
        
    def scroll_down(self):
        # (1) java script 기반의 스크롤 다운
        self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # (2) END 키를 통한 스크롤 다운
        body_elements = self.find_element_by_css_selector('body')
        for _ in range(10):
            body_elements.send_keys(Keys.END)
    
    def find_element_by_css_selector(self,
                                     selector: str,
                                     condition = EC.presence_of_element_located) -> WebElement:
        """
        selector를 통해서 element를 찾는다.
        
        Args:
            selector (str): 웹 페이지 내에서 특정 요소를 식별하기 위해 사용되는 CSS 선택자 문자열.
            condition (Callable, optional): 선택된 요소의 상태를 평가하는 데 사용되는 함수. selenium.webdriver.support.expected_conditions의 함수를 인자로 받는다. default=EC.presence_of_element_located.

        Returns:
            selenium.webdriver.remote.webelement.WebElement: 선택된 조건에 따라 상태가 평가된 웹 요소.
        """
        
        time.sleep(3)
        if self.timeout is None:
            elements = self.driver.find_element(By.CSS_SELECTOR, selector)
        else:
            wait = WebDriverWait(self.driver, self.timeout)
            elements = wait.until(condition((By.CSS_SELECTOR, selector)))
        time.sleep(3)
        return elements
    
    def find_element_by_css_selector_with_clickable(self,
                                                    selector: str) -> WebElement:
        """
        selector를 통해서 element를 찾고, 해당 요소가 클릭 가능한 상태인지 확인한다.
        
        Args:
            selector (str): 웹 페이지 내에서 특정 요소를 식별하기 위해 사용되는 CSS 선택자 문자열.

        Returns:
            selenium.webdriver.remote.webelement.WebElement: 선택된 조건에 따라 상태가 평가된 웹 요소.
        """
        
        condition = EC.element_to_be_clickable
        return self.find_element_by_css_selector(selector, condition)