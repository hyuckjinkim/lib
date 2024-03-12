import selenium
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.ie.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

class BaseChromeWebdriver:
    def __init__(self,
                 chromedriver_path: str = None,
                 set_options: bool = False,
                 timeout: int = None):
        
        self.timeout = timeout
        
        if chromedriver_path is None:
            print('** Install ChromeDriver **')
            
            # chromedriver 설치하여 copy해서 가져오기
            # (참조) https://stackoverflow.com/questions/78093580/chrome-driver-version-122-0-6261-95
            chromedriver_path = ChromeDriverManager().install()
            # !cp {chromedriver_path} 'tools/chromedriver'
        
        options = self._get_options() if set_options else None
        service = ChromeService(chromedriver_path)
        self.driver = webdriver.Chrome(service=service, options=options)
        self.driver.implicitly_wait(10)
    
    def _get_options(self) -> selenium.webdriver.chrome.options.Options:
        # 압축해제한 웹드라이버의 경로와 파일명 지정
        options = webdriver.ChromeOptions()
        options.add_argument('headless')
        options.add_argument('window-size=1920x1080')
        options.add_argument("disable-gpu")
        options.page_load_strategy = 'none'
        return options
    
    def get(self, url: str) -> None:
        self.driver.get(url=url)
        
    def close(self) -> None:
        self.driver.close()
    
    def find_element_by_css_selector(self, selector: str) -> selenium.webdriver.remote.webelement.WebElement:
        if self.timeout is None:
            elements = self.driver.find_element(By.CSS_SELECTOR, selector)
        else:
            wait = WebDriverWait(self.driver, self.timeout)
            condition = EC.presence_of_element_located
            elements = wait.until(condition((By.CSS_SELECTOR, selector)))
        return elements
    
    def find_element_by_css_selector_with_clickable(self,
                                                    selector: str,
                                                    timeout: int) -> selenium.webdriver.remote.webelement.WebElement:
        if timeout is None:
            elements = self.driver.find_element(By.CSS_SELECTOR, selector)
        else:
            wait = WebDriverWait(self.driver, timeout)
            condition = EC.element_to_be_clickable
            elements = wait.until(condition((By.CSS_SELECTOR, selector)))
        return elements