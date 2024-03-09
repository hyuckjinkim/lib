# Configuration과 관련된 함수들을 저장하는 공간

## install.py
- .py 파일로 자동화 설정 시, !pip install이 불가능하기 때문에 os.system()으로 설정이 필요

## send_mail.py
- email 전송을 하는 함수. 자동화 설정 시, 알람설정이 가능
- send_mail에서 password에는 앱 비밀번호를 설정해야함. gmail에서 앱비밀번호를 설정하는 방법은 [tutorial](https://greensul.tistory.com/31) 참조.