from discord import SyncWebhook
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import ntpath

def notify_to_discord(discord_webhook_url: str, msg: str) -> None:
    """디스코드에 알림을 보낸다.

    Args:
        discord_webhook_url (str): 원하는 디스코드 채널의 webhook url
        msg (str): 전송할 메세지  
            - 일반 메세지, 에러 메세지, 템플릿화된 메세지 등
    """
    try:
        webhook = SyncWebhook.from_url(discord_webhook_url)
        webhook.send(content=msg)
    except RuntimeError as e :
        print(e)
        print("디스코드 메시지 발송에 문제가 생겼습니다.")

def send_email(to: list,
               subject: str,
               html_message_body: str,
               cc: list = None,
               attach_files: list = None,
               ses_config = None) -> None:
    """이메일을 보낸다.

    Args:
        to (list) : 이메일 수신자
        cc (list) : 이메일 참조자
        subject (str) : 제목
        html_message_body (str): 본문으로, html 형식으로 작성가능함
        attach_files (list): 메세지에 첨부할 엑셀 파일 경로 리스트
    """
    
    SENDER_EMAIL = ses_config["email"]
    SMTP_ID = ses_config["access_key"]
    SMTP_PW = ses_config["secret_key"]
    SMTP_SERVER = ses_config["server"]
    SMTP_PORT = ses_config["port"]

    # 메세지 메타데이터
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = ','.join(to)
    if cc is not None:
        msg['Cc'] = ','.join(cc)

    # 메세지 본문
    content = MIMEText(html_message_body, 'html')
    msg.attach(content)

    # 파일 첨부
    if attach_files is not None:
        for attach_file in attach_files:
            filename = ntpath.basename(attach_file)
            with open(attach_file, 'rb') as f:
                attachment = MIMEApplication(f.read())
                attachment.add_header('Content-Disposition', 'attachment', filename=filename)
                msg.attach(attachment)
            
    # 로그인
    s = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    s.starttls()
    s.login(SMTP_ID, SMTP_PW)

    # 메일 전송
    for recipient in to:
        s.sendmail(msg['From'], recipient, msg.as_string())
        print(f"* Sended to {recipient}")