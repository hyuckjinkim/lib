# (참조) https://stackoverflow.com/questions/3362600/how-to-send-email-attachments

import smtplib
from pathlib import Path
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email.utils import COMMASPACE, formatdate
from email import encoders
import ntpath

# # 예시
# # - password의 APP PASSWORD는 README.md 참조
# send_mail(
#     send_from='E-MAIL ADDRESS',
#     send_to=['E-MAIL ADDRESS'],
#     subject='SUBJECT',
#     message='MESSAGE',
#     files=[FILE_PATH_1,FILE_PATH_2],
#     server='smtp.gmail.com',
#     port=587,
#     username='E-MAIL ADDRESS',
#     password='APP PASSWORD',
#     use_tls=True,
# )
def send_mail(
    send_from, send_to, subject, message, files=[],
    server="localhost", port=587, username='', password='',
    use_tls=True
):
    """Compose and send email with provided info and attachments.

    Args:
        send_from (str): from name
        send_to (list[str]): to name(s)
        subject (str): message title
        message (str): message body
        files (list[str]): list of file paths to be attached to email
        server (str): mail server host name
        port (int): port number
        username (str): server auth username
        password (str): server auth password
        use_tls (bool): use TLS mode
    """
    
    assert isinstance(send_to,list), \
        "send_to must by type list consisting of e-mail."
    assert isinstance(files,list), \
        "files must by type list consisting of file paths"
    
    msg = MIMEMultipart()
    msg['From'] = send_from
    msg['To'] = COMMASPACE.join(send_to)
    msg['Date'] = formatdate(localtime=True)
    msg['Subject'] = subject

    msg.attach(MIMEText(message))

    for path in files:
        #part = MIMEBase('application', "octet-stream")
        #with open(path, 'rb') as file:
        #    part.set_payload(file.read())
        #encoders.encode_base64(part)
        #part.add_header('Content-Disposition',
        #                'attachment; filename={}'.format(Path(path).name))
        #msg.attach(part)
        
        # 파일 추가
        filename = ntpath.basename(path)
        with open(path, 'rb') as etcFD:
            etcPart = MIMEApplication(etcFD.read())
            # 첨부파일의 정보를 헤더로 추가
            etcPart.add_header('Content-Disposition', 'attachment', filename=filename)
            msg.attach(etcPart)

    smtp = smtplib.SMTP(server, port)
    if use_tls:
        smtp.starttls()
    smtp.login(username, password)
    smtp.sendmail(send_from, send_to, msg.as_string())
    smtp.quit()