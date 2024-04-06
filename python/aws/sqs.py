"""
AWS SQS  메시지를 읽거나 삭제하는 작업을 수행합니다.
"""

# root경로 추가
import os, sys
sys.path.append(os.path.abspath(''))

from lib.python.config import get_config

import boto3

# AWS 키 정보
athena_config = get_config(config_file_name="aws.ini", section="ATHENA")
AWS_ACCESS_KEY_ID = athena_config["access_key"]
AWS_SECRET_ACCESS_KEY = athena_config["secret_key"]

class QueueConnector:

    def __init__(self, queue_name):
        self.queue_client = boto3.client(
                                'sqs',
                                region_name = 'ap-northeast-2',
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        self.queue_name = queue_name
        self.queue_url = self._get_url(queue_name)


    def _get_url(self, queue_name : str) -> str:
        """ 큐 이름을 전달받아 AWS 상의 URL 로 반환합니다.

        Args:
            queue_name (str): 큐이름

        Returns:
            str: 큐 url
        """
        sqs_resource = boto3.resource(
                                'sqs',
                                region_name = 'ap-northeast-2',
                                aws_access_key_id=AWS_ACCESS_KEY_ID,
                                aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        queue_url = sqs_resource.get_queue_by_name(QueueName = queue_name).url

        return queue_url

    def read_message(self) -> dict or None:
        """
        queue_name 의 큐에서 메시지를 한개씩 읽어서 반환합니다. 
            - 읽을 메시지가 없으면 None 을 반환합니다.
            - 읽기만하고 지우진 않습니다.
        
        Returns:
            dict or None:  ``{"body" : body, "message_handle" : message_handle}`` or ``None``

        """

        response = self.queue_client.receive_message(
            QueueUrl = self.queue_url,
            MaxNumberOfMessages = 1
        )

        try:
            body = response['Messages'][0]['Body']
            message_handle = response['Messages'][0]['ReceiptHandle']
        except KeyError:
            return None
        
        return {
            "body" : body,
            "message_handle" : message_handle
        }
    
    def send_queue(self, message : str) -> None:
        """Queue 이름 확인 후 없으면 생성하고, message 를 큐에 전송합니다.

        Args:
            queue_name (str): message 를 전송할 큐 이름
            message (str): message
        """
        print("SEND_QUEUE", message)
        self.queue_client.send_message(QueueUrl=self.queue_url, MessageBody=message)
        return None
    
    def delete_message(self, message_handle: str) -> None:
        """message_handle 로 전달받은 메시지를 삭제합니다.

        Args:
            message_handle (str): 메시지 핸들러

        Returns:
            None: 
        """
        
        self.queue_client.delete_message(
                                QueueUrl = self.queue_url,
                                ReceiptHandle = message_handle)

        return None
    
    def check_empty_sqs(self) -> bool:
        """큐에 메시지가 있는지 확인합니다.

        Returns:
            bool: 메시지가 있으면 True, 없으면 False
        """
        
        response = self.queue_client.get_queue_attributes(
            QueueUrl = self.queue_url,
            AttributeNames = [
                'ApproximateNumberOfMessages',
                'ApproximateNumberOfMessagesNotVisible',
            ]
        )
        
        number_of_approximate_number_of_messages = int(response['Attributes']['ApproximateNumberOfMessages'])
        number_of_approximate_number_of_messages_not_visible = int(response['Attributes']['ApproximateNumberOfMessagesNotVisible'])
        
        if number_of_approximate_number_of_messages == 0 and number_of_approximate_number_of_messages_not_visible == 0:
            return True
        
        return False
