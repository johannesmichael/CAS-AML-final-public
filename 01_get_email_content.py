import requests
from datetime import datetime, timezone
import msal
import json
import os
import re
import argparse
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.data.tables import TableServiceClient, TableEntity

import html2text
from email.mime.text import MIMEText
from email.mime.multipart import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.parser import Parser
from email.utils import parseaddr
import logging
from timeit import default_timer

START_TIME = default_timer()

# Create a logger for successful operations
success_logger = logging.getLogger('log/success')
success_logger.setLevel(logging.INFO)
success_handler = logging.FileHandler('log/uccess.log')
success_handler.setLevel(logging.INFO)
success_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
success_handler.setFormatter(success_formatter)
success_logger.addHandler(success_handler)

# Create a logger for failed operations
failure_logger = logging.getLogger('log/failure')
failure_logger.setLevel(logging.ERROR)
failure_handler = logging.FileHandler('log/failure.log')
failure_handler.setLevel(logging.ERROR)
failure_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
failure_handler.setFormatter(failure_formatter)
failure_logger.addHandler(failure_handler)



#define environment variables
OUTLOOK_API_USER_ID = os.environ.get('OUTLOOK_API_USER_ID')
OUTLOOK_API_CLIENT_ID = os.environ.get('OUTLOOK_API_CLIENT_ID')
OUTLOOK_API_SECRET = os.environ.get('OUTLOOK_API_SECRET')
OUTLOOK_CONTENT_CONNECTION_STRING = os.environ.get('OUTLOOK_CONTENT_CONNECTION_STRING')

#config for msal
config = {
    "authority": "https://login.microsoftonline.com/c08b32af-3535-4f63-8a3a-51247cf1f022",
    "client_id": OUTLOOK_API_CLIENT_ID,
    "scope": ["https://graph.microsoft.com/.default"],
    "secret": OUTLOOK_API_SECRET
}


#function to get token
def get_token():
    app = msal.ConfidentialClientApplication(
        config['client_id'], authority=config['authority'], client_credential=config['secret']
    )

    result = None
    result = app.acquire_token_silent(config['scope'], account=None)

    if not result:
        result = app.acquire_token_for_client(scopes=config['scope'])

    if "access_token" in result:
        access_token = result['access_token']
        return access_token

#function to get folder ids by name
def get_folder_id(folder_name, access_token, user_id):

    # Set the URL and headers
    folder_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders"
    headers = {'Authorization': f"Bearer {access_token}", 'Content-Type': "application/json"}

    response = requests.get(folder_url, headers=headers)

    # Collect the ids of the folders
    folder_ids = { folder['displayName']: folder['id'] for folder in response.json()['value']}

    folder_id = folder_ids[folder_name]
    return folder_id


#function to get subfolder ids by name
def get_subfolder_id(subfolder_name, access_token, user_id, folder_id):
    
        # Set the URL and headers
        subfolder_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders/{folder_id}/childFolders"
        headers = {'Authorization': f"Bearer {access_token}", 'Content-Type': "application/json"}
    
        response = requests.get(subfolder_url, headers=headers)
    
        # Collect the ids of the folders
        subfolder_ids = { folder['displayName']: folder['id'] for folder in response.json()['value']}
    
        subfolder_id = subfolder_ids[subfolder_name]
        return subfolder_id

#function to get email ids by daterange
def get_email_ids_by_daterange(start_time, end_time, access_token, user_id, folder_id, subfolder_id):
        
        if subfolder_id is not None:
            # Set the URL and headers
            message_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders/{folder_id}/childFolders/{subfolder_id}/messages"
        else:
            # Set the URL and headers
            message_url = f"https://graph.microsoft.com/v1.0/users/{user_id}/mailFolders/{folder_id}/messages"
        
        headers = {'Authorization': f"Bearer {access_token}"}
        
        # Filter the messages by the receivedDateTime
        params = {
            '$filter': f"receivedDateTime ge {start_time} and receivedDateTime lt {end_time}",
            '$select': 'id, webLink, conversationId, receivedDateTime',
            '$top': '1000'
        }
        
        response = requests.get(message_url, headers=headers, params=params)
        
        message_ids = [(email['id'], email['webLink'],email['conversationId'],email['receivedDateTime'] ) for email in response.json()['value']]
        return message_ids





#class to get email content as MIME, sender, recipient, subject, receivedDateTime, conversationId, webLink

class EmailMime:
    def __init__(self, message_id, token, user_id):
        self.message_id, self.web_link, self.conversation_id, self.received_datetime = message_id
        self.token = token
        self.user_id = user_id
        self.mime = self.fetch_mime()
        self.parse_message()
        self.sender = self.get_sender()
        self.recipients = self.get_recipients()
        self.subject = self.get_subject()
        self.content = self.extract_content()
        

    def fetch_mime(self):
        headers = {'Authorization': f"Bearer {self.token}"}
        url = f"https://graph.microsoft.com/v1.0//users/{self.user_id}/messages/{self.message_id}/$value"
        response = requests.get(url, headers=headers)
        return response.content

    def parse_message(self):
        self.message = Parser().parsestr(self.mime.decode('utf-8'))

    def get_sender(self):
        message = Parser().parsestr(self.mime.decode('utf-8'))
        sender = parseaddr(message['From'])[1]
        return sender

    def get_recipients(self):
        message = Parser().parsestr(self.mime.decode('utf-8'))
        if message['To'] is not None:
            recipients = [parseaddr(recipient)[1] for recipient in message['To'].split(',')]
        else:
            recipients = []
        return recipients


    def get_subject(self):
        message = Parser().parsestr(self.mime.decode('utf-8'))
        conversation_id = message['Subject']
        return conversation_id
    
    #check if part is not attachment and return True
    def is_attachment(self, part):
        if part.get('Content-Disposition') is None:
            return False
        else:
            return True

    def extract_content(self):
        message = Parser().parsestr(self.mime.decode('utf-8'))
        if message.is_multipart():
            for part in message.walk():
                if self.is_attachment(part):
                    continue
                content_type = part.get_content_type()
                if content_type == 'text/plain':
                    charset = part.get_content_charset()
                    content = part.get_payload(decode=True).decode(charset)
                elif content_type == 'text/html':
                    charset = part.get_content_charset()
                    content = html2text.html2text(part.get_payload(decode=True).decode(charset))
        else:
            content_type = message.get_content_type()
            charset = message.get_content_charset()
            if charset is None:
                charset = 'utf-8'
            if content_type == 'text/plain':
                content = message.get_payload(decode=True).decode(charset)
            elif content_type == 'text/html':
                content = html2text.html2text(message.get_payload(decode=True).decode(charset))
        return content
    
    def _clean_content(self, content):
        # Replace multiple newlines with a single newline
        content = re.sub(r'\n+', '\n', content)
        content = re.sub(r'http\S+|www.\S+', '[link removed]', content)# Replace URLs with '[link removed]'
        content = re.sub(r'An:.*?Sie erreichen Ihr PKM unter folgendem Link', '', content, flags=re.DOTALL)
        content = re.sub(r'An:.*?Betreff:', '', content) # remove email headers
        content =re.sub(r'[\w]+@[\.\w]+',"",content) #removing email addresses
        content = re.sub(r'\d{4}\s\w+',"",content) #removing adress
        content = re.sub(r'[PMT:]*\s*\+\d{1,3}\s[(0)]?(?:[()\s]?\d{1,3}){1,10}',"",content) #removing phone numbers

        return content

    def to_dict(self):
        return {
            #'message_id': self.message_id,
            'subject': self.subject,
            'content': self._clean_content(self.content),
            'sender': self.sender,
            'recipients': ', '.join(self.recipients),  # convert list to string
            'received_datetime': str(self.received_datetime),
            'conversation_id': self.conversation_id,
            'web_link': self.web_link
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=4)



    def save_to_azure(self, azure_table):
        entity = self.to_dict()
        # Azure Table Storage needs a PartitionKey and a RowKey
        entity['PartitionKey'] = self.conversation_id
        entity['RowKey'] = self.message_id
        azure_table.create_entity(entity)   


class AzureTable:
    def __init__(self, connection_string=OUTLOOK_CONTENT_CONNECTION_STRING, table_name="outlookjohannes"):
        self.table_name = table_name
        self.table_client = TableServiceClient.from_connection_string(connection_string).get_table_client(table_name)
        

    def create_entity(self, entity):
        try:
            self.table_client.create_entity(entity=entity)
            #print(f"Entity with RowKey:  added to table: ")
        except Exception as ex:
            print(f"Could not add entity to table: {ex}")

# create the top-level parser
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", help="start_date")
    parser.add_argument("--end_date", help="end_time")
    parser.add_argument("--subfolder_name", help="subfolder_name", default=None)
    #parser.add_argument("azure_table", help="azure_table", default=AzureTable())
    return parser.parse_args()


#main function to run the script for a specific time range
def main():

    azure_table=AzureTable()
    args = parse_args()

    start_time = datetime.strptime(args.start_date, "%d-%m-%Y").replace(tzinfo=timezone.utc).isoformat()
    end_time = datetime.strptime(args.end_date, "%d-%m-%Y").replace(tzinfo=timezone.utc).isoformat()

    user_id = OUTLOOK_API_USER_ID
    # Get the access token
    access_token = get_token()

    # Get the folder id for "Posteingang"
    # TODO: Make this configurable
    folder_id = get_folder_id("Posteingang", access_token, user_id)

    # Get the subfolder id if a subfolder name is not none
    if args.subfolder_name is not None:
        subfolder_id = get_subfolder_id(args.subfolder_name, access_token, user_id, folder_id)
    else:
        subfolder_id = None
    # Get the message ids
    message_ids = get_email_ids_by_daterange(start_time, end_time, access_token, user_id, folder_id, subfolder_id)
    print(f"Found {len(message_ids)} emails")
    # Get the email content
    START_TIME = default_timer()
    for message_id in message_ids:
        #print(f"Processing email with id: {message_id}")
        try:
            email = EmailMime(message_id, access_token, user_id)
            email.save_to_azure(azure_table)
            success_logger.info(f"Success upload email with id: {message_id[0]}")
        except:
            #print(f"Could not upload email with id: {message_id}")
            failure_logger.error(f"Could not upload email with id: {message_id}")
    elapsed_time = default_timer() - START_TIME
    completed_at = "{:5.2f}s".format(elapsed_time)
    print(f"completed in {completed_at}")
    print("Done")

if __name__ == "__main__":
    main()

#python 01_get_email_content.py --start_date "01-07-2022" --end_date "31-07-2022"