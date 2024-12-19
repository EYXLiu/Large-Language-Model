import requests
import json

from dotenv import load_dotenv
import os
load_dotenv()

def retrieve_messages(channelID):
    headers = { 
               'authorization': os.getenv('authorization'),
    }
    with open('last.txt', 'r') as f:
        last_message_id = f.read()
    
    with open('discord_messages.txt', 'a') as f:
        for i in range(10):
            params = {}
            if last_message_id:
                params['before'] = last_message_id
            r = requests.get(f'https://discord.com/api/v9/channels/{channelID}/messages', headers=headers, params=params)
            if r.status_code != 200:
                print(f"Failed to fetch messages: {r.status_code} {r.text} on iteration {i}")
                break
            
            messages = json.loads(r.text)
            for message in messages:
                f.write(f"{message['content']} <|endoftext|>\n")
            last_message_id = messages[-1]['id']

    with open('last.txt', 'w') as f:
        f.write(last_message_id)

msg = retrieve_messages(1151382718387585034)