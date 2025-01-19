import requests
import json

from dotenv import load_dotenv
import os

import pandas as pd

load_dotenv()


        


def retrieve_messages(channelID):
    headers = { 
               'authorization': os.getenv('authorization'),
    }
    with open('lastf.txt', 'r') as f:
        last_message_id = f.read()
        
    try: 
        df = pd.read_csv("discord_messages.csv")
    except: 
        df = pd.DataFrame(columns=['input', 'output'])
    
    for i in range(50):
        params = {}
        if last_message_id:
            params['before'] = last_message_id
        r = requests.get(f'https://discord.com/api/v9/channels/{channelID}/messages', headers=headers, params=params)
        if r.status_code != 200:
            print(f"Failed to fetch messages: {r.status_code} {r.text} on iteration {i}")
            break
        
        messages = json.loads(r.text)
        
        new = pd.DataFrame(columns=['input', 'output'])
        user = False
        inputText = ""
        outputText = ""
        messages = messages[::-1]
        for message in messages:
            if message['author']['username'] == os.getenv('user'):
                outputText += message['content'] + " " # + "<|end_of_text|>\n" does not require endoftext because it happens in the fine tuning
                user = True
            else:
                if user:
                    new_row = pd.DataFrame({"input": [inputText], "output": [outputText]})
                    new = pd.concat([new, new_row], ignore_index=True)
                    inputText = ""
                    outputText = ""
                    user = False
                inputText += message['content'] + " " # + "<|endoftext|>\n"
                
        last_message_id = messages[0]['id']
        df = pd.concat([df, new], ignore_index=True)
    df.to_csv("discord_messages.csv", index=False)
    print(df.shape)
    with open('lastf.txt', 'w') as f:
        f.write(last_message_id)

msg = retrieve_messages(os.getenv('channelID'))