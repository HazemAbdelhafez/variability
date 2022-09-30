import json
import random
import sys

import requests

_url = "https://hooks.slack.com/services/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"


def send(msg):
    title = f"New Message 2022 15 16"
    # Generating random hex color code
    hex_number = random.randint(1118481, 16777215)
    hex_number = str(hex(hex_number))
    hex_number = '#' + hex_number[2:]

    slack_data = {
        "username": "Job Progress Bot",
        "channel": "#random",
        "attachments": [
            {
                "color": hex_number,
                "fields": [
                    {
                        "title": title,
                        "value": msg,
                        "short": "false",
                    }
                ]
            }
        ]
    }
    byte_length = str(sys.getsizeof(slack_data))
    headers = {'Content-Type': "application/json", 'Content-Length': byte_length}
    response = requests.post(_url, data=json.dumps(slack_data), headers=headers)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)


if __name__ == '__main__':
    send("Test.")
