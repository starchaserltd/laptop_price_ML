import json
import os
import pdb
import requests

URL = 'http://localhost:6667/predict'

ids = [
    [],
    [],
]

response = requests.post(URL, data=json.dumps(dict(ids=ids)))
print(json.dumps(json.loads(response.text), indent=4))
