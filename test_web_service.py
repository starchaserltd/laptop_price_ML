import json
import os
import pdb
import requests

URL = 'http://localhost:6667/predict'

ids = [
    {
        "ACUM": 17,
        "CHASSIS": 21,
        "CPU": 320,
        "DISPLAY": 63,
        "GPU": 388,
        "HDD": 8,
        "MODEL": 1,
        "MDB": 18,
        "MEM": 24,
        "ODD": 0,
        "SIST": 1,
        "SHDD": 0,
        "WAR": 1,
        "WNET": 35,
    },
    {
        "ACUM": 26,
        "CHASSIS": 38,
        "CPU": 136,
        "DISPLAY": 71,
        "GPU": 32,
        "HDD": 29,
        "MODEL": 7,
        "MDB": 35,
        "MEM": 24,
        "ODD": 0,
        "SIST": 1,
        "SHDD": 0,
        "WAR": 1,
        "WNET": 41,
    },
    {
        "ACUM": 26,
        "CHASSIS": 38,
        "CPU": 136,
        "DISPLAY": 71,
        "GPU": 32,
        "HDD": 29,
        "MODEL": 7,
        "MDB": 35,
        "MEM": 24,
        "ODD": 0,
        "SIST": 2,
        "SHDD": 0,
        "WAR": 1,
        "WNET": 41,
    },
]

response = requests.post(URL, data=json.dumps(dict(ids=ids)))
print(response.status_code)
print(json.dumps(json.loads(response.text), indent=4))
