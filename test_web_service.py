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
    {
        "ACUM": 299,
        "CHASSIS": 690,
        "CPU": 318,
        "DISPLAY": 5,
        "GPU": 486,
        "HDD": 29,
        "MDB": 686,
        "MEM": 24,
        "MODEL": 689,
        "ODD": 0,
        "SHDD": 0,
        "SIST": 1,
        "WAR": 2,
        "WNET": 35,
    },
]

# keys = "ACUM,CHASSIS,CPU,DISPLAY,GPU,HDD,MDB,MEM,MODEL,ODD,SHDD,SIST,WAR,WNET"
# keys = keys.split(',')
#
# with open('data/692.csv', 'r') as f:
#     for line in f.readlines():
#         values = map(int, line.split(','))
#         d = dict(zip(keys, values))
#         ids.append(d)

# with open('data/ids.2017-05-10.json', 'r') as f:
#     ids = json.loads(f.read())

response = requests.post(URL, data=json.dumps(dict(ids=ids)))
print(response.status_code)
print(json.dumps(json.loads(response.text), indent=4))
