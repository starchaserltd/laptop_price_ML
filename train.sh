#!/bin/bash
cd /var/www/vault/genconf/noteb-price
source venv3/bin/activate
#python learn.py -t evaluate train -e xgb -f silviu.1
python learn.py -t train -e xgb -f silviu.1 --query-limit=145000 --random-select=1
#previous 210000
service noteb-price restart

