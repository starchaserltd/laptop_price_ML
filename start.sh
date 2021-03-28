#!/bin/bash
cd /var/www/vault/genconf/noteb-price
source venv3/bin/activate 
python web_service.py &
