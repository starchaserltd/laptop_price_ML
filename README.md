## Setup

Install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set database credentials:

```
export DATABASE_URL=xxx
```

Start and test the prediction web-service:

```
python web_service.py
python test_web_service.py
```


## TODO

- [ ] Save classifier to disk
