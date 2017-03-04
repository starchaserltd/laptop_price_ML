## Setup

Install dependencies:

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a JSON file `credentials.json` containing the database credentials:

```
{
    "database": {
        "host": xxx,
        "port": xxx,
        "username": xxx,
        "password": xxx,
        "db": xxx
    }
}
```

Start and test the prediction web-service:

```
python web_service.py
python test_web_service.py
```


## TODO

- [ ] Save classifier to disk
