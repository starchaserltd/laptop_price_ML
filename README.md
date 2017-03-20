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

## Deployement on CentOS

```
sudo yum -y install gcc-c++
```

## Usage

Start and test the prediction web-service:

```
python web_service.py
python test_web_service.py
```

Train new models:

```
python learn.py -t evaluate train -e xgb -f silviu.1 -d pricing-02-03-2017
```

## SQL

Create tables to hold configurations to check:

```
CREATE TABLE configs_to_check (
    id          INT NOT NULL AUTO_INCREMENT,
    model       INT,
    cpu         INT,
    display     INT,
    mem         INT,
    hdd         INT,
    shdd        INT,
    gpu         INT,
    wnet        INT,
    odd         INT,
    mdb         INT,
    chassis     INT,
    acum        INT,
    war         INT,
    sist        INT,
    realprice   FLOAT,
    predprice   FLOAT,
    date        DATE,
    valid       BOOLEAN,
    primary key(id)
);
CREATE TABLE models_to_check (
    id          INT NOT NULL AUTO_INCREMENT,
    model       INT,
    error       FLOAT,
    date        DATE,
    valid       BOOLEAN,
    primary key(id)
);
```
