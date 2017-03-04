import datetime
import json
import os
import logging
import pdb
import pickle
import re
import time

from flask import (
    Flask,
    request,
)

from functools import partial

import numpy as np

from typing import (
    Any,
    Callable,
    Dict,
    List,
)

from learn import (
    load_classifier,
    load_data,
    RidgeEstimator,
)

import pandas

from pandas import (
    read_sql_query,
    read_sql_table,
)

from sqlalchemy.engine import create_engine

from urllib.parse import quote_plus as urlquote



def create_sql_engine(host, port, username, password, db):
    return create_engine("mysql+pymysql://{}:{}@{}:{}/{}".format(username, urlquote(password), host, port, db))


def load_sql_table(table_name, sql_engine):
    # data = read_sql_query("SELECT * FROM {} LIMIT 10".format(table_name), sql_engine)
    data = read_sql_table(table_name, sql_engine)
    data = data.set_index('id')
    return data


TABLE_NAMES = {
    "ACUM",
    "CHASSIS",
    "CPU",
    "DISPLAY",
    "GPU",
    "HDD",
    "MDB",
    "MEM",
    "MODEL",
    "ODD",
    "SIST",
    "WAR",
    "WNET",
}


ID_TO_TABLE_NAME = {t: t for t in TABLE_NAMES}
ID_TO_TABLE_NAME['SHDD'] = 'HDD'


classifier = load_classifier('models/classifier.pickle')
classifier.nthread = int(os.environ.get('NOTEB_PRICE_NTHREAD', 8))
sql_engine = create_sql_engine(**json.load(open('credentials.json', 'r')).get('database'))
tables = {table_name: load_sql_table(table_name, sql_engine) for table_name in TABLE_NAMES}


def create_column_names():
    handle_special_case = lambda name: name.lower() if name == 'MODEL' else name
    return [
        handle_special_case(i) + '_' + c
        for i in sorted(ID_TO_TABLE_NAME.keys())
        for c in tables[ID_TO_TABLE_NAME[i]].columns.values
        if c != 'id'
    ]


def group_ids(list_of_ids: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    return {t: [ids[t] for ids in list_of_ids] for t in ID_TO_TABLE_NAME.keys()}


def ids_to_data_frame(list_of_ids):
    column_names = create_column_names()
    name_to_ids = group_ids(list_of_ids)
    data_frame = pandas.concat(
        [
            tables[ID_TO_TABLE_NAME[id_]].ix[name_to_ids[id_]].reset_index(drop=True)
            for id_ in sorted(ID_TO_TABLE_NAME.keys())
        ],
        ignore_index=True,
        axis=1,
    )
    data_frame.columns = column_names
    data_frame.fillna('', inplace=True)
    return data_frame


app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    ids = request.get_json(force=True)['ids']
    if ids is None or len(ids) == 0:
        return "Bad request, header Content-type should be 'binary/octet-stream' ", 400

    data = ids_to_data_frame(ids)
    # data = load_data('data/pricing-01-03-2017.csv')
    predictions = classifier.predict(data)
    predictions = ['{:.2f}'.format(p) for p in predictions]

    json_data = json.dumps(predictions, indent=4)
    return json_data, 200


if __name__ == '__main__':
    app.run('0.0.0.0', port=6667, debug=True)
