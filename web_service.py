import datetime
import os
import logging
import pdb
import pickle
import re

from flask import (
    Flask,
    request,
)

from functools import partial

from json import dumps

import numpy as np

from learn import (
    load_classifier,
    load_data,
    RidgeEstimator,
)


app = Flask(__name__)


classifier = load_classifier('models/classifier.pickle')


def ids_to_data_frame(ids):
    for i in ids:
        pass


@app.route('/predict', methods=['POST'])
def predict():
    # TODO
    # ids = request.get_json(force=True)['ids']
    # if ids is None or len(ids) == 0:
    #     return "Bad request, header Content-type should be 'binary/octet-stream' ", 400

    # TODO
    # data = ids_to_data_frame(ids)
    data = load_data('data/pricing-01-03-2017.csv')
    predictions = classifier.predict(data)
    predictions = ['{:.2f}'.format(p) for p in predictions]

    json_data = dumps(predictions, indent=4)
    return json_data, 200


if __name__ == '__main__':
    app.run('0.0.0.0', port=6667, debug=True)
