import collections
import csv
import pdb

import numpy as np


def load_data(path):
    model_to_errors = collections.defaultdict(list)
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            model, _, _, _, error = row
            model_to_errors[model].append(float(error))
    return model_to_errors


def process_errors(errors):
    return np.mean(errors), len(errors)


path = '/tmp/test_predictions.csv'
model_to_errors = load_data(path)
model_to_errors = {m: process_errors(xs) for m, xs in model_to_errors.items()}
model_errors = sorted(model_to_errors.items(), key=lambda kv: (kv[1][1], kv[1][0]), reverse=True)

with open('/tmp/difficult_models.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for model, errors in model_errors:
        csvwriter.writerow((model, ) + errors)
