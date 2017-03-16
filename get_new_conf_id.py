import csv
import pdb
import os
import sys

from learn import load_data


def get_data_path(name):
    return os.path.join('data', name + '.csv')


def get_conf_path(name):
    return os.path.join('data', 'configs_to_check', name + '.csv')


def load_conf(path):
    with open(path, 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        rows = list(spamreader)
    return rows


sel_cols = [
    "model",
    "CPU_id",
    "DISPLAY_id",
    "MEM_id",
    "HDD_id",
    "SHDD_id",
    "GPU_id",
    "WNET_id",
    "ODD_id",
    "MDB_id",
    "CHASSIS_id",
    "ACUM_id",
    "WAR_id",
    "SIST_id",
]


def load_conf_data(path_data, path_conf):
    rows = load_conf(path_conf)
    data = load_data(path_data)
    conf = [int(row[1]) for row in rows]
    data = data[data.id.isin(conf)]
    data = [tuple(r) for r in data[sel_cols].values]
    return rows, data


REFERENCE_NAMES = [
    'pricing-21-02-2017',
    'pricing-22-02-2017',
    'pricing-24-02-2017',
    'pricing-25-02-2017',
    'pricing-28-02-2017',
    'pricing-01-03-2017',
    'pricing-02-03-2017',
    'pricing-06-03-2017',
    'pricing-08-03-2017',
    'pricing-09-03-2017',
    'pricing-09-03-2017-v2',
    'pricing-12-03-2017',
    'pricing-14-03-2017',
]
new_name = sys.argv[1]

data1 = sum([load_conf_data(get_data_path(p), get_conf_path(p))[1] for p in REFERENCE_NAMES], [])
rows2, data2 = load_conf_data(get_data_path(new_name), '/tmp/test_predictions.csv')
sel_rows = [row for row, datum in zip(rows2, data2) if datum not in data1]

with open(get_conf_path(new_name), 'w') as csvfile:
    csvwriter = csv.writer(csvfile, delimiter=',')
    for row in sel_rows:
        csvwriter.writerow(row)
