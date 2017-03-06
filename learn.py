import argparse
import datetime
import dill as pickle  # type: ignore
import functools
import json
import os
import pdb
import random
import re
import sys
import warnings

from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
)

import numpy as np  # type: ignore

from pandas import (  # type: ignore
    DataFrame,
    read_csv,
)

from scipy.stats import describe

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
)

from sklearn.model_selection import (
    train_test_split,
    KFold,
    RandomizedSearchCV,
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures,
)

from sklearn.svm import SVR

from xgboost import XGBRegressor


SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


def rm(path):
    if os.path.exists(path):
        os.remove(path)


def remove_ext(filename):
    name, _ = os.path.splitext(filename)
    return name


def load_data(path: str) -> DataFrame:
    dateparse = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
    dataframe = read_csv(
        path,
        sep='|',
        parse_dates=['CPU_ldate', 'GPU_ldate'],
        date_parser=dateparse,
    )
    dataframe.fillna('', inplace=True)
    return dataframe


def select_targets(data_frame):
    return np.array(data_frame.realprice).astype(np.float)
    # return np.array(data_frame.realprice - data_frame.price).astype(np.float)


def rel_error(x, y):
    return np.abs(x - y) / x * 100


def mean_rel_error(true_values, estimated_values):
    xs = np.array(true_values)
    ys = np.array(estimated_values)
    return np.mean(rel_error(xs, ys))


def scorer(estimator, X, y):
    # return sum(rel_error(y, estimator.predict(X)) < 30) / len(y)
    return 100 - mean_rel_error(y, estimator.predict(X))


def evaluate_fold(classifier, data: DataFrame, idxs: Any, verbose: int=0) -> Tuple[float, float]:
    # idxs = np.arange(len(data))
    # tr_idxs, te_idxs = train_test_split(idxs, test_size=300, random_state=i)
    tr_idxs, te_idxs = idxs

    tr_data = data.iloc[tr_idxs]
    te_data = data.iloc[te_idxs]

    classifier.fit(tr_data)

    tr_preds = classifier.predict(tr_data)
    te_preds = classifier.predict(te_data)

    if verbose:
        print_predictions(tr_data, tr_preds, verbose - 1)
        print_predictions(te_data, te_preds, verbose - 1)
        save_predictions(te_data, te_preds)
        classifier.print_feature_importance()

    tr_error = mean_rel_error(select_targets(tr_data), tr_preds)
    te_error = mean_rel_error(select_targets(te_data), te_preds)
    return tr_error, te_error


def evaluate(classifier, data: DataFrame, verbose: int=0):
    results = []
    kf = KFold(n_splits=6, shuffle=True, random_state=SEED)
    for idxs in kf.split(np.arange(len(data))):
        results.append(evaluate_fold(classifier, data, idxs, verbose))
    return zip(*results)
    # return zip(*[evaluate_fold(classifier, data, i, verbose) for i in range(3)])


def print_model(data, model_id, classifier):
    sample = data[data.id.isin([model_id])]
    X = classifier._select_features(sample, 'test')
    price_per_component = (X * classifier.estimator_.coef_)[0, :len(classifier.feature_names_)]
    print(
        price_per_component.argmax(),
        price_per_component.max(),
        sample.model_prod,
    )


def print_predictions(data, preds, verbose):
    relative_errors = rel_error(select_targets(data), preds)
    absolute_errors = np.abs(select_targets(data) - preds)
    print(describe(relative_errors))
    print('Number of configurations with error greater than 10%')
    print(sum(relative_errors > 10))
    print(sum(relative_errors > 20))
    print(sum(relative_errors > 30))
    if verbose:
        print('\n'.join('{:4d} {:10d} {:10d} {:10.2f} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(i, id_, model, r_price, p_price, pred, rel_e, abs_e)
            for i, (id_, model, r_price, p_price, pred, rel_e, abs_e) in enumerate(zip(np.array(data.id), data.model, data.realprice, data.price, preds, relative_errors, absolute_errors))
            if rel_e > 30))
            # ))
        # print('\n'.join('{:4d} {:10d} {:10.2f} {:10.2f} {:32s} {:10.2f}'.format(i, j, t, p, s, e)
        #    for i, (j, t, p, s, e) in enumerate(zip(np.array(data.id), data.realprice - data.price, preds, data.CHASSIS_made, errors))
        #    if e > 10))


def save_predictions(data, preds):
    relative_errors = rel_error(select_targets(data), preds)
    with open('/tmp/test_predictions.csv', 'a') as f:
        for id_, model, r_price, pred, rel_e in zip(np.array(data.id), data.model, data.realprice, preds, relative_errors):
            if rel_e <= 30:
                continue
            f.write('{:d},{:d},{:.2f},{:.2f},{:.2f}\n'.format(model, id_, r_price, pred, rel_e))


def print_results(results: List[float]):
    print('{:5.2f} ± {:.2f}'.format(
        np.mean(results),
        np.std(results) / len(results)), end=' | ')
    for r in results:
        print('{:4.1f}'.format(r), end=' ')
    print()


def extract_n_antennas(args):
    n = len(args[0])
    n_antennas = np.zeros(n)
    for i, line in enumerate(args[0]):
        matches = re.findall('([0-9]+) x antennas', line)
        n_antennas[i] = matches[0] if matches else 0
    return n_antennas


def wrap_key_error(func):
    @functools.wraps(func)
    def func_wrapper(self, value):
        try:
            return func(self, value)
        except KeyError:
            print("WARN Missing value {} for field {}. Available values: {}".format(
                value,
                self.name,
                ', '.join(self.values),
            ))
            return []
    return func_wrapper


class ProcessedFeatures:

    def __init__(self, name, selected, func):
        self.feature_names_ = [name]
        self.selected_names_ = selected
        self.func_ = func

    def __call__(self, data_frame):
        r = self.func_([np.array(data_frame[n]) for n in self.selected_names_])
        return np.atleast_2d(r).T


class LaunchDateFeatures:

    def __init__(self, name):
        self.feature_names_ = [name]
        self.name_ = name

    def __call__(self, data_frame):
        c = data_frame[self.name_]
        d = (datetime.date.today() - c.dt.date).dt.days
        d = np.array(d).astype(np.float)
        return np.atleast_2d(d).T


class SubsetFeatures:

    def __init__(self, feature_names):
        self.feature_names_ = feature_names

    def __call__(self, data_frame):
        return np.array(data_frame[self.feature_names_])


class ChassisMadeTransformer:

    def __init__(self):
        self.name = 'CHASSIS_made'
        self.text_to_value_ = {
            "Aluminium": "aluminium",
            "Anodized aluminium": "aluminium",
            "Carbon fiber": "carbon",
            "Carbon fiber reinforced plastic": "carbon",
            "Glass fiber reinforced plastic": "glass",
            "Magnesium": "magnesium",
            "Magnesium alloy": "magnesium",
            "Magnesium aluminium alloy": "magnesium",
            "Metal": "aluminium",
            "Plastic": "plastic",
            "Rubber": "rubber",
            "Shock-absorbing ultra-polymer": "polymer",
            "Steel reinforcements": "other",
        }
        self.values = sorted(list(set(self.text_to_value_.values())))
        self.value_to_id_ = {v: i for i, v in enumerate(self.values)}

    @wrap_key_error
    def __call__(self, v):
        return [self.value_to_id_[self.text_to_value_[w]] for w in v.split(',')]


class ChassisPiTransformer:

    def __init__(self):
        self.name = 'CHASSIS_pi'
        self.values = [
            'other',
            'usb',
            'lan',
            'rs232',
            'card-reader',
            'docking-port',
            'express-card',
            'external-graphics-port',
            'sim-card',
            'smart-card',
            'thunderbolt',
            'none',
        ]
        self.matchers = [
            lambda t: True,
            lambda t: re.search('USB', t),
            lambda t: re.search('LAN', t),
            lambda t: re.search('RS-232', t),
            lambda t: re.search('card reader', t),
            lambda t: re.search('Docking port', t),
            lambda t: re.search('ExpressCard', t),
            lambda t: re.search('External graphics port', t),
            lambda t: re.search('SIM card', t),
            lambda t: re.search('SmartCard', t),
            lambda t: re.search('Thunderbolt', t) or re.search('OneLink+', t),
            lambda t: t == '',
        ]

    def text_to_ids_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if self.matchers[i](text) :
                n_times = re.findall('^([0-9]+) X ', text)
                n_times = int(n_times[0]) if n_times else 1
                return [i] * n_times
        return [0]

    def __call__(self, v):
        return sum([self.text_to_ids_(w) for w in v.split(',')], [])


class ChassisMscTransformer:

    def __init__(self):
        self.name = 'CHASSIS_msc'
        self.values = [
            'other',
            'premium-speakers',
            'speakers',
            'fingerprint',
            'rear-camera',
            'legacy',
            'stylus',
        ]
        self.matchers = [
            lambda t: True,
            lambda t: any(re.search(s, t) for s in ('JBL', 'Klipsch', 'Bang & Olufsen', 'SonicMaster')),
            lambda t: re.search('speakers', t),
            lambda t: re.search('fingerprint reader', t.lower()),
            lambda t: re.search('Rear camera', t),
            lambda t: re.search('Legacy', t),
            lambda t: re.search('Stylus', t),
        ]

    def text_to_ids_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if self.matchers[i](text) :
                n_times = re.findall('^([0-9]+)[ ]*[xX]', text)
                n_times = int(n_times[0]) if n_times else 1
                return [i] * n_times
        return [0]

    def __call__(self, v):
        return sum([self.text_to_ids_(w) for w in v.split(',')], [])


class ChassisViTransformer:

    def __init__(self):
        self.name = 'CHASSIS_vi'
        self.values = [
            'dp',
            'hdmi',
            'vga',
            'none',
        ]
        self.matchers = [
            lambda t: re.search('DP', t),
            lambda t: re.search('HDMI', t),
            lambda t: re.search('VGA', t),
            lambda t: t == '',
        ]

    def text_to_ids_(self, text):
        for i, v in enumerate(self.values, 0):
            if self.matchers[i](text) :
                n_times = re.findall('^([0-9]+) X ', text)
                n_times = int(n_times[0]) if n_times else 1
                return [i] * n_times
        raise KeyError

    @wrap_key_error
    def __call__(self, v):
        return sum([self.text_to_ids_(w) for w in v.split(',')], [])


class ACUMTipcTransformer:

    def __init__(self):
        self.name = 'ACUM_tipc'
        self.values = ["Li-Ion", "Li-Pol"]
        self.value_to_id_ = {v: i for i, v in enumerate(self.values)}

    @wrap_key_error
    def __call__(self, v):
        return [self.value_to_id_[v]]


class CPUModelTransformer:

    def __init__(self):
        self.name = 'CPU_model'
        self.values = [
            'other',
            'atom',
            'a[0-9]',
            'i[0-9]',
            'm[0-9]',
            'xeon',
            'pentium',
            'celeron',
        ]

    def text_to_id_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if re.match('^{}'.format(v), text.lower()):
                return i
        return 0

    def __call__(self, v):
        return [self.text_to_id_(w) for w in v.split(',')]

class SISTSistTransformer:

    def __init__(self):
        self.name = 'SIST_sist'
        self.values = [
            "Android",
            "Chrome OS",
            "Linux Ubuntu",
            "No OS",
            "Windows",
            "macOS",
        ]

    def text_to_id_(self, text):
        for i, v in enumerate(self.values):
            if re.match('^{}'.format(v), text):
                return i
        raise KeyError

    @wrap_key_error
    def __call__(self, v):
        return [self.text_to_id_(w) for w in v.split(',')]


class MDBNetwTransformer:

    def __init__(self):
        self.name = 'MDB_netw'
        self.values = [
            "Broadcom",
            "Intel",
            "Killer",
            "OEM",
            "Qualcomm",
            "Realtek",
            "NONE",
        ]

    def text_to_id_(self, text):
        for i, v in enumerate(self.values):
            if re.match('^{}'.format(v), text):
                return i
        raise KeyError

    @wrap_key_error
    def __call__(self, v):
        return [self.text_to_id_(w) for w in v.split(',')]


class MDBInterfaceTransformer:

    def __init__(self):
        self.name = 'MDB_interface'
        self.values = [
            'other',
            'MXM',
            'M.2',
            'none',
        ]
        self.matchers = [
            lambda t: True,
            lambda t: re.search('MXM', t),
            lambda t: re.search('M.2', t),
            lambda t: t == '',
        ]

    def text_to_ids_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if self.matchers[i](text) :
                n_times = re.findall('^([0-9]+) X ', text)
                n_times = int(n_times[0]) if n_times else 1
                return [i] * n_times
        return [0]

    def __call__(self, v):
        return sum([self.text_to_ids_(w) for w in v.split(',')], [])


class MDBSubmodelTransformer:

    def __init__(self):
        self.name = 'MDB_submodel'
        self.values = [
            'other',
            'Standard',
            'WWAN',
        ]
        self.matchers = [
            lambda t: True,
            lambda t: re.search('Standard', t),
            lambda t: re.search('WWAN', t),
        ]

    def text_to_ids_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if self.matchers[i](text) :
                n_times = re.findall('^([0-9]+) X ', text)
                n_times = int(n_times[0]) if n_times else 1
                return [i] * n_times
        return [0]

    def __call__(self, v):
        return sum([self.text_to_ids_(w) for w in v.split(',')], [])


class GPUModelTransformer:

    def __init__(self):
        self.name = 'GPU_model'
        self.values = [
            'other',
            'firepro',
            'geforce',
            'quadro',
            'radeon',
            'iris',
            'intel',
        ]

    def text_to_id_(self, text):
        for i, v in enumerate(self.values[1:], 1):
            if re.match('^{}'.format(v), text.lower()):
                return i
        return 0

    def __call__(self, v):
        return [self.text_to_id_(w) for w in v.split(',')]


class ModelProdTransformer:

    def __init__(self):
        self.name = 'model_prod'
        self.values = [
            "Acer",
            "Apple",
            "Asus",
            "Clevo",
            "Dell",
            "Fujitsu",
            "Gigabyte",
            "HP",
            "Lenovo",
            "MSI",
            "Microsoft",
            "Razer",
            "Samsung",
            "Panasonic",
            "VAIO",
        ]
        self.value_to_id_ = {v: i for i, v in enumerate(self.values)}

    @wrap_key_error
    def __call__(self, v):
        return [self.value_to_id_[v]]


class OneHotEncoderFeatures:

    def __init__(self, transformer):
        self.transformer_ = transformer
        self.feature_names_ = [transformer.name + ':' + v for v in transformer.values]

    def __call__(self, data_frame):
        I = [self.transformer_(m) for m in data_frame[self.transformer_.name]]
        X = np.zeros((data_frame.shape[0], len(self.transformer_.values)))
        for i, ids in enumerate(I):
            for j in ids:
                X[i, j] = 1
        return X


class ExtractIP:

    def __init__(self):
        self.name = 'CHASSIS_msc'
        self.values = ['IP']
        self.feature_names_ = [self.name + ':' + v for v in self.values]

    def extract_ip_(self, text):
        for word in text.split(','):
            matches = re.findall('IP(\d+)', text)
            if matches:
                return int(matches[0])
        return 0

    def __call__(self, data_frame):
        X = np.atleast_2d([self.extract_ip_(m) for m in data_frame[self.name]]).T
        return X


SELECT_FEATURES = {
    'numeric.1': [
        SubsetFeatures(
            [
                "CPU_rating",
                "CPU_tdp",
                "GPU_rating",
                "GPU_power",
                "ACUM_rating",
                "CHASSIS_thic",
                "CHASSIS_weight",
                "CHASSIS_rating",
                "DISPLAY_rating",
                "HDD_rating",
                "MDB_rating",
                "MEM_rating",
                "ODD_price",
                "SIST_price",
                "WAR_rating",
                "WNET_rating",
            ],
        ),
    ],
    'prices': [
        SubsetFeatures(
            [
                "CPU_rating",
                "GPU_rating",
                "CPU_price",
                "GPU_price",
                "ACUM_price",
                "DISPLAY_price",
                "HDD_price",
                "MEM_price",
                "ODD_price",
                "SIST_price",
                "WAR_price",
                "WNET_price",
                "MDB_rating",
                # "CHASSIS_rating",
                "CHASSIS_width",
                "CHASSIS_weight",
                "CHASSIS_thic",
            ],
        ),
        OneHotEncoderFeatures(ModelProdTransformer()),
    ],
    'mdb+chassis': [
        SubsetFeatures(
            [
                # "price",
                # "CHASSIS_rating",
                "MDB_rating",
                "CHASSIS_thic",
                "CHASSIS_depth",
                "CHASSIS_width",
                "CHASSIS_weight",
                # "CHASSIS_msc",
                # "CHASSIS_vi",
            ],
        ),
        OneHotEncoderFeatures(ChassisMadeTransformer()),
        # OneHotEncoderFeatures("CHASSIS_pi"),
    ],
    'numeric.1': [
        SubsetFeatures(
            [
                "CPU_rating",
                "CPU_tdp",
                "GPU_rating",
                "GPU_power",
                "ACUM_rating",
                "CHASSIS_thic",
                "CHASSIS_weight",
                "CHASSIS_rating",
                "DISPLAY_rating",
                "HDD_rating",
                "MDB_rating",
                "MEM_rating",
                "ODD_price",
                "SIST_price",
                "WAR_rating",
                "WNET_rating",
            ],
        ),
    ],
    'silviu.1': [
        SubsetFeatures(
            [
                "CPU_rating",
                "CPU_tdp",
                "CPU_price",
                "DISPLAY_size",
                "DISPLAY_touch",
                "MEM_cap",
                "MEM_volt",
                "HDD_cap",
                "HDD_readspeed",
                "HDD_writes",
                "SHDD_cap",
                "SHDD_readspeed",
                "SHDD_writes",
                "GPU_rating",
                "GPU_power",
                # "GPU_price",
                "WNET_speed",
                "ODD_price",
                "ACUM_cap",
                "WAR_years",
                "WAR_typewar",
                "SIST_price",
                "CHASSIS_thic",
                "CHASSIS_depth",
                "CHASSIS_width",
                "CHASSIS_weight",
                "MDB_rating",
            ],
        ),
        ProcessedFeatures(
            name="DISPLAY_hres*vres",
            selected=["DISPLAY_hres", "DISPLAY_vres"],
            func=lambda xs: xs[0] * xs[1],
        ),
        ProcessedFeatures(
            name="MEM_freq/lat",
            selected=["MEM_freq", "MEM_lat"],
            func=lambda xs: xs[0] / xs[1],
        ),
        ProcessedFeatures(
            name="WNET_n_antennas",
            selected=["WNET_msc"],
            func=extract_n_antennas,
        ),
        OneHotEncoderFeatures(CPUModelTransformer()),
        OneHotEncoderFeatures(GPUModelTransformer()),
        OneHotEncoderFeatures(ACUMTipcTransformer()),
        OneHotEncoderFeatures(ChassisMadeTransformer()),
        OneHotEncoderFeatures(ChassisPiTransformer()),
        OneHotEncoderFeatures(ChassisViTransformer()),
        OneHotEncoderFeatures(ChassisMscTransformer()),
        OneHotEncoderFeatures(MDBNetwTransformer()),
        OneHotEncoderFeatures(MDBInterfaceTransformer()),
        OneHotEncoderFeatures(MDBSubmodelTransformer()),
        OneHotEncoderFeatures(SISTSistTransformer()),
        OneHotEncoderFeatures(ModelProdTransformer()),
        ExtractIP(),
        LaunchDateFeatures("CPU_ldate"),
        LaunchDateFeatures("GPU_ldate"),
    ],
    'prices': [
        SubsetFeatures(
            [
                "CPU_rating",
                "GPU_rating",
                "CPU_price",
                "GPU_price",
                "ACUM_price",
                "DISPLAY_price",
                "HDD_price",
                "MEM_price",
                "ODD_price",
                "SIST_price",
                "WAR_price",
                "WNET_price",
                "MDB_rating",
                # "CHASSIS_rating",
                "CHASSIS_width",
                "CHASSIS_weight",
                "CHASSIS_thic",
            ],
        ),
        OneHotEncoderFeatures(ModelProdTransformer()),
    ],
    'mdb+chassis': [
        SubsetFeatures(
            [
                # "price",
                # "CHASSIS_rating",
                "MDB_rating",
                "CHASSIS_thic",
                "CHASSIS_depth",
                "CHASSIS_width",
                "CHASSIS_weight",
                # "CHASSIS_msc",
                # "CHASSIS_vi",
            ],
        ),
        OneHotEncoderFeatures(ChassisMadeTransformer()),
        # OneHotEncoderFeatures("CHASSIS_pi"),
    ],
}


class Estimator:

    # Estimators
    # estimator_ = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.05)
    # estimator_ = KernelRidge(alpha=10, kernel='polynomial', degree=2)

    def _select_features(self, data_frame):
        return np.hstack([
            select_features(data_frame)
            for select_features in self.select_features_list_
        ])

    def _select_targets(self, data_frame):
        return select_targets(data_frame)

    def print_feature_importance(self):
        pass


class PrecomputedEstimator(Estimator):

    def fit(self, data_frame: DataFrame):
        return self

    def predict(self, data_frame):
        return data_frame.price


class SVREstimator(Estimator):

    def __init__(self, select_features_list):
        self.select_features_list_ = select_features_list

        estimator_ = SVR(C=5000, kernel='rbf', gamma=0.05)
        param_dist = {
            "C": [0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 51.2, 102.4, 204.8],
            "gamma": [0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28],
        }

        # run randomized search
        n_iter_search = 32
        self.estimator_ = RandomizedSearchCV(
            estimator_,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring=scorer,
            n_jobs=3,
            verbose=1,
        )

        # Preprocessing
        self.scaler_ = StandardScaler()

    def fit(self, data_frame):
        X = self._select_features(data_frame)
        y = self._select_targets(data_frame)
        X = self.scaler_.fit_transform(X)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame)
        X = self.scaler_.transform(X)
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        print(json.dumps(self.estimator_.best_params_, indent=True, sort_keys=True))


class AdaboostEstimator(Estimator):

    def __init__(self, select_features_list):
        self.select_features_list_ = select_features_list

        estimator_ = AdaBoostRegressor(
            DecisionTreeRegressor(max_depth=8),
            n_estimators=100,
            loss='linear',
        )
        param_dist = {
            "base_estimator__max_depth": [4, 8, 16, 32, 64],
            "base_estimator__splitter": ["best", "random"],
            "base_estimator__max_features": ["auto", "sqrt", "log2"],
            "n_estimators": [64, 128, 256, 512, 1024],
        }

        # run randomized search
        n_iter_search = 32
        self.estimator_ = RandomizedSearchCV(
            estimator_,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring=scorer,
            n_jobs=3,
            verbose=1,
        )

        # Preprocessing
        self.scaler_ = StandardScaler()

    def fit(self, data_frame):
        X = self._select_features(data_frame)
        y = self._select_targets(data_frame)
        X = self.scaler_.fit_transform(X)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame)
        X = self.scaler_.transform(X)
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        estimator = self.estimator_.best_estimator_
        feature_names = sum([s.feature_names_ for s in self.select_features_list_], [])
        feat_imp = zip(feature_names, estimator.feature_importances_)
        feat_imp = sorted(feat_imp, key=lambda t: t[1], reverse=True)
        for feat, imp in feat_imp:
            print('{:40s} {:.3f}'.format(feat, imp))
        print(json.dumps(self.estimator_.best_params_, indent=True, sort_keys=True))


class RidgeEstimator(Estimator):

    def __init__(self, select_features_list):
        self.select_features_list_ = select_features_list
        self.estimator_ = Ridge(alpha=100)
        # Preprocessing
        self.scaler_ = StandardScaler()
        self.poly_ = PolynomialFeatures(degree=2)

    def fit(self, data_frame):
        X = self._select_features(data_frame)
        X = self.scaler_.fit_transform(X)
        X = self.poly_.fit_transform(X)
        y = self._select_targets(data_frame)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame)
        X = self.scaler_.transform(X)
        X = self.poly_.transform(X)
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        return
        feature_names = sum([s.feature_names_ for s in self.select_features_list_], [])
        assert len(feature_names) == len(self.estimator_.coef_)
        feat_imp = zip(feature_names, self.estimator_.coef_)
        feat_imp = sorted(feat_imp, key=lambda t: t[1], reverse=True)
        for feat, imp in feat_imp:
            print('{:40s} {:+8.3f}'.format(feat, imp))
        print('{:40s} {:+8.3f}'.format("bias", self.estimator_.intercept_))


class XGBoostEstimator(Estimator):

    def __init__(self, select_features_list):
        self.select_features_list_ = select_features_list

        estimator_ = XGBRegressor()
        param_dist = {
            "max_depth": [4, 8, 16, 32],
            "learning_rate": [0.01, 0.1, 1.0],
            "n_estimators": [16, 32, 64, 128, 256, 512],
        }

        n_iter_search = 32
        self.estimator_ = RandomizedSearchCV(
            estimator_,
            param_distributions=param_dist,
            n_iter=n_iter_search,
            scoring=scorer,
            n_jobs=1,
            verbose=1,
        )

    def fit(self, data_frame):
        X = self._select_features(data_frame)
        y = self._select_targets(data_frame)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame)
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        estimator = self.estimator_.best_estimator_
        feature_names = sum([s.feature_names_ for s in self.select_features_list_], [])
        feat_imp = zip(feature_names, estimator.feature_importances_)
        feat_imp = sorted(feat_imp, key=lambda t: t[1], reverse=True)
        for feat, imp in feat_imp:
            print('{:40s} {:.6f}'.format(feat, imp))
        print(self.estimator_.best_score_)
        print(json.dumps(self.estimator_.best_params_, indent=True, sort_keys=True))


GET_ESTIMATOR = {
    'baseline': PrecomputedEstimator,
    'adaboost': AdaboostEstimator,
    'ridge': RidgeEstimator,
    'svr': SVREstimator,
    'xgb': XGBoostEstimator,
}

def save_classifier(path, classifier):
    with open(path, 'wb') as f:
        pickle.dump(classifier, f)


def load_classifier(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    parser = argparse.ArgumentParser(description='Learn a predictive function for price estimation.')
    parser.add_argument(
        '-t', '--todo',
        choices=['evaluate', 'train'],
        default=[],
        nargs='+',
        help='what to do',
    )
    parser.add_argument(
        '-e', '--estimator',
        choices=GET_ESTIMATOR.keys(),
        help='type of estimator',
    )
    parser.add_argument(
        '-f', '--features',
        choices=SELECT_FEATURES.keys(),
        help='type of features',
    )
    parser.add_argument(
        '-d', '--data',
        choices=[remove_ext(f) for f in os.listdir('data')],
        help='name of CSV data file',
    )
    args = parser.parse_args()

    rm('/tmp/test_predictions.csv')
    classifier = GET_ESTIMATOR[args.estimator](SELECT_FEATURES[args.features])
    data = load_data('data/{}.csv'.format(args.data))

    if 'evaluate' in args.todo:
        tr_errors, te_errors = evaluate(classifier, data, 2)

        print('Tr:', end=' ')
        print_results(tr_errors)

        print('Te:', end=' ')
        print_results(te_errors)

    if 'train' in args.todo:
        classifier.fit(data)
        save_classifier('models/classifier.pickle'.format(), classifier)


if __name__ == '__main__':
    main()
