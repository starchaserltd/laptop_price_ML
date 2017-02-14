import pdb
import random

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
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
)

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    PolynomialFeatures,
)

from sklearn.svm import SVR


SEED = 1337
np.random.seed(SEED)
random.seed(SEED)


def load_data() -> DataFrame:
    return read_csv('data/pricing.csv', sep='|')


def rel_error(x, y):
    return np.abs(x - y) / x * 100


def mean_rel_error(true_values, estimated_values):
    xs = np.array(true_values)
    ys = np.array(estimated_values)
    return np.mean(rel_error(xs, ys))


def evaluate_fold(classifier, data: DataFrame, i: int, verbose: int=0) -> float:
    idxs = np.arange(len(data))
    tr_idxs, te_idxs = train_test_split(idxs, test_size=300, random_state=i)

    tr_data = data.iloc[tr_idxs]
    te_data = data.iloc[te_idxs]

    classifier.fit(tr_data)

    tr_preds = classifier.predict(tr_data)
    te_preds = classifier.predict(te_data)

    if verbose:
        print_predictions(tr_data, tr_preds, verbose - 1)
        print_predictions(te_data, te_preds, verbose - 1)
        classifier.print_feature_importance()

    tr_error = mean_rel_error(tr_data.realprice, tr_preds)
    te_error = mean_rel_error(te_data.realprice, te_preds)
    return tr_error, te_error


def evaluate(classifier, data: DataFrame, verbose: int=0):
    return zip(*[evaluate_fold(classifier, data, i, verbose) for i in range(3)])


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
    errors = [rel_error(t, p) for t, p in zip(data.realprice, preds)]
    print(describe(errors))
    if verbose:
        print('\n'.join('{:4d} {:10d} {:10.2f} {:10.2f} {:10.2f} {:10.2f}'.format(i, j, t, p, s, e)
            for i, (j, t, p, s, e) in enumerate(zip(np.array(data.id), data.realprice, preds, data.price, errors))
            if e > 100))


def print_results(results: List[float]):
    print('{:5.2f} ± {:.2f}'.format(
        np.mean(results),
        np.std(results) / len(results)), end=' | ')
    for r in results:
        print('{:4.1f}'.format(r), end=' ')
    print()


FEATURE_NAMES = {
    'numeric.1': [
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
    'prices': [
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
        "CHASSIS_rating",
        "CHASSIS_weight",
        "CHASSIS_thic",
    ],
    'mdb+chassis': [
        "MDB_rating",
        "CHASSIS_thic",
        "CHASSIS_depth",
        "CHASSIS_width",
        "CHASSIS_weight",
        "CHASSIS_made",
        "CHASSIS_pi",
        "CHASSIS_msc",
        "CHASSIS_vi",
    ],
}


class Estimator:

    def _select_features(self, data_frame):
        return np.array(data_frame[self.feature_names_])

    def _select_targets(self, data_frame):
        return np.array(data_frame.realprice).astype(np.float)


class PrecomputedEstimator:

    def fit(self, data_frame: DataFrame):
        return self

    def predict(self, data_frame):
        return data_frame.price


class SklearnEstimator(Estimator):

    def __init__(self, feature_type):
        self.feature_names_ = FEATURE_NAMES[feature_type]
        # Estimators
        # self.estimator_ = Ridge(alpha=0.1)
        # self.estimator_ = KernelRidge(alpha=0.1, kernel='rbf', gamma=0.05)
        # self.estimator_ = KernelRidge(alpha=10, kernel='polynomial', degree=2)
        # self.estimator_ = SVR(C=5000, kernel='rbf', gamma=0.05)
        self.estimator_ = AdaBoostRegressor(DecisionTreeRegressor(max_depth=16), n_estimators=200, loss='linear')

        # Preprocessing
        self.scaler_ = StandardScaler()
        self.poly_ = PolynomialFeatures()

    def fit(self, data_frame):
        X = self._select_features(data_frame)
        y = self._select_targets(data_frame)
        X = self.scaler_.fit_transform(X)
        # X = self.poly_.fit_transform(X)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame)
        X = self.scaler_.transform(X)
        # X = self.poly_.transform(X)
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        feat_imp = zip(self.feature_names_, self.estimator_.feature_importances_)
        feat_imp = sorted(feat_imp, key=lambda t: t[1], reverse=True)
        for feat, imp in feat_imp:
            print('{:18s} {:.3f}'.format(feat, imp))


class AdditivePricesEsimator(Estimator):

    def __init__(self):
        self.feature_names_ = FEATURE_NAMES['prices']
        self.estimator_ = Ridge(alpha=10)
        self.one_hot_encoder_ = OneHotEncoder(n_values=12, sparse=False)
        self.models_ = [
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
            "Razer",
            "Samsung",
        ]
        self.model_to_id_ = {m: i for i, m in enumerate(self.models_)}

    def _select_features(self, data_frame, stage):
        ratings = np.array(data_frame[["MDB_rating", "CHASSIS_rating"]])
        models = [self.model_to_id_[m] for m in data_frame["model_prod"]]
        models = np.atleast_2d(models).T
        if stage == 'train':
            self.one_hot_encoder_.fit(models)
        models = self.one_hot_encoder_.transform(models)
        return np.hstack([
            np.array(data_frame[self.feature_names_]),
            np.array(data_frame[self.feature_names_]) ** 2,
            # np.array(data_frame[self.feature_names_]) ** 3,
            ratings,
            ratings ** 2,
            models,
        ])

    def fit(self, data_frame):
        X = self._select_features(data_frame, 'train')
        y = self._select_targets(data_frame)
        return self.estimator_.fit(X, y)

    def predict(self, data_frame):
        X = self._select_features(data_frame, 'test')
        return self.estimator_.predict(X)

    def print_feature_importance(self):
        feature_names_ = (
            self.feature_names_
            + [
                "MDB_rating",
                "CHASSIS_rating",
                "MDB_rating ** 2",
                "CHASSIS_rating ** 2",
            ]
            + [f + '**2' for f  in self.feature_names_]
            # + [f + '**3' for f  in self.feature_names_]
            + self.models_
        )
        assert len(feature_names_) == len(self.estimator_.coef_)
        feat_imp = zip(feature_names_, self.estimator_.coef_)
        feat_imp = sorted(feat_imp, key=lambda t: t[1], reverse=True)
        for feat, imp in feat_imp:
            print('{:20s} {:+8.3f}'.format(feat, imp))
        print('{:20s} {:+8.3f}'.format("bias", self.estimator_.intercept_))


def main():
    # classifier = PrecomputedEstimator()
    classifier = SklearnEstimator('prices')
    # classifier = AdditivePricesEsimator()
    data = load_data()
    tr_errors, te_errors = evaluate(classifier, data, 2)

    print('Tr:', end=' ')
    print_results(tr_errors)

    print('Te:', end=' ')
    print_results(te_errors)


if __name__ == '__main__':
    main()
